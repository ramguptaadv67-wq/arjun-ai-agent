"""
core_engine.py
==============
Cloud Database Memory & Brain Orchestration.

Class ``SuperAssistant`` — connects to external scalable databases instead of
local data files:

* **Cloudflare D1** (SQL) — chat history, global knowledge core, creators,
  strategies, audit trail, calendar events.
* **Qdrant Cloud** (Vector) — 3072-dim ``text-embedding-3-large`` vectors
  for semantic similarity search across all learned knowledge.
* **yfinance** — live market data (price, highs, lows, volume, moving
  averages).
* **OpenRouter** — LLM gateway via ``ChatOpenAI`` (base_url pointed at
  OpenRouter) supporting Gemini, Claude, GPT-4o, DeepSeek, Llama.

The ``converse(user_message)`` workflow executes a parallel memory sweep
(pull D1 timeline + Qdrant similarity), builds a system context template,
calls OpenRouter, persists the exchange, and returns the response string.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import yfinance as yf

# LangChain OpenAI integration (used for OpenRouter gateway)
from langchain_openai import ChatOpenAI

# Qdrant remote client
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# OpenAI SDK for embeddings + Whisper + Vision
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QDRANT_COLLECTION_NAME = "multimodal_knowledge"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072

# Cloudflare D1 REST endpoint template
D1_QUERY_URL_TEMPLATE = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}"
    "/d1/database/{database_id}/query"
)

# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Financial keyword regex — triggers yfinance lookup
_FINANCIAL_TICKER_RE = re.compile(
    r"\b(?:stock|share|price|ticker|equity|nifty|sensex|bank nifty|"
    r"reliance|tcs|infosys|tata|sbi|hdfc|icici|bitcoin|ethereum|"
    r"crypto)\b",
    re.IGNORECASE,
)

_TICKER_EXTRACTION_RE = re.compile(
    r"\b(?:reliance|tcs|infosys|infy|tata motors|tata steel|"
    r"sbi|state bank|hdfc|icici|wipro|hcl|lt|bajaj|maruti|"
    r"asian paints|sun pharma| reliance industries)\b",
    re.IGNORECASE,
)

# Table-creation DDL for D1 auto-bootstrap
_D1_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    speaker TEXT NOT NULL,
    message TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS global_knowledge_core (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS creators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    handle TEXT UNIQUE NOT NULL,
    platform TEXT DEFAULT 'instagram',
    reputation_score REAL DEFAULT 0.5,
    accuracy_score REAL DEFAULT 0.5,
    total_reels_watched INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    total_predictions INTEGER DEFAULT 0,
    strengths TEXT,
    weaknesses TEXT,
    created_at TEXT NOT NULL,
    last_crawled TEXT
);

CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    confidence REAL,
    backtest_win_rate REAL,
    backtest_return REAL,
    created_at TEXT NOT NULL,
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS knowledge_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    source_id TEXT,
    content TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    decay_score REAL DEFAULT 1.0,
    timestamp TEXT NOT NULL,
    tags TEXT
);

CREATE TABLE IF NOT EXISTS audit_trail (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    description TEXT,
    reasoning TEXT,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calendar_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    event_date TEXT NOT NULL,
    event_time TEXT,
    description TEXT,
    event_type TEXT DEFAULT 'reminder',
    priority TEXT DEFAULT 'normal',
    status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    prediction TEXT NOT NULL,
    deadline TEXT,
    actual TEXT,
    result TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type TEXT NOT NULL,
    message TEXT NOT NULL,
    priority TEXT DEFAULT 'normal',
    triggered INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    triggered_at TEXT
);
"""


# ---------------------------------------------------------------------------
# SuperAssistant
# ---------------------------------------------------------------------------

class SuperAssistant:
    """Unified persistent cloud memory stack + LLM brain.

    Parameters
    ----------
    model_string : str, optional
        The OpenRouter model string (e.g. ``"google/gemini-pro-1.5"``).
        If not provided, reads from env ``OPENROUTER_MODEL`` or defaults
        to ``"openai/gpt-4o-mini"``.
    confidence_scoring : bool, optional
        Whether to attach confidence scores to knowledge entries.
        Toggle-able from the dashboard.
    knowledge_decay : bool, optional
        Whether old knowledge decays in relevance over time.
        Toggle-able from the dashboard.
    """

    def __init__(
        self,
        model_string: Optional[str] = None,
        confidence_scoring: bool = True,
        knowledge_decay: bool = True,
    ) -> None:
        # --- Environment variables ---
        self.openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
        self.openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
        self.qdrant_url: str = os.environ.get("QDRANT_CLOUD_URL", "")
        self.qdrant_api_key: str = os.environ.get("QDRANT_CLOUD_API_KEY", "")
        self.cf_account_id: str = os.environ.get("CF_ACCOUNT_ID", "")
        self.cf_d1_database_id: str = os.environ.get("CF_D1_DATABASE_ID", "")
        self.cf_api_token: str = os.environ.get("CF_API_TOKEN", "")

        self.model_string: str = (
            model_string
            or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        )
        self.confidence_scoring: bool = confidence_scoring
        self.knowledge_decay: bool = knowledge_decay

        # --- OpenAI client (embeddings, Whisper, Vision) ---
        self.openai_client: OpenAI = OpenAI(api_key=self.openai_api_key)

        # --- Qdrant Cloud client ---
        self.qdrant: Optional[QdrantClient] = None
        if self.qdrant_url and self.qdrant_api_key:
            self.qdrant = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
            self._ensure_qdrant_collection()
        else:
            logger.warning(
                "Qdrant Cloud URL/API key not set — vector search disabled."
            )

        # --- OpenRouter LLM via ChatOpenAI ---
        self.llm: ChatOpenAI = ChatOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.openrouter_api_key,
            model=self.model_string,
            temperature=0.7,
            max_tokens=2000,
        )

        # --- Cloudflare D1 REST endpoint ---
        self.d1_url: str = D1_QUERY_URL_TEMPLATE.format(
            account_id=self.cf_account_id,
            database_id=self.cf_d1_database_id,
        )

        # --- Auto-bootstrap D1 tables on first run ---
        self._ensure_d1_tables()

        # --- Thread pool for parallel memory sweeps ---
        self._executor = ThreadPoolExecutor(max_workers=4)

    # ------------------------------------------------------------------
    # Cloudflare D1 SQL REST integration
    # ------------------------------------------------------------------

    def _d1_headers(self) -> Dict[str, str]:
        """Build the authorization headers for D1 REST calls."""
        return {
            "Authorization": f"Bearer {self.cf_api_token}",
            "Content-Type": "application/json",
        }

    def _d1_query(
        self, sql: str, params: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a SQL query on Cloudflare D1 via REST.

        Returns the ``result.results`` rows array as a list of dicts.
        For INSERT/UPDATE/DELETE the rows list is empty.
        """
        payload: Dict[str, Any] = {"sql": sql}
        if params:
            payload["params"] = params

        resp = requests.post(
            self.d1_url,
            headers=self._d1_headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success"):
            errors = data.get("errors", [])
            raise RuntimeError(f"D1 query failed: {errors}")

        results = data.get("result", [])
        if isinstance(results, list) and len(results) > 0:
            return results[0].get("results", [])
        return []

    def _ensure_d1_tables(self) -> None:
        """Create all required D1 tables if they don't exist."""
        try:
            self._d1_query(_D1_BOOTSTRAP_SQL)
            logger.info("D1 tables ensured.")
        except Exception as exc:
            logger.warning("D1 bootstrap skipped: %s", exc)

    # ---- Chat history (D1) ----

    def push_chat_line(self, speaker: str, message: str) -> None:
        """Push a single conversation line to D1 ``chat_history``."""
        ts = datetime.now(timezone.utc).isoformat()
        self._d1_query(
            "INSERT INTO chat_history (timestamp, speaker, message) "
            "VALUES (?, ?, ?)",
            [ts, speaker, message],
        )

    def retrieve_chat_timeline(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent chat lines in chronological order from D1."""
        rows = self._d1_query(
            "SELECT timestamp, speaker, message FROM chat_history "
            "ORDER BY id DESC LIMIT ?",
            [limit],
        )
        rows.reverse()
        return rows

    # ---- Global knowledge core (D1) ----

    def update_global_knowledge(self, key: str, value: str) -> None:
        """Upsert a key-value pair into ``global_knowledge_core``."""
        ts = datetime.now(timezone.utc).isoformat()
        self._d1_query(
            "INSERT INTO global_knowledge_core (key, value, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?",
            [key, value, ts, value, ts],
        )

    def get_global_knowledge(self, key: str) -> Optional[str]:
        """Retrieve a single value from ``global_knowledge_core``."""
        rows = self._d1_query(
            "SELECT value FROM global_knowledge_core WHERE key = ?",
            [key],
        )
        if rows:
            return rows[0].get("value")
        return None

    # ---- Calendar events (D1) ----

    def create_calendar_event(
        self,
        title: str,
        event_date: str,
        event_time: str = "",
        description: str = "",
        event_type: str = "reminder",
        priority: str = "normal",
    ) -> Dict[str, Any]:
        """Insert a calendar event into D1."""
        ts = datetime.now(timezone.utc).isoformat()
        self._d1_query(
            "INSERT INTO calendar_events "
            "(title, event_date, event_time, description, event_type, "
            "priority, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            [title, event_date, event_time, description, event_type, priority, ts],
        )
        return {"title": title, "date": event_date, "time": event_time}

    def get_upcoming_events(self, days: int = 7) -> List[Dict[str, Any]]:
        """Retrieve upcoming calendar events."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = self._d1_query(
            "SELECT * FROM calendar_events "
            "WHERE event_date >= ? AND status = 'pending' "
            "ORDER BY event_date ASC, event_time ASC LIMIT ?",
            [today, days * 5],
        )
        return rows

    # ---- Knowledge entries (D1) ----

    def push_knowledge_entry(
        self,
        content: str,
        source_type: str,
        source_id: str = "",
        confidence: float = 0.5,
        tags: str = "",
    ) -> None:
        """Push a structured knowledge entry to D1."""
        ts = datetime.now(timezone.utc).isoformat()
        decay = 1.0 if self.knowledge_decay else 1.0
        self._d1_query(
            "INSERT INTO knowledge_entries "
            "(source_type, source_id, content, confidence, decay_score, "
            "timestamp, tags) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [source_type, source_id, content, confidence, decay, ts, tags],
        )

    def retrieve_knowledge_entries(
        self, source_type: str = "", limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve knowledge entries, optionally filtered by source type."""
        if source_type:
            return self._d1_query(
                "SELECT * FROM knowledge_entries "
                "WHERE source_type = ? ORDER BY id DESC LIMIT ?",
                [source_type, limit],
            )
        return self._d1_query(
            "SELECT * FROM knowledge_entries ORDER BY id DESC LIMIT ?",
            [limit],
        )

    # ---- Predictions ledger (D1) ----

    def log_prediction(
        self, source: str, prediction: str, deadline: str = ""
    ) -> None:
        """Log a prediction for later verification."""
        ts = datetime.now(timezone.utc).isoformat()
        self._d1_query(
            "INSERT INTO predictions_ledger "
            "(source, prediction, deadline, created_at) "
            "VALUES (?, ?, ?, ?)",
            [source, prediction, deadline, ts],
        )

    def get_prediction_accuracy(self, source: str = "") -> Dict[str, Any]:
        """Calculate prediction accuracy for a source."""
        if source:
            rows = self._d1_query(
                "SELECT result FROM predictions_ledger WHERE source = ?",
                [source],
            )
        else:
            rows = self._d1_query(
                "SELECT result FROM predictions_ledger",
                [],
            )
        total = len(rows)
        correct = sum(1 for r in rows if r.get("result") == "correct")
        return {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total * 100) if total > 0 else 0,
        }

    # ---- Audit trail (D1) ----

    def log_audit(
        self, event_type: str, description: str, reasoning: str = ""
    ) -> None:
        """Append an entry to the audit trail."""
        ts = datetime.now(timezone.utc).isoformat()
        self._d1_query(
            "INSERT INTO audit_trail (event_type, description, reasoning, timestamp) "
            "VALUES (?, ?, ?, ?)",
            [event_type, description, reasoning, ts],
        )

    # ------------------------------------------------------------------
    # Qdrant Cloud integration
    # ------------------------------------------------------------------

    def _ensure_qdrant_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist."""
        if self.qdrant is None:
            return
        try:
            collections = self.qdrant.get_collections()
            existing = {c.name for c in collections.collections}
            if QDRANT_COLLECTION_NAME not in existing:
                self.qdrant.create_collection(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection '%s' (%d dims).",
                    QDRANT_COLLECTION_NAME,
                    EMBEDDING_DIMENSIONS,
                )
        except Exception as exc:
            logger.warning("Qdrant collection creation skipped: %s", exc)

    def _get_embedding(self, text: str) -> List[float]:
        """Call OpenAI text-embedding-3-large and return the vector."""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def upsert_knowledge_vector(
        self,
        text: str,
        point_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Embed text and upsert the vector into Qdrant with metadata."""
        if self.qdrant is None:
            logger.warning("Qdrant not configured — skipping vector upsert.")
            return
        vector = self._get_embedding(text)
        self.qdrant.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload or {"text": text},
                )
            ],
        )

    def search_similar_knowledge(
        self, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Embed the query and search Qdrant for similar knowledge."""
        if self.qdrant is None:
            return []
        vector = self._get_embedding(query_text)
        hits = self.qdrant.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
        )
        return [
            {"score": hit.score, "payload": hit.payload}
            for hit in hits
        ]

    # ------------------------------------------------------------------
    # yfinance live market data
    # ------------------------------------------------------------------

    def fetch_market_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch live market data for a ticker via yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1mo")
            if hist.empty:
                return {"ticker": ticker, "error": "No historical data."}
            latest = hist.iloc[-1]
            highs = hist["High"].tolist()
            lows = hist["Low"].tolist()
            volumes = hist["Volume"].tolist()
            return {
                "ticker": ticker,
                "name": info.get("longName", ticker),
                "current_price": float(latest["Close"]),
                "day_high": float(latest["High"]),
                "day_low": float(latest["Low"]),
                "volume": int(latest["Volume"]),
                "ma_20": float(hist["Close"].tail(20).mean()),
                "ma_50": float(hist["Close"].tail(50).mean()) if len(hist) >= 50 else None,
                "highs": highs,
                "lows": lows,
                "volumes": volumes,
            }
        except Exception as exc:
            return {"ticker": ticker, "error": str(exc)}

    def extract_ticker_from_text(self, text: str) -> Optional[str]:
        """Extract a stock ticker symbol from free text."""
        match = _TICKER_EXTRACTION_RE.search(text)
        if not match:
            return None
        keyword = match.group(0).lower().strip()
        mapping = {
            "reliance": "RELIANCE.NS",
            "reliance industries": "RELIANCE.NS",
            "tcs": "TCS.NS",
            "infosys": "INFY.NS",
            "infy": "INFY.NS",
            "tata motors": "TATAMOTORS.NS",
            "tata steel": "TATASTEEL.NS",
            "sbi": "SBIN.NS",
            "state bank": "SBIN.NS",
            "hdfc": "HDFCBANK.NS",
            "icici": "ICICIBANK.NS",
            "wipro": "WIPRO.NS",
            "hcl": "HCLTECH.NS",
            "lt": "LT.NS",
            "bajaj": "BAJFINANCE.NS",
            "maruti": "MARUTI.NS",
            "asian paints": "ASIANPAINT.NS",
            "sun pharma": "SUNPHARMA.NS",
        }
        return mapping.get(keyword)

    # ------------------------------------------------------------------
    # Whisper audio transcription
    # ------------------------------------------------------------------

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe an audio file via OpenAI Whisper (multilingual)."""
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                )
            return transcript.text
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            return f""

    # ------------------------------------------------------------------
    # GPT-4o Vision
    # ------------------------------------------------------------------

    def analyze_image(self, image_b64: str) -> str:
        """Send a base64-encoded image to GPT-4o Vision for analysis."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this image in detail. "
                                    "Describe charts, tables, code, "
                                    "and any actionable insights."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1500,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Vision analysis failed: %s", exc)
            return f""

    # ------------------------------------------------------------------
    # Knowledge feeding helpers
    # ------------------------------------------------------------------

    def feed_text_knowledge(
        self,
        text: str,
        source_type: str,
        source_id: str = "",
        confidence: float = 0.5,
        tags: str = "",
    ) -> None:
        """Push a text blob to D1 knowledge entries + Qdrant vector store."""
        self.push_knowledge_entry(
            content=text,
            source_type=source_type,
            source_id=source_id,
            confidence=confidence,
            tags=tags,
        )
        point_id = f"{source_type}_{source_id}_{int(time.time())}"
        self.upsert_knowledge_vector(
            text=text,
            point_id=point_id,
            payload={
                "source_type": source_type,
                "source_id": source_id,
                "text": text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ------------------------------------------------------------------
    # Parallel memory sweep
    # ------------------------------------------------------------------

    def _parallel_memory_sweep(self, user_message: str) -> Dict[str, Any]:
        """Pull D1 chat timeline + Qdrant similarity in parallel."""
        future_timeline = self._executor.submit(self.retrieve_chat_timeline)
        future_similar = self._executor.submit(
            self.search_similar_knowledge, user_message
        )
        future_knowledge = self._executor.submit(
            self.retrieve_knowledge_entries, "", 10
        )
        future_events = self._executor.submit(self.get_upcoming_events)

        return {
            "chat_timeline": future_timeline.result(),
            "similar_knowledge": future_similar.result(),
            "recent_knowledge": future_knowledge.result(),
            "upcoming_events": future_events.result(),
        }

    def _build_system_context(self, user_message: str, memory: Dict[str, Any]) -> str:
        """Build the system prompt template with memory injected."""
        timeline_str = "\n".join(
            f"[{row.get('timestamp', '')}] {row.get('speaker', '')}: "
            f"{row.get('message', '')}"
            for row in memory.get("chat_timeline", [])
        )
        similar_str = "\n".join(
            f"- (score: {hit.get('score', 0):.2f}) "
            f"{(hit.get('payload') or {}).get('text', '')[:200]}"
            for hit in memory.get("similar_knowledge", [])
        )
        knowledge_str = "\n".join(
            f"- [{entry.get('source_type', '')}] {entry.get('content', '')[:200]}"
            for entry in memory.get("recent_knowledge", [])
        )
        events_str = "\n".join(
            f"- {ev.get('event_date', '')} {ev.get('event_time', '')} "
            f"{ev.get('title', '')}"
            for ev in memory.get("upcoming_events", [])
        )

        market_section = ""
        if _FINANCIAL_TICKER_RE.search(user_message):
            ticker = self.extract_ticker_from_text(user_message)
            if ticker:
                market_data = self.fetch_market_data(ticker)
                if "error" not in market_data:
                    market_section = (
                        f"\n\n## Live Market Data ({ticker})\n"
                        f"Current Price: {market_data.get('current_price')}\n"
                        f"Day High: {market_data.get('day_high')}\n"
                        f"Day Low: {market_data.get('day_low')}\n"
                        f"Volume: {market_data.get('volume')}\n"
                        f"20-day MA: {market_data.get('ma_20')}\n"
                    )

        return f"""You are SuperAssistant, a persistent cloud-hybrid AI agent.
You have access to long-term memory stored in Cloudflare D1 (SQL) and
Qdrant Cloud (vector similarity). Use the context below to ground your
response. Be concise, accurate, and helpful.

## Recent Conversation History
{timeline_str or '(none)'}

## Semantically Similar Knowledge
{similar_str or '(none)'}

## Recent Knowledge Entries
{knowledge_str or '(none)'}

## Upcoming Calendar Events
{events_str or '(none)'}
{market_section}
"""

    # ------------------------------------------------------------------
    # Main conversation entrypoint
    # ------------------------------------------------------------------

    def converse(self, user_message: str) -> str:
        """Main conversation workflow.

        1. Parallel memory sweep (D1 timeline + Qdrant similarity).
        2. Build system context template.
        3. Call OpenRouter LLM.
        4. Persist exchange to D1 + Qdrant.
        5. Return the response string.
        """
        # Persist user message
        self.push_chat_line("user", user_message)

        # Parallel memory sweep
        memory = self._parallel_memory_sweep(user_message)

        # Build context
        system_prompt = self._build_system_context(user_message, memory)

        # Call LLM
        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ])
            reply = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            reply = f"(LLM error: {exc})"

        # Persist assistant response
        self.push_chat_line("assistant", reply)

        # Feed the exchange into vector memory for future retrieval
        exchange_text = f"User: {user_message}\nAssistant: {reply}"
        self.upsert_knowledge_vector(
            text=exchange_text,
            point_id=f"exchange_{int(time.time())}",
            payload={
                "source_type": "conversation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": exchange_text,
            },
        )

        # Audit trail
        self.log_audit(
            event_type="conversation",
            description="Processed user message and generated response.",
            reasoning=f"User message: {user_message[:200]}",
        )

        return reply

    # ------------------------------------------------------------------
    # Autonomous debug & repair
    # ------------------------------------------------------------------

    def autonomous_debug_and_repair(
        self, traceback_str: str, original_code: str
    ) -> Dict[str, Any]:
        """Send a traceback + original code to the LLM for auto-repair."""
        prompt = (
            "The following Python code crashed with the traceback below. "
            "Analyze the error and return ONLY the corrected Python code "
            "in a single code block. Do not include explanations.\n\n"
            f"--- ORIGINAL CODE ---\n{original_code}\n\n"
            f"--- TRACEBACK ---\n{traceback_str}\n"
        )
        try:
            response = self.llm.invoke([
                {"role": "system", "content": "You are a Python debugging expert."},
                {"role": "user", "content": prompt},
            ])
            content = response.content if hasattr(response, "content") else str(response)
            # Extract code from markdown fence if present
            match = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
            fixed_code = match.group(1).strip() if match else content.strip()
            return {
                "success": True,
                "fixed_code": fixed_code,
                "raw_response": content,
            }
        except Exception as exc:
            logger.error("Autonomous debug failed: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "fixed_code": None,
            }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean up resources."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
