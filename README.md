# 🧠 Arjun — Cloud-Hybrid Permanent AI Agent

A production-ready, headless-capable AI agent with permanent cloud memory, multimodal ingestion (text/audio/image/code), live financial data, autonomous self-healing code, and a Streamlit dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLOUDFLARE (All Data)                     │
│                                                               │
│  D1 (SQL)          │ R2 (Objects)    │ Vectorize (Vectors)    │
│  • Chat history     │ • Image files   │ • 3072-dim embeddings │
│  • Knowledge core   │ • Audio chunks   │ • Semantic search     │
│  • Creators        │ • User uploads   │ • Similarity match    │
│  • Strategies      │                  │                       │
│  • Audit trail     │                  │                       │
│  • Calendar        │                  │                       │
│  • Predictions     │                  │                       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              RENDER (Python Compute Only)                     │
│                                                               │
│  • Streamlit dashboard (app.py)                              │
│  • SuperAssistant engine (core_engine.py)                    │
│  • Integrity manager (integrity_engine.py)                   │
│  • yt-dlp / Whisper / GPT-4o Vision processing               │
│  • Headless browser for reel watching                        │
│  • Background daemon threads                                  │
│                                                               │
│  NO data stored locally. ALL data → Cloudflare.              │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
├── app.py                  ← Streamlit dashboard (main entry point)
├── core_engine.py          ← SuperAssistant (cloud memory + LLM brain)
├── integrity_engine.py     ← SystemIntegrityManager (security + restore points)
├── requirements.txt        ← Python dependencies
├── Dockerfile              ← Container for Render/Docker deployment
├── .env.example            ← Environment variable template
├── DEPLOYMENT.md           ← Step-by-step deployment guide
├── CF_D1_SETUP.md          ← Cloudflare D1 database setup guide
├── README.md               ← This file
└── LICENSE                 ← MIT License
```

## Features

### Core Engine (`core_engine.py`)
- **Cloudflare D1 SQL** — chat history, global knowledge core, creators, strategies, knowledge entries, audit trail, calendar events, predictions ledger, alerts (9 tables, auto-bootstrapped)
- **Qdrant Cloud** — 3072-dim vector search with `text-embedding-3-large`
- **yfinance** — live market data (price, highs/lows, volume, moving averages)
- **OpenRouter** — multi-model LLM gateway (Gemini, Claude, GPT-4o, DeepSeek, Llama)
- **`converse()` workflow** — parallel memory sweep (D1 + Qdrant) → financial keyword detection → system context build → LLM call → dual-write to D1 + Qdrant
- **Autonomous debug engine** — sends crash tracebacks to OpenRouter, gets fixed code, validates with `compile()`, deploys
- **Whisper transcription** — multilingual (Telugu/Hindi/English/mixed) → English text
- **GPT-4o Vision** — chart/document/image analysis → Markdown

### Integrity Engine (`integrity_engine.py`)
- SHA-256 cryptographic file hashing
- Timestamped restore points with atomic manifest writes
- Tampering detection (runs on every page load)
- Atomic rollback with post-restore verification
- Pre-injection and post-injection baseline snapshots

### Streamlit Dashboard (`app.py`)
- **Sidebar**: model dropdown, confidence scoring toggle, knowledge decay toggle, live tampering status, force rollback button, inject-live-update-code with `compile()` validation + autonomous debug, debug log, env var status
- **Left pane (Ingestion & Tool Core)**:
  - Web document scraper (BeautifulSoup)
  - Audio/video → Whisper multilingual transcription (yt-dlp)
  - GPT-4o Vision image/chart inspector
  - `GitRepositoryManager` — clone repos, install deps, read .py/.md files, expose to LLM context
  - Calendar event creator with alerts
  - Daily intelligence briefing generator
- **Right pane**: Chat terminal with session state, repo context injection
- **Background daemon**: async ingestion queue processing

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/arjun-ai-agent.git
cd arjun-ai-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env with your actual API keys

# 5. Load environment variables
export $(cat .env | xargs)  # Linux/Mac

# 6. Run
streamlit run app.py
```

Access at `http://localhost:8501`

## Environment Variables

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | LLM text/reasoning gateway |
| `OPENAI_API_KEY` | Embeddings (text-embedding-3-large), Whisper, GPT-4o Vision |
| `QDRANT_CLOUD_URL` | Remote vector database URL |
| `QDRANT_CLOUD_API_KEY` | Qdrant Cloud authentication |
| `CF_ACCOUNT_ID` | Cloudflare account ID |
| `CF_D1_DATABASE_ID` | Cloudflare D1 database ID |
| `CF_API_TOKEN` | Cloudflare D1 REST API token |
| `OPENROUTER_MODEL` | (Optional) Default model string |

See `.env.example` for the template.

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for complete step-by-step instructions for:
- Cloudflare D1 setup
- Qdrant Cloud setup
- OpenRouter + OpenAI setup
- Render deployment (Docker or Python direct)
- Local development
- Troubleshooting

## Cloudflare D1 Tables

The system auto-creates 9 tables on first run. See **[CF_D1_SETUP.md](CF_D1_SETUP.md)** for details.

## Requirements

- Python 3.10+
- ffmpeg (for yt-dlp audio extraction)
- All API keys (OpenRouter, OpenAI, Qdrant Cloud, Cloudflare)

## License

MIT License — see [LICENSE](LICENSE).
