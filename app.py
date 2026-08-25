"""
app.py
======
Streamlit Workspace Dashboard — Headless-Capable Cloud-Hybrid Permanent AI Agent.

Layout
------
* Sidebar: Model selector, security panel (tampering check + rollback),
  inject-live-update-code interface, debug log, feature toggles.
* Left pane: Multimodal ingestion — web scraper, audio/video scraper
  (yt-dlp → Whisper multilingual), vision/image inspector, Git repository
  manager.
* Right pane: Chat terminal connected to ``SuperAssistant.converse()``.
* Background: headless daemon thread for async ingestion queue processing.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Conditional imports — the app should still load even if a dependency
# is missing, showing a clear error message instead of a crash.
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

try:
    from core_engine import SuperAssistant
except ImportError:
    SuperAssistant = None

try:
    from integrity_engine import SystemIntegrityManager
except ImportError:
    SystemIntegrityManager = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_MODELS = {
    "Gemini Pro 1.5": "google/gemini-pro-1.5",
    "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "DeepSeek V3": "deepseek/deepseek-chat",
    "Llama 3 70B": "meta-llama/llama-3-70b-instruct",
}

WORKSPACE_DIR = Path("./agent_git_workspace")
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

INGESTION_QUEUE_DIR = Path("./ingestion_queue")
INGESTION_QUEUE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# GitRepositoryManager
# ---------------------------------------------------------------------------

class GitRepositoryManager:
    """Controls local workspace sandbox paths under ``./agent_git_workspace/``.

    Clones GitHub repositories, auto-runs ``pip install`` on
    ``requirements.txt``, reads internal ``.py`` and ``.md`` files, and
    exposes file contents to the OpenRouter system context.
    """

    def __init__(self, workspace_dir: Path = WORKSPACE_DIR) -> None:
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.repo_path: Optional[Path] = None
        self.repo_files: Dict[str, str] = {}

    def clone_repo(self, repo_url: str) -> Dict[str, Any]:
        """Clone a GitHub repository into the workspace sandbox.

        Returns a dict with the clone result.
        """
        repo_name = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        target = self.workspace_dir / repo_name

        if target.exists():
            self.repo_path = target
            return {
                "success": True,
                "message": f"Repository '{repo_name}' already exists at {target}",
                "path": str(target),
            }

        try:
            result = subprocess.run(
                ["git", "clone", repo_url, str(target)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return {
                    "success": False,
                    "message": f"Git clone failed: {result.stderr}",
                    "path": None,
                }
            self.repo_path = target
            return {
                "success": True,
                "message": f"Cloned '{repo_name}' to {target}",
                "path": str(target),
            }
        except Exception as exc:
            return {
                "success": False,
                "message": f"Clone error: {exc}",
                "path": None,
            }

    def install_dependencies(self, repo_path: Optional[Path] = None) -> Dict[str, Any]:
        """Run ``pip install -r requirements.txt`` if one exists."""
        target = repo_path or self.repo_path
        if target is None:
            return {"success": False, "message": "No repository path set."}

        req_file = target / "requirements.txt"
        if not req_file.exists():
            return {"success": True, "message": "No requirements.txt found, skipping."}

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            return {
                "success": result.returncode == 0,
                "message": (
                    "Dependencies installed."
                    if result.returncode == 0
                    else f"pip install failed: {result.stderr}"
                ),
                "stdout": result.stdout[-500:],
                "stderr": result.stderr[-500:],
            }
        except Exception as exc:
            return {"success": False, "message": f"pip install error: {exc}"}

    def read_repo_files(self, repo_path: Optional[Path] = None) -> Dict[str, str]:
        """Read all ``.py`` and ``.md`` files in the repository.

        Returns a dict mapping relative filename → file content.
        """
        target = repo_path or self.repo_path
        if target is None:
            return {}

        files: Dict[str, str] = {}
        for ext in ("*.py", "*.md"):
            for fpath in target.rglob(ext):
                # Skip virtual environments and hidden directories
                parts = fpath.relative_to(target).parts
                if any(
                    p.startswith(".") or p in ("venv", "__pycache__", "node_modules")
                    for p in parts
                ):
                    continue
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    rel = str(fpath.relative_to(target))
                    # Truncate very large files
                    if len(content) > 10000:
                        content = content[:10000] + "\n... (truncated)"
                    files[rel] = content
                except Exception:
                    pass
        self.repo_files = files
        return files

    def expose_to_context(self) -> str:
        """Package file contents into a context payload string.

        This gets injected into the OpenRouter system context on the next
        ``converse()`` call.
        """
        if not self.repo_files:
            return ""
        parts: List[str] = []
        for fname, content in self.repo_files.items():
            parts.append(f"--- FILE: {fname} ---\n{content}\n")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Autonomous Debug Engine
# ---------------------------------------------------------------------------

def execute_injected_code(code: str, assistant: Optional[Any] = None) -> Dict[str, Any]:
    """Execute injected code with an error boundary.

    If the code crashes, the traceback is piped to the OpenRouter model
    via ``assistant.autonomous_debug_and_repair()``. The repaired code is
    then deployed autonomously.

    Returns a dict with execution status, output, and any auto-repair info.
    """
    result: Dict[str, Any] = {
        "executed": False,
        "output": "",
        "error": "",
        "auto_repaired": False,
        "repair_info": None,
    }

    # Step 1: validate syntax with compile()
    try:
        compile(code, "<injected_code>", "exec")
    except SyntaxError as exc:
        result["error"] = f"SyntaxError: {exc}"
        return result

    # Step 2: execute in a try/except boundary
    try:
        local_ns: Dict[str, Any] = {}
        exec(code, {"__name__": "__main__"}, local_ns)
        result["executed"] = True
        result["output"] = str(local_ns)
    except Exception:
        tb = traceback.format_exc()
        result["error"] = tb

        # Step 3: autonomous debug — send to OpenRouter
        if assistant is not None:
            st.session_state.debug_log.append(
                f"[{datetime.now().isoformat()}] Code crashed. "
                "Triggering autonomous debug engine ..."
            )
            repair = assistant.autonomous_debug_and_repair(tb, code)
            result["repair_info"] = repair

            if repair.get("success") and repair.get("fixed_code"):
                fixed = repair["fixed_code"]
                # Validate the fix compiles
                try:
                    compile(fixed, "<autonomous_repair>", "exec")
                    # Execute the fixed code
                    local_ns2: Dict[str, Any] = {}
                    exec(fixed, {"__name__": "__main__"}, local_ns2)
                    result["executed"] = True
                    result["auto_repaired"] = True
                    result["output"] = str(local_ns2)
                    result["error"] = ""
                    st.session_state.debug_log.append(
                        f"[{datetime.now().isoformat()}] Auto-repair SUCCESS. "
                        "Fixed code executed cleanly."
                    )
                    # Log the recovery event
                    if hasattr(assistant, "log_audit"):
                        assistant.log_audit(
                            "auto_repair",
                            "Autonomous debug engine repaired and deployed code.",
                            f"Original error: {tb[:200]}",
                        )
                except Exception as exc2:
                    st.session_state.debug_log.append(
                        f"[{datetime.now().isoformat()}] Auto-repair code "
                        f"failed to execute: {exc2}"
                    )
            else:
                st.session_state.debug_log.append(
                    f"[{datetime.now().isoformat()}] Auto-repair failed: "
                    f"{repair.get('error', 'unknown')}"
                )
        else:
            result["error"] = tb + "\n(No assistant available for auto-repair)"

    return result


def inject_code_to_file(code: str) -> Dict[str, Any]:
    """Append validated code to the running ``app.py`` script file.

    Returns a dict with the injection result.
    """
    app_path = Path("app.py")
    try:
        current_content = app_path.read_text(encoding="utf-8")
    except Exception:
        current_content = ""

    timestamp = datetime.now(timezone.utc).isoformat()
    block = (
        f"\n\n# --- INJECTED CODE [{timestamp}] ---\n"
        f"{code}\n"
        f"# --- END INJECTED CODE [{timestamp}] ---\n"
    )

    try:
        app_path.write_text(current_content + block, encoding="utf-8")
        return {"success": True, "message": "Code injected into app.py"}
    except Exception as exc:
        return {"success": False, "message": f"Injection failed: {exc}"}


# ---------------------------------------------------------------------------
# Web scraper
# ---------------------------------------------------------------------------

def scrape_web_content(url: str) -> str:
    """Fetch a URL and extract visible text via BeautifulSoup."""
    if requests is None or BeautifulSoup is None:
        return "Error: requests/beautifulsoup4 not installed."
    try:
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        # Remove scripts, styles, nav
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Compress multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:20000]  # Cap at 20K chars
    except Exception as exc:
        return f"Scrape error: {exc}"


# ---------------------------------------------------------------------------
# Audio/video scraper (yt-dlp → Whisper)
# ---------------------------------------------------------------------------

def download_and_transcribe_audio(
    url: str,
    assistant: Optional[Any] = None,
) -> Dict[str, Any]:
    """Download audio from a video URL via yt-dlp, transcribe via Whisper.

    Supports multilingual code-switching (Telugu, Hindi, English, mixed)
    and returns clean English text.
    """
    result: Dict[str, Any] = {
        "url": url,
        "transcript": "",
        "error": "",
        "audio_path": "",
    }

    if yt_dlp is None:
        result["error"] = "yt-dlp not installed."
        return result

    if assistant is None:
        result["error"] = "No SuperAssistant instance for Whisper."
        return result

    # yt-dlp options — download audio only, best quality, temp file
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(INGESTION_QUEUE_DIR / "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "audio")

        # Find the downloaded mp3
        audio_files = list(INGESTION_QUEUE_DIR.glob("*.mp3"))
        if not audio_files:
            result["error"] = "No audio file found after download."
            return result

        audio_path = audio_files[-1]
        result["audio_path"] = str(audio_path)

        # Transcribe via Whisper
        transcript = assistant.transcribe_audio(str(audio_path))
        result["transcript"] = transcript

        # Clean up the audio file (don't store — privacy + disk space)
        audio_path.unlink(missing_ok=True)

        return result
    except Exception as exc:
        result["error"] = f"Audio processing error: {exc}"
        return result


# ---------------------------------------------------------------------------
# Vision / chart inspector
# ---------------------------------------------------------------------------

def process_image_upload(
    image_file,
    assistant: Optional[Any] = None,
) -> str:
    """Take an uploaded image, base64-encode it, and send to GPT-4o Vision.

    Returns the Markdown analysis.
    """
    if assistant is None:
        return "Error: No SuperAssistant instance for Vision."

    try:
        # Read and base64-encode
        image_bytes = image_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        result = assistant.analyze_image(image_b64)
        return result
    except Exception as exc:
        return f"Vision analysis error: {exc}"


# ---------------------------------------------------------------------------
# Headless daemon loop
# ---------------------------------------------------------------------------

def background_ingestion_daemon():
    """Background daemon thread that processes the ingestion queue.

    Polls ``./ingestion_queue/`` for JSON job files and processes them
    asynchronously without frontend interaction.
    """
    while True:
        try:
            job_files = sorted(INGESTION_QUEUE_DIR.glob("*.json"))
            for job_file in job_files:
                try:
                    job = json.loads(job_file.read_text(encoding="utf-8"))
                    job_type = job.get("type", "")

                    if job_type == "web_scrape":
                        url = job.get("url", "")
                        if url:
                            text = scrape_web_content(url)
                            # Feed to assistant if available
                            assistant = st.session_state.get("assistant")
                            if assistant and hasattr(assistant, "feed_text_knowledge"):
                                assistant.feed_text_knowledge(
                                    text, source_type="web_scrape", source_id=url
                                )
                    elif job_type == "audio_transcribe":
                        url = job.get("url", "")
                        assistant = st.session_state.get("assistant")
                        if url and assistant:
                            res = download_and_transcribe_audio(url, assistant)
                            if res.get("transcript"):
                                assistant.feed_text_knowledge(
                                    res["transcript"],
                                    source_type="whisper_audio",
                                    source_id=url,
                                )

                    # Remove the job file after processing
                    job_file.unlink(missing_ok=True)
                except Exception:
                    job_file.unlink(missing_ok=True)
        except Exception:
            pass
        # Sleep before next poll
        time.sleep(5)


def start_background_daemon():
    """Start the ingestion daemon in a daemon thread (if not already running)."""
    if "daemon_thread" not in st.session_state:
        thread = threading.Thread(
            target=background_ingestion_daemon,
            daemon=True,
            name="ingestion-daemon",
        )
        st.session_state["daemon_thread"] = thread
        thread.start()


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize Streamlit session state with sane defaults."""
    if "assistant" not in st.session_state:
        st.session_state.assistant = None
    if "git_manager" not in st.session_state:
        st.session_state.git_manager = GitRepositoryManager()
    if "debug_log" not in st.session_state:
        st.session_state.debug_log = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "openai/gpt-4o-mini"
    if "confidence_scoring" not in st.session_state:
        st.session_state.confidence_scoring = True
    if "knowledge_decay" not in st.session_state:
        st.session_state.knowledge_decay = True
    if "injected_context" not in st.session_state:
        st.session_state.injected_context = ""


def initialize_assistant(model_string: str):
    """Create a SuperAssistant instance with the selected model."""
    if SuperAssistant is None:
        st.error(
            "core_engine.py not found or dependencies missing. "
            "SuperAssistant cannot be initialized."
        )
        return None
    try:
        assistant = SuperAssistant(
            model_string=model_string,
            confidence_scoring=st.session_state.confidence_scoring,
            knowledge_decay=st.session_state.knowledge_decay,
        )
        st.session_state.assistant = assistant
        st.session_state.debug_log.append(
            f"[{datetime.now().isoformat()}] SuperAssistant initialized "
            f"with model={model_string}."
        )
        return assistant
    except Exception as exc:
        st.error(f"Failed to initialize SuperAssistant: {exc}")
        st.session_state.debug_log.append(
            f"[{datetime.now().isoformat()}] Init error: {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with model selector and controls."""
    st.sidebar.title("🤖 SuperAssistant")
    st.sidebar.caption("Cloud-Hybrid Permanent AI Agent")

    # Model selector
    model_label = st.sidebar.selectbox(
        "Select LLM Model",
        options=list(OPENROUTER_MODELS.keys()),
        index=0,
    )
    model_string = OPENROUTER_MODELS[model_label]
    st.session_state.selected_model = model_string

    if st.sidebar.button("Initialize / Reconnect", type="primary"):
        initialize_assistant(model_string)

    st.sidebar.divider()

    # Feature toggles
    st.sidebar.subheader("Feature Toggles")
    st.session_state.confidence_scoring = st.sidebar.toggle(
        "Confidence Scoring", value=st.session_state.confidence_scoring
    )
    st.session_state.knowledge_decay = st.sidebar.toggle(
        "Knowledge Decay", value=st.session_state.knowledge_decay
    )

    st.sidebar.divider()

    # Security panel
    st.sidebar.subheader("🔐 Security Panel")
    if st.sidebar.button("Run Integrity Check"):
        if SystemIntegrityManager is not None:
            try:
                mgr = SystemIntegrityManager()
                report = mgr.run_full_integrity_check()
                st.sidebar.write(report)
            except Exception as exc:
                st.sidebar.error(f"Integrity check failed: {exc}")
        else:
            st.sidebar.warning("integrity_engine.py not available.")

    if st.sidebar.button("Rollback to Last Known Good"):
        if SystemIntegrityManager is not None:
            try:
                mgr = SystemIntegrityManager()
                result = mgr.rollback()
                st.sidebar.write(result)
                st.session_state.debug_log.append(
                    f"[{datetime.now().isoformat()}] Rollback: {result}"
                )
            except Exception as exc:
                st.sidebar.error(f"Rollback failed: {exc}")
        else:
            st.sidebar.warning("integrity_engine.py not available.")

    st.sidebar.divider()

    # Inject live code
    st.sidebar.subheader("💉 Inject Live Code")
    injected_code = st.sidebar.text_area(
        "Python code to inject & execute",
        height=150,
        placeholder="print('Hello from injected code')",
    )
    if st.sidebar.button("Execute Injected Code"):
        if injected_code.strip():
            res = execute_injected_code(
                injected_code, st.session_state.get("assistant")
            )
            st.sidebar.json(res)
        else:
            st.sidebar.warning("No code provided.")

    if st.sidebar.button("Append to app.py"):
        if injected_code.strip():
            res = inject_code_to_file(injected_code)
            st.sidebar.write(res.get("message", ""))
        else:
            st.sidebar.warning("No code provided.")

    st.sidebar.divider()

    # Debug log
    st.sidebar.subheader("📜 Debug Log")
    debug_expander = st.sidebar.expander("View Debug Log", expanded=False)
    with debug_expander:
        for line in st.session_state.debug_log[-20:]:
            debug_expander.write(line)


def render_left_pane():
    """Render the left pane: multimodal ingestion."""
    st.header("📥 Multimodal Ingestion")

    # Web scraper
    st.subheader("Web Scraper")
    web_url = st.text_input(
        "URL to scrape", placeholder="https://example.com/article"
    )
    if st.button("Scrape & Ingest"):
        if web_url:
            with st.spinner("Scraping..."):
                text = scrape_web_content(web_url)
            st.text_area("Scraped Content", value=text, height=200)
            assistant = st.session_state.get("assistant")
            if assistant and hasattr(assistant, "feed_text_knowledge"):
                assistant.feed_text_knowledge(
                    text, source_type="web_scrape", source_id=web_url
                )
                st.success("Scraped content ingested into knowledge base.")
        else:
            st.warning("Enter a URL first.")

    st.divider()

    # Audio/video scraper
    st.subheader("Audio / Video Transcription")
    media_url = st.text_input(
        "Video URL (yt-dlp)", placeholder="https://youtube.com/watch?v=..."
    )
    if st.button("Download & Transcribe"):
        assistant = st.session_state.get("assistant")
        if media_url and assistant:
            with st.spinner("Downloading and transcribing..."):
                res = download_and_transcribe_audio(media_url, assistant)
            if res.get("transcript"):
                st.text_area("Transcript", value=res["transcript"], height=200)
                assistant.feed_text_knowledge(
                    res["transcript"],
                    source_type="whisper_audio",
                    source_id=media_url,
                )
                st.success("Transcript ingested into knowledge base.")
            else:
                st.error(f"Transcription failed: {res.get('error', 'unknown')}")
        else:
            st.warning("Enter a URL and initialize the assistant first.")

    st.divider()

    # Vision / image inspector
    st.subheader("Vision / Image Inspector")
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "gif", "webp"],
    )
    if uploaded_image is not None and st.button("Analyze Image"):
        assistant = st.session_state.get("assistant")
        if assistant:
            with st.spinner("Analyzing image..."):
                analysis = process_image_upload(uploaded_image, assistant)
            st.markdown(analysis)
            assistant.feed_text_knowledge(
                analysis,
                source_type="vision_analysis",
                source_id=uploaded_image.name,
            )
        else:
            st.warning("Initialize the assistant first.")

    st.divider()

    # Git repository manager
    st.subheader("🐙 Git Repository Manager")
    repo_url = st.text_input(
        "GitHub repo URL to clone", placeholder="https://github.com/owner/repo"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clone Repo"):
            if repo_url:
                with st.spinner("Cloning..."):
                    res = st.session_state.git_manager.clone_repo(repo_url)
                if res.get("success"):
                    st.success(res.get("message", ""))
                else:
                    st.error(res.get("message", "Clone failed."))
            else:
                st.warning("Enter a repo URL.")
    with col2:
        if st.button("Install Dependencies"):
            with st.spinner("Running pip install..."):
                res = st.session_state.git_manager.install_dependencies()
            if res.get("success"):
                st.success(res.get("message", ""))
            else:
                st.error(res.get("message", "Install failed."))

    if st.button("Read Repo Files & Inject into Context"):
        files = st.session_state.git_manager.read_repo_files()
        if files:
            context = st.session_state.git_manager.expose_to_context()
            st.session_state.injected_context = context
            st.success(f"Loaded {len(files)} files into context.")
            with st.expander("Loaded Files"):
                for fname in files:
                    st.write(f"📄 {fname}")
        else:
            st.warning("No files found. Clone a repo first.")


def render_right_pane():
    """Render the right pane: chat terminal."""
    st.header("💬 Chat Terminal")

    assistant = st.session_state.get("assistant")
    if assistant is None:
        st.info("👈 Initialize the assistant from the sidebar to start chatting.")
        return

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            with st.chat_message(role):
                st.markdown(content)

    # Chat input
    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = assistant.converse(user_input)
                except Exception as exc:
                    reply = f"(Error: {exc})"
            st.markdown(reply)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply}
        )
        st.rerun()


def main():
    """Main entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="SuperAssistant Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    start_background_daemon()

    render_sidebar()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        render_left_pane()
    with col_right:
        render_right_pane()


if __name__ == "__main__":
    main()
