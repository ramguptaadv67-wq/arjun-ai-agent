# Deployment Guide — Cloud-Hybrid AI Agent (Arjun)

## Prerequisites — API Keys & Accounts

Before deploying, you need accounts and API keys for:

| Service | Purpose | Free Tier |
|---|---|---|
| OpenRouter | LLM gateway (GPT-4o, Claude, Gemini, etc.) | Pay-per-token |
| OpenAI | Embeddings, Whisper, GPT-4o Vision | Pay-per-token |
| Qdrant Cloud | Remote vector database (3072 dims) | 1GB free |
| Cloudflare | D1 SQL database + R2 storage + Workers | Generous free tier |
| Render | Python hosting (Streamlit) | Free (sleeps) / $7/mo |

---

## Step 1 — Cloudflare D1 Setup

1. Create a Cloudflare account at cloudflare.com
2. Go to **Workers & Pages → D1**
3. Click **Create Database**
4. Name it `arjun_ai_memory`
5. Note the **Database ID** — this is your `CF_D1_DATABASE_ID`
6. Go to your account dashboard, note your **Account ID** — this is `CF_ACCOUNT_ID`
7. Create an API token:
   - Go to **My Profile → API Tokens → Create Token**
   - Use template "Cloudflare D1" or create custom with D1 edit permissions
   - This is your `CF_API_TOKEN`

---

## Step 2 — Qdrant Cloud Setup

1. Go to cloud.qdrant.io
2. Sign up and create a free cluster
3. Note the **cluster URL** — this is your `QDRANT_CLOUD_URL`
4. Go to cluster settings → API Keys
5. Create a key — this is your `QDRANT_CLOUD_API_KEY`
6. The collection `multimodal_knowledge` (3072 dims) is auto-created on first run

---

## Step 3 — OpenRouter Setup

1. Go to openrouter.ai
2. Sign up and add credits
3. Go to Keys → Create Key
4. This is your `OPENROUTER_API_KEY`
5. Available models in the dashboard:
   - `google/gemini-pro-1.5`
   - `anthropic/claude-3.5-sonnet`
   - `openai/gpt-4o-mini`
   - `deepseek/deepseek-chat`
   - `meta-llama/llama-3-70b-instruct`

---

## Step 4 — OpenAI Setup

1. Go to platform.openai.com
2. Add billing (required for embeddings + Whisper + Vision)
3. Create an API key
4. This is your `OPENAI_API_KEY`

---

## Step 5 — Deploy on Render

### Option A: Docker (Recommended)

1. Push all files to a GitHub repository
2. Go to render.com → New → **Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Name**: `arjun-ai-agent`
   - **Runtime**: Docker
   - **Root Directory**: `/` (root — Dockerfile is at root)
   - **Instance Type**: Free or Starter ($7/mo for always-on)
5. Go to **Environment** → add all environment variables:
   ```
   OPENROUTER_API_KEY=sk-or-v1-xxx
   OPENAI_API_KEY=sk-xxx
   QDRANT_CLOUD_URL=https://xxx.aws.cloud.qdrant.io:6333
   QDRANT_CLOUD_API_KEY=xxx
   CF_ACCOUNT_ID=xxx
   CF_D1_DATABASE_ID=xxx
   CF_API_TOKEN=xxx
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ```
6. Click **Create Web Service**
7. Wait for build to complete (~3-5 minutes)
8. Access at `https://arjun-ai-agent.onrender.com`

### Option B: Python Direct

1. Push all files to GitHub
2. Go to render.com → New → **Web Service**
3. Connect repo
4. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Plan**: Free or Starter
5. Add the same environment variables as above
6. Deploy

### Important Render Notes

- **Free tier sleeps** after 15 min of inactivity. Cloud data persists because
  it lives in Cloudflare D1 + Qdrant, not on the Render disk.
- **Starter ($7/mo)** keeps the app always-on. Recommended for production.
- **Disk**: Not needed. All data goes to Cloudflare. Render is pure compute.

---

## Step 6 — Local Development

```bash
# 1. Clone or copy files
cd ai-agent-system

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

# 6. Run the app
streamlit run app.py
```

Access at `http://localhost:8501`

---

## Step 7 — Verify Everything Works

1. Open the dashboard
2. Check the sidebar — all environment variables should show ✅
3. The security panel should show "✅ SYSTEM VERIFIED"
4. Type "hello" in the chat → should get a response from OpenRouter
5. Try the web scraper with any URL
6. Upload an image to the Vision inspector
7. Create a calendar alert

---

## Architecture — Where Data Lives

```
┌─────────────────────┐
│   Render (Compute)   │
│  • Streamlit UI     │
│  • Python engine    │
│  • No local data    │
└──────────┬──────────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
┌──────┐┌──────┐┌──────┐
│  D1  ││Qdrant││ R2*  │
│(SQL) ││(Vec) ││(Obj) │
└──────┘└──────┘└──────┘
    ▲      ▲
    │      │
┌───┴──────┴───┐
│  External    │
│  APIs        │
│  • OpenRouter│
│  • OpenAI    │
│  • yfinance  │
└──────────────┘
```

* R2 object storage is available for future file storage (user uploads,
reel screenshots, etc). Currently images are processed in-memory and not
stored permanently.

---

## Troubleshooting

### App loads but chat doesn't work
- Check `OPENROUTER_API_KEY` is set
- Check `OPENAI_API_KEY` is set (needed for embeddings even if using OpenRouter for LLM)

### Qdrant connection fails
- Verify `QDRANT_CLOUD_URL` includes the port (`:6333`)
- Verify `QDRANT_CLOUD_API_KEY` is correct

### D1 queries fail
- Verify all 3 Cloudflare vars: `CF_ACCOUNT_ID`, `CF_D1_DATABASE_ID`, `CF_API_TOKEN`
- Check the API token has D1 edit permissions

### yt-dlp fails
- Ensure ffmpeg is installed (Docker image includes it)
- For local dev: `apt install ffmpeg` or `brew install ffmpeg`

### Vision analysis fails
- Ensure `OPENAI_API_KEY` has billing enabled
- GPT-4o Vision requires a paid OpenAI account

---

## Security Notes

- Never commit `.env` to git
- The integrity engine creates restore points in `./system_restore_points/`
- Tampering detection runs on every page load
- Code injection is validated with `compile()` before writing
- The autonomous debug engine validates repaired code before deploying
