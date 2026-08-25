# Cloudflare D1 SQL Setup Scripts

This directory contains SQL scripts to set up your Cloudflare D1 database
tables. The application auto-creates these on first run, but you can run
them manually if needed.

## Tables

1. **chat_history** — every conversation line (timestamp, speaker, message)
2. **global_knowledge_core** — persistent key-value store for user identity,
   model config, tracking criteria
3. **creators** — followed Instagram accounts with reputation and accuracy tracking
4. **strategies** — generated strategies with confidence and backtest results
5. **knowledge_entries** — text extracted from reels, scrapes, vision, audio
6. **audit_trail** — every decision logged with reasoning chain
7. **calendar_events** — reminders, meetings, market events
8. **predictions_ledger** — prediction tracking and accountability
9. **alerts** — pending notifications and triggered conditions

## Auto-bootstrap

The `SuperAssistant` class in `core_engine.py` automatically runs the
table creation DDL on initialization. You do NOT need to run these manually
unless you want to pre-create them.

## Manual setup via Cloudflare Dashboard

1. Go to Cloudflare Dashboard → Workers & Pages → D1
2. Select your database
3. Go to "Query" tab
4. Paste and run the SQL from `core_engine.py`'s `_D1_BOOTSTRAP_SQL` constant
