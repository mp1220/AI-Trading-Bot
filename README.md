Trader_Model Hybrid Runtime
===========================

The orchestrator now coordinates two complementary data paths:

- **Scheduled data layer** keeps the full S&P100 universe fresh on predictable cadences for model training and feature engineering.
- **Real-time stream layer** watches a compact focus list (10–20 tickers) through the Alpaca IEX feed.
- **Decision engine** consumes both layers, evaluates live opportunities, and submits trades once confidence, sentiment, and safety filters agree.

Setup
-----
- Initialise PostgreSQL once:
  ```bash
  createdb trader
  psql trader < schema.sql
  ```
- Copy `.env.example` to `.env` and fill secrets. New keys of note:
  - `FOCUS_TICKERS` – comma list of 10–20 tickers to stream continuously.
  - `AI_AUTONOMOUS_TRADING` – enable/disable the autonomous trade loop.
  - `DISCORD_ALERT_LEVEL` – minimum severity for webhook diagnostics (`debug`, `info`, `warn`, `error`, `critical`).
  - `ALPACA_API_BASE_URL` – overrides the Alpaca REST endpoint (paper by default).
- Install dependencies:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

Hybrid Data Flow
----------------
- **Scheduled jobs (APScheduler)**
  - `price_update` every 5 minutes (full S&P 100 history refresh).
  - `news_update` every 15 minutes (News + GDELT sentiment).
  - `fundamentals_update` every 60 minutes (FMP ratios).
  - `social_scan` every 15 minutes (Reddit/Twitter pulse).
  - `model_checkpoint` every 2 hours (online training snapshots).
  - `diagnostics` every 4 hours (system heartbeat + error summary).
- **Real-time stream**
  - Alpaca live quotes for configured `FOCUS_TICKERS` (capped to 20 symbols).
  - Quotes persisted in `live_quotes` for the decision engine and volatility gating.
- **Decision engine**
  - Merges latest scheduled signal, live quote, sentiment aggregates, and volatility estimate.
  - Executes only when:
    - Confidence ≥ `decision_engine.confidence_threshold` (default 0.75).
    - Sentiment alignment and technical reasoner confirm direction.
    - Cooldown (1 trade/min/symbol) and position limits are clear.
  - Trades are logged to `positions` and `trades`, with Discord notifications for executions and skips.

Running
-------
- Start the orchestrator (loads dotenv automatically):
  ```bash
  python -m app.orchestrator
  ```
- On startup:
  - Scheduler kicks off all scheduled jobs.
  - Alpaca stream begins for focus tickers (paper/live based on `.env`).
  - The decision engine runs continuously with adaptive sleep (30s under high volatility, 60–120s otherwise).

Diagnostics & Safety
--------------------
- Discord notifications:
  - Executed trades post `💸 AI Trade ...` messages at `warn` level.
  - Skipped trades (cooldown, sentiment mismatch, etc.) publish `⚠️` diagnostics respecting `DISCORD_ALERT_LEVEL`.
  - Four-hour system report summarises error health, open positions, and trade counts.
- Database schema now includes:
  - `positions.side`, `trades.rationale`, and helper indices for rapid cooldown checks.
  - Helper methods in `app/storage.py` for live quote retrieval, sentiment aggregation, and trade logging.
- `AI_AUTONOMOUS_TRADING=false` immediately parks the decision engine while keeping the scheduler active.

Next Steps
----------
- Tune `decision_engine.order_notional` and thresholds in `config.yaml`.
- Expand catalyst/news layers with additional providers if required.
- Integrate richer Discord embeds or dashboards for executed trades if desired.
