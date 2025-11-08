CREATE TABLE IF NOT EXISTS minute_bars (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    ts TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    UNIQUE (ts, ticker)
);

CREATE TABLE IF NOT EXISTS hour_bars (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    ts TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    UNIQUE (ts, ticker)
);

CREATE TABLE IF NOT EXISTS daily_bars (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    ts TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    UNIQUE (ts, ticker)
);

CREATE TABLE IF NOT EXISTS news_headlines (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    headline TEXT NOT NULL,
    source TEXT,
    url TEXT,
    published_at TIMESTAMP WITHOUT TIME ZONE,
    fingerprint TEXT UNIQUE,
    sentiment DOUBLE PRECISION,
    inserted_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sync_log (
    id SERIAL PRIMARY KEY,
    source TEXT UNIQUE NOT NULL,
    last_updated TIMESTAMP WITHOUT TIME ZONE
);

CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    ts TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    prob_up DOUBLE PRECISION,
    sentiment DOUBLE PRECISION,
    action TEXT,
    details JSONB,
    UNIQUE (ticker, ts)
);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    opened_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    closed_at TIMESTAMP WITHOUT TIME ZONE,
    qty DOUBLE PRECISION,
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    pnl DOUBLE PRECISION,
    status TEXT,
    UNIQUE (ticker, opened_at)
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    ts TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    side TEXT,
    qty DOUBLE PRECISION,
    price DOUBLE PRECISION,
    fees DOUBLE PRECISION,
    pnl DOUBLE PRECISION,
    signal_id INT,
    position_id INT,
    UNIQUE (ticker, ts, side)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id SERIAL PRIMARY KEY,
    run_started TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    run_ended TIMESTAMP WITHOUT TIME ZONE,
    bars_inserted INT,
    news_inserted INT,
    signals_created INT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS provider_payloads (
    id SERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    ticker TEXT,
    pulse TEXT,
    ts TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    payload JSONB,
    UNIQUE (provider, ticker, pulse, ts)
);
