"""Manual backfill utility for Trader_Model."""
from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import requests
import yfinance as yf

from app.config import load_config
from app.storage import DB

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger("Backfill")


# ----------------------------------------------------------------------
# ARGUMENT PARSING
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical market and news data.")
    parser.add_argument("--start", default="2025-09-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument("--symbols", help="Comma separated symbols to backfill.")
    parser.add_argument("--interval", default="1h", help="yfinance interval (default 1h)")
    parser.add_argument("--include-news", action="store_true", help="Backfill news via GDELT")
    return parser.parse_args()


# ----------------------------------------------------------------------
# LOAD SYMBOL LIST
# ----------------------------------------------------------------------
def _load_symbol_list(cfg: dict, arg_symbols: str | None) -> List[str]:
    """Load tickers either from arguments or config."""
    if arg_symbols:
        return [symbol.strip().upper() for symbol in arg_symbols.split(",") if symbol.strip()]

    runtime_list = cfg.get("runtime", {}).get("benchmark_tickers")
    if runtime_list:
        return [str(symbol).upper() for symbol in runtime_list]

    universe_section = cfg.get("universe", {})
    if isinstance(universe_section, dict):
        universe = universe_section.get("tickers", [])
    elif isinstance(universe_section, list):
        universe = universe_section
    else:
        universe = []

    return [str(symbol).upper() for symbol in universe]


# ----------------------------------------------------------------------
# FETCH MARKET DATA
# ----------------------------------------------------------------------
def _fetch_market_data(db: DB, symbol: str, start: datetime, end: datetime, interval: str) -> None:
    LOGGER.info("📡 Fetching %s %s → %s", symbol, start.date(), end.date())
    try:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        LOGGER.warning("⚠️ yfinance fetch failed for %s: %s", symbol, exc)
        return

    if raw is None or raw.empty:
        LOGGER.info("ℹ️ No market data returned for %s", symbol)
        return

    frame = raw.reset_index()

    # ✅ flatten MultiIndex columns if present
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = ['_'.join([str(c) for c in col if c]) for col in frame.columns]

    # ✅ normalize column names to lowercase
    frame.columns = [str(col).strip().lower() for col in frame.columns]

    # ✅ detect timestamp column
    timestamp_col = None
    for candidate in ["datetime", "date", "index"]:
        if candidate in frame.columns:
            timestamp_col = candidate
            break
    if timestamp_col is None:
        timestamp_col = frame.columns[0]
        LOGGER.warning("⚠️ Using first column '%s' as timestamp for %s", timestamp_col, symbol)

    # ✅ fuzzy column detection
    def find_col(keyword: str) -> str:
        matches = [c for c in frame.columns if keyword in c]
        if not matches:
            raise KeyError(f"Missing column containing '{keyword}'")
        return matches[0]

    try:
        open_col = find_col("open")
        high_col = find_col("high")
        low_col = find_col("low")
        close_col = find_col("close")
        volume_col = find_col("volume")
    except KeyError as e:
        LOGGER.error(f"❌ Failed to find expected column for {symbol}: {e}")
        LOGGER.error(f"Columns found: {frame.columns.tolist()}")
        return

    working = pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce"),
            "open": pd.to_numeric(frame[open_col], errors="coerce"),
            "high": pd.to_numeric(frame[high_col], errors="coerce"),
            "low": pd.to_numeric(frame[low_col], errors="coerce"),
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
            "volume": pd.to_numeric(frame[volume_col], errors="coerce").fillna(0),
        }
    ).dropna(subset=["timestamp"])

    inserted, skipped = db.upsert_bars_multi(working, update=False)
    LOGGER.info("✅ Inserted %d bars for %s (skipped %d)", inserted, symbol, skipped)


# ----------------------------------------------------------------------
# FETCH NEWS DATA (GDELT) — with delay and retry
# ----------------------------------------------------------------------
def _fetch_gdelt_news(symbol: str, start: datetime, end: datetime, retries: int = 3, delay: float = 1.5) -> List[dict]:
    """Fetch headlines for a ticker from GDELT with retry + delay control."""
    params = {"query": symbol, "maxrecords": 250, "format": "json"}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json() or {}
            articles = payload.get("articles") or []
            if not articles:
                LOGGER.warning("⚠️ No GDELT articles found for %s (attempt %d)", symbol, attempt)
            break
        except Exception as exc:
            LOGGER.warning("⚠️ GDELT fetch failed for %s (attempt %d/%d): %s", symbol, attempt, retries, exc)
            if attempt < retries:
                time.sleep(delay * attempt)  # exponential backoff
                continue
            else:
                return []

    rows: List[dict] = []
    for article in articles:
        headline = article.get("title")
        url = article.get("url") or ""
        if not headline:
            continue
        news_id = hashlib.sha1(f"{headline}{url}".encode("utf-8")).hexdigest()
        published = article.get("seendate")
        timestamp = None
        if published:
            try:
                timestamp = datetime.strptime(published, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        rows.append(
            {
                "id": news_id,
                "ticker": symbol,
                "headline": headline,
                "source": article.get("sourcecommonname"),
                "url": url,
                "published_at": timestamp,
                "sentiment": None,
                "headline_hash": news_id,
            }
        )


    # ✅ short random pause between symbols to avoid rate limit
    time.sleep(delay + random.uniform(0.5, 1.5))
    return rows


# ----------------------------------------------------------------------
# MAIN ENTRY
# ----------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    cfg = load_config()
    symbols = _load_symbol_list(cfg, args.symbols)
    if not symbols:
        LOGGER.error("No symbols supplied; aborting backfill.")
        sys.exit(1)

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) + timedelta(days=1)

    db = DB()
    try:
        for symbol in symbols:
            _fetch_market_data(db, symbol, start, end, args.interval)

        if args.include_news:
            total_inserted = total_skipped = 0
            for i, symbol in enumerate(symbols, start=1):
                LOGGER.info("📰 [%d/%d] Fetching GDELT headlines for %s", i, len(symbols), symbol)
                rows = _fetch_gdelt_news(symbol, start, end, retries=3, delay=1.5)
                if rows:
                    inserted, skipped = db.insert_news_rows(rows)
                    total_inserted += inserted
                    total_skipped += skipped
                time.sleep(random.uniform(1.0, 2.0))  # throttle per symbol
            LOGGER.info("📰 GDELT backfill complete — %d inserted, %d skipped", total_inserted, total_skipped)
    finally:
        db.close()


if __name__ == "__main__":
    main()
