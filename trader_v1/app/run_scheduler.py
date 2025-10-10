"""Entry point for the pulse-based orchestration architecture."""
from __future__ import annotations

import signal
from typing import List

from app.config import load_config, resolve_interval
from app.data_ingest import DataIngest
from app.news_fetch import NewsFetch
from app.sentiment import Sentiment
from app.scheduler import PulseScheduler
from app.storage import DB, make_engine
from app.watchlist import WatchlistManager


def _limit(tickers: List[str], n: int) -> List[str]:
    return list(tickers[:n])


def main() -> None:
    cfg = load_config()
    engine = make_engine()
    db = DB(engine)
    db.ensure_schema()

    ingest = DataIngest(db, cfg)
    news = NewsFetch(cfg)
    sentiment = Sentiment(cfg)
    watchlist_mgr = WatchlistManager(db, ingest, cfg)
    scheduler = PulseScheduler()

    watchlist_mgr.refresh()

    def current_top() -> List[str]:
        tickers = watchlist_mgr.current()
        if not tickers:
            base = ingest.universe or ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]
            return _limit(base, watchlist_mgr.top_n)
        return tickers

    # Watchlist refresh pulse
    scheduler.register(
        "watchlist_refresh",
        resolve_interval(cfg, "watchlist"),
        lambda: watchlist_mgr.refresh(),
        run_immediately=True,
    )

    # Finnhub quote pulse (Top 10 tickers)
    scheduler.register(
        "finnhub_quote",
        resolve_interval(cfg, "finnhub_quotes"),
        lambda: ingest.pulse_finnhub_quotes(_limit(current_top(), watchlist_mgr.top_n)),
        run_immediately=True,
    )

    # Finnhub news & sentiment pulse
    scheduler.register(
        "finnhub_news",
        resolve_interval(cfg, "finnhub_news"),
        lambda: ingest.pulse_finnhub_news(_limit(current_top(), watchlist_mgr.top_n)),
    )

    providers_cfg = cfg.get("providers") or {}
    alpaca_cfg = providers_cfg.get("alpaca") or {}
    if alpaca_cfg.get("enabled"):
        scheduler.register(
            "alpaca_bars",
            resolve_interval(cfg, "alpaca_bars"),
            lambda: ingest.update_from_alpaca(timeframe="1Min"),
            run_immediately=True,
        )
        scheduler.register(
            "alpaca_news",
            resolve_interval(cfg, "alpaca_news"),
            lambda: news.fetch_latest(
                db, _limit(current_top(), watchlist_mgr.top_n), lookback_days=1
            ),
            run_immediately=True,
        )

    # yfinance 15m pulse for entire universe
    scheduler.register(
        "yfinance_15m",
        resolve_interval(cfg, "yfinance_15m"),
        lambda: ingest.pulse_yfinance_bars(ingest.universe, interval="15m"),
    )

    # TwelveData backup pulse
    scheduler.register(
        "twelvedata_backup",
        resolve_interval(cfg, "twelvedata_backup"),
        lambda: ingest.pulse_twelvedata_backup(ingest.universe),
    )

    # FMP fundamentals pulse
    scheduler.register(
        "fmp_fundamentals",
        resolve_interval(cfg, "fmp_fundamentals"),
        lambda: ingest.pulse_fmp_fundamentals(ingest.universe),
    )

    # MarketStack EOD pulse
    scheduler.register(
        "marketstack_eod",
        resolve_interval(cfg, "marketstack_eod"),
        lambda: ingest.pulse_marketstack_eod(ingest.universe),
    )

    if sentiment.client is not None:
        scheduler.register(
            "sentiment_refresh",
            resolve_interval(cfg, "sentiment_refresh"),
            lambda: sentiment.update_unscored(db),
        )

    def shutdown(_signo, _frame):
        scheduler.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    scheduler.run_forever()


if __name__ == "__main__":
    main()
