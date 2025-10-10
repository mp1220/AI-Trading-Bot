"""Dynamic watchlist manager that re-ranks companies of interest."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

SP100_TICKERS: List[str] = [
    "AAPL",
    "ABBV",
    "ABT",
    "ACN",
    "ADBE",
    "AIG",
    "AMD",
    "AMGN",
    "AMT",
    "AMZN",
    "AVGO",
    "AXP",
    "BA",
    "BAC",
    "BK",
    "BKNG",
    "BLK",
    "BMY",
    "C",
    "CAT",
    "CHTR",
    "CL",
    "CMCSA",
    "COF",
    "COP",
    "COST",
    "CRM",
    "CSCO",
    "CVS",
    "CVX",
    "DE",
    "DHR",
    "DIS",
    "DOW",
    "DVN",
    "EMR",
    "EXC",
    "F",
    "FDX",
    "GD",
    "GE",
    "GILD",
    "GM",
    "GOOG",
    "GOOGL",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "INTU",
    "ISRG",
    "JNJ",
    "JPM",
    "KHC",
    "KMI",
    "KO",
    "LIN",
    "LMT",
    "LOW",
    "LRCX",
    "MCD",
    "MDT",
    "META",
    "MMM",
    "MO",
    "MRK",
    "MS",
    "MSFT",
    "MU",
    "NKE",
    "NFLX",
    "NOC",
    "NVDA",
    "ORCL",
    "OXY",
    "PEP",
    "PFE",
    "PG",
    "PM",
    "PYPL",
    "QCOM",
    "RTX",
    "SBUX",
    "SLB",
    "SO",
    "SPG",
    "T",
    "TGT",
    "TMO",
    "TSLA",
    "TXN",
    "UNH",
    "UNP",
    "UPS",
    "USB",
    "V",
    "VZ",
    "WBA",
    "WFC",
    "WMT",
    "XOM",
]


class WatchlistManager:
    def __init__(self, db, ingest, cfg: Dict, logger=None):
        self.db = db
        self.ingest = ingest
        self.cfg = cfg
        self._logger = logger or (lambda msg: print(msg, flush=True))
        self._watch_cfg = cfg.get("watchlist") or {}
        self._universe_cfg = cfg.get("universe") or {}
        self.top_n = int(self._watch_cfg.get("top_n", 10))
        self.weights = self._watch_cfg.get("weights") or {"momentum": 0.5, "volume": 0.3, "sentiment": 0.2}
        self.lookbacks = self._watch_cfg.get("lookbacks") or {
            "momentum_days": 5,
            "volume_days": 10,
            "sentiment_hours": 6,
        }
        self._current: List[str] = self._initial_watchlist()

    def _initial_watchlist(self) -> List[str]:
        tickers = list(self._universe_iterable())[: self.top_n]
        if not tickers:
            self._logger("⚠️  Watchlist initialization found no tickers.")
        return tickers

    def _universe_iterable(self) -> Iterable[str]:
        if self._universe_cfg.get("use_sp100"):
            return SP100_TICKERS
        tickers = self._universe_cfg.get("tickers") or []
        return tickers

    def current(self) -> List[str]:
        return list(self._current)

    def _load_sentiment_scores(self, tickers: List[str]) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame(columns=["ticker", "sentiment"])
        hours = float(self.lookbacks.get("sentiment_hours", 6))
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        sql = """
            SELECT ticker, AVG(sentiment) AS sentiment
            FROM news_headlines
            WHERE ticker = ANY(%(tickers)s)
              AND sentiment IS NOT NULL
              AND published_at >= %(since)s
            GROUP BY ticker
        """
        df = self.db.to_df(sql, {"tickers": tickers, "since": since.replace(tzinfo=None)})
        if df.empty:
            return pd.DataFrame({"ticker": tickers, "sentiment": [0.0] * len(tickers)})
        return df

    def refresh(self) -> List[str]:
        tickers = list(self._universe_iterable())
        if not tickers:
            self._logger("⚠️  No universe tickers configured; watchlist not updated.")
            return self.current()

        metrics = self.ingest.compute_watchlist_metrics(
            tickers=tickers,
            lookbacks=self.lookbacks,
        )
        if metrics.empty:
            self._logger("⚠️  Watchlist metrics empty; retaining previous watchlist.")
            return self.current()

        sentiment_df = self._load_sentiment_scores(tickers)
        combined = metrics.merge(sentiment_df, on="ticker", how="left").fillna({"sentiment": 0.0})

        combined["momentum_rank"] = combined["momentum"].rank(pct=True)
        combined["volume_rank"] = combined["volume_ratio"].rank(pct=True)
        combined["sentiment_score"] = np.clip(combined["sentiment"], -1.0, 1.0)
        combined["sentiment_rank"] = (combined["sentiment_score"] + 1.0) / 2.0

        w = self.weights
        combined["score"] = (
            combined["momentum_rank"] * float(w.get("momentum", 0))
            + combined["volume_rank"] * float(w.get("volume", 0))
            + combined["sentiment_rank"] * float(w.get("sentiment", 0))
        )
        combined.sort_values("score", ascending=False, inplace=True)

        new_watchlist = combined.head(self.top_n)["ticker"].tolist()
        self._current = new_watchlist
        self._logger(f"📈 Watchlist updated: {', '.join(new_watchlist)}")
        return self.current()
