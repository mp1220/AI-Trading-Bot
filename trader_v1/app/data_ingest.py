"""External data ingestion utilities for the trading bot."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from app.config import get_provider
from app.market_data import fetch_bars
from app.watchlist import SP100_TICKERS


LOG = logging.getLogger("MarketData")


FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
FINNHUB_NEWS_SENTIMENT_URL = "https://finnhub.io/api/v1/news-sentiment"
TWELVEDATA_SERIES_URL = "https://api.twelvedata.com/time_series"
FMP_PROFILE_URL = "https://financialmodelingprep.com/api/v3/profile/{ticker}"
MARKETSTACK_EOD_URL = "http://api.marketstack.com/v1/eod"


def safe_download(ticker: str, **kwargs):
    kwargs.setdefault("progress", False)
    for attempt in range(3):
        try:
            return yf.download(ticker, **kwargs)
        except Exception as exc:  # noqa: BLE001 - retryable API failure
            print(f"[Retry {attempt + 1}] yfinance failed for {ticker}: {exc}")
            time.sleep(2**attempt)
    print(f"[ERROR] yfinance permanently failed for {ticker}")
    return None


class DataIngest:
    def __init__(self, db, cfg: Dict, logger=None):
        self.db = db
        self.cfg = cfg
        self._logger = logger or (lambda msg: print(msg, flush=True))
        self.providers: Dict[str, object] = {
            name: get_provider(cfg, name)
            for name in [
                "finnhub",
                "yfinance",
                "alpaca",
                "fmp",
                "twelvedata",
                "marketstack",
                "polygon",
                "newsapi",
            ]
        }
        self._providers_cfg = cfg.get("providers") or {}
        self.universe = self._resolve_universe()
        self._logger = logging.getLogger("MarketData")
        key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        if key and secret:
            self.alpaca_data_client = StockHistoricalDataClient(key, secret)
            feed = (os.getenv("ALPACA_FEED") or os.getenv("ALPACA_DATA_FEED") or "iex").upper()
            self._alpaca_feed = DataFeed.SIP if feed == "SIP" else DataFeed.IEX
        else:
            self.alpaca_data_client = None
            self._alpaca_feed = DataFeed.IEX

    # --------------------------------------------------------------------- #
    # Helpers                                                                #
    # --------------------------------------------------------------------- #
    def _log(self, msg: str, level: int = logging.INFO) -> None:
        self._logger.log(level, msg)

    def _provider_cfg(self, name: str) -> Dict:
        return self._providers_cfg.get(name, {})

    def _provider_token(self, name: str) -> Optional[str]:
        provider = self.providers.get(name)
        if provider is None:
            return None
        return provider.api_key

    @staticmethod
    def _naive(ts: datetime) -> datetime:
        if ts.tzinfo:
            return ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts

    def _resolve_universe(self) -> List[str]:
        universe_cfg = self.cfg.get("universe") or {}
        tickers = list(universe_cfg.get("tickers") or [])
        if universe_cfg.get("use_sp100"):
            tickers = list(dict.fromkeys(SP100_TICKERS + tickers))
        return tickers

    def ensure_initial_history(self, interval: str = "15m") -> None:
        tickers = self.universe or []
        missing = [t for t in tickers if self.db.latest_bar_timestamp(t) is None]
        if not missing:
            return
        self._log(
            f"📦 Initial market data backfill required for {', '.join(missing)}"
        )
        self.pulse_yfinance_bars(missing, interval=interval)

    @staticmethod
    def _chunks(seq: Iterable[str], size: int) -> List[List[str]]:
        seq = list(seq)
        if size <= 0:
            size = len(seq) or 1
        return [seq[i : i + size] for i in range(0, len(seq), size)]

    @staticmethod
    def _timeframe_from_label(label: str) -> TimeFrame:
        cleaned = label.strip().lower()
        match = re.match(r"(\d+)([a-z]+)", cleaned)
        amount = int(match.group(1)) if match else 1
        unit = match.group(2) if match else "min"
        if unit.startswith("min"):
            return TimeFrame(amount, TimeFrameUnit.Minute)
        if unit.startswith("h"):
            return TimeFrame(amount, TimeFrameUnit.Hour)
        if unit.startswith("d"):
            return TimeFrame(amount, TimeFrameUnit.Day)
        return TimeFrame(amount, TimeFrameUnit.Minute)

    def _http_get(self, url: str, *, params=None, headers=None, timeout: int = 15) -> Optional[Dict]:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - log and continue
            self._log(f"ℹ️  Request failed ({url}): {exc}")
            return None
    # --------------------------------------------------------------------- #
    # Watchlist metrics                                                     #
    # --------------------------------------------------------------------- #
    def compute_watchlist_metrics(self, tickers: List[str], lookbacks: Dict) -> pd.DataFrame:
        momentum_days = int(lookbacks.get("momentum_days", 5))
        volume_days = int(lookbacks.get("volume_days", 10))
        window_days = max(momentum_days, volume_days) + 5

        rows: List[Dict] = []
        for ticker in tickers:
            raw = safe_download(
                ticker,
                period=f"{window_days}d",
                interval="1d",
                auto_adjust=False,
            )
            if raw is None or raw.empty:
                print(f"[MarketData] No data for {ticker}, skipping.")
                continue

            raw = raw.rename(columns=str.lower)
            closes = raw.get("close")
            volumes = raw.get("volume")
            if closes is None or volumes is None:
                continue

            momentum = 0.0
            if len(closes) > momentum_days:
                momentum = float(closes.iloc[-1] / closes.iloc[-momentum_days - 1] - 1.0)

            volume_ratio = 1.0
            if len(volumes) >= volume_days:
                recent_volume = float(volumes.iloc[-1])
                avg_volume = float(volumes.tail(volume_days).mean())
                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume

            rows.append({"ticker": ticker, "momentum": momentum, "volume_ratio": volume_ratio})

        return pd.DataFrame(rows)

    # --------------------------------------------------------------------- #
    # Finnhub                                                               #
    # --------------------------------------------------------------------- #
    def pulse_finnhub_quotes(self, tickers: List[str]) -> None:
        provider = self.providers.get("finnhub")
        if not provider or not provider.enabled:
            self._log("ℹ️ Finnhub quotes disabled in config.")
            return
        token = provider.api_key
        if not token:
            self._log("⚠️ Finnhub token missing; skipping quotes pulse.")
            return

        rows = []
        for ticker in tickers:
            params = {"symbol": ticker, "token": token}
            data = self._http_get(FINNHUB_QUOTE_URL, params=params)
            if not data:
                continue
            ts_raw = data.get("t") or time.time()
            ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
            close = data.get("c") or data.get("pc") or 0.0
            open_ = data.get("o") or close
            high = data.get("h") or close
            low = data.get("l") or close
            volume = data.get("v") or 0.0
            rows.append(
                {
                    "ts": ts,
                    "ticker": ticker,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        if not rows:
            self._log("ℹ️ Finnhub quotes pulse found no data.")
            return

        df = pd.DataFrame(rows)
        self.db.upsert_bars(df)
        latest_ts = max(row["ts"] for row in rows)
        self.db.set_last_sync("finnhub_quote", self._naive(latest_ts))
        self._log(f"💾 Finnhub quotes upserted for {len(rows)} tickers.")

    def pulse_finnhub_news(self, tickers: List[str]) -> None:
        provider = self.providers.get("finnhub")
        if not provider or not provider.enabled:
            self._log("ℹ️ Finnhub news disabled in config.")
            return
        token = provider.api_key
        if not token:
            self._log("⚠️ Finnhub token missing; skipping news pulse.")
            return

        lookback_hours = float(self._provider_cfg("finnhub").get("news_lookback_hours", 6))
        now = datetime.now(timezone.utc)
        from_date = (now - timedelta(hours=lookback_hours)).strftime("%Y-%m-%d")
        to_date = now.strftime("%Y-%m-%d")

        total_new = 0
        for ticker in tickers:
            news_params = {"symbol": ticker, "from": from_date, "to": to_date, "token": token}
            articles = self._http_get(FINNHUB_COMPANY_NEWS_URL, params=news_params) or []

            sentiment_params = {"symbol": ticker, "token": token}
            sentiment_payload = self._http_get(FINNHUB_NEWS_SENTIMENT_URL, params=sentiment_params) or {}
            sentiment_score = sentiment_payload.get("companyNewsScore")

            rows = []
            for article in articles:
                headline = article.get("headline")
                if not headline:
                    continue
                published_ts = article.get("datetime") or article.get("time") or time.time()
                published_at = datetime.fromtimestamp(published_ts, tz=timezone.utc)
                news_id = hashlib.sha1(
                    f"{headline}{article.get('url', '')}".encode("utf-8")
                ).hexdigest()
                rows.append(
                    {
                        "id": news_id,
                        "ticker": ticker,
                        "headline": headline,
                        "source": article.get("source"),
                        "url": article.get("url"),
                        "published_at": self._naive(published_at),
                        "sentiment": sentiment_score,
                    }
                )

            inserted, skipped = self.db.insert_news_rows(rows)
            total_new += inserted
            if inserted or skipped:
                self._log(
                    f"📰 Finnhub added {inserted} articles for {ticker} (skipped {skipped})."
                )

        self.db.set_last_sync("finnhub_news", self._naive(datetime.now(timezone.utc)))
        self._log(f"💾 Finnhub news pulse complete — {total_new} new headlines.")

    # --------------------------------------------------------------------- #
    # Yahoo Finance                                                         #
    # --------------------------------------------------------------------- #
    def pulse_yfinance_bars(self, tickers: Optional[List[str]] = None, interval: str = "15m") -> None:
        tickers = tickers or (self.universe or [])
        if not tickers:
            self._log("ℹ️ No tickers available for yfinance pulse.")
            return

        end = datetime.now(timezone.utc)
        step = timedelta(minutes=15)
        if interval.lower().endswith("m"):
            step = timedelta(minutes=int(interval[:-1]))
        elif interval.lower().endswith("h"):
            step = timedelta(hours=int(interval[:-1]))
        elif interval.lower().endswith("d"):
            step = timedelta(days=int(interval[:-1]))
        total_inserted = 0
        for ticker in tickers:
            last_ts = self.db.latest_bar_timestamp(ticker)
            if last_ts:
                start = last_ts + step
                if start >= end:
                    start = last_ts
                self._log(
                    f"⏳ Fetching incremental bars for {ticker} since {start.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                start = end - timedelta(days=30)
                self._log(f"📦 No history for {ticker}; backfilling 30 days of data.")

            raw = fetch_bars(ticker, start=start, end=end, interval=interval)
            if raw.empty:
                print(f"[MarketData] No data for {ticker}, skipping.")
                continue

            raw.columns = [str(col).lower() for col in raw.columns]
            if "timestamp" not in raw.columns:
                self._log(
                    f"⚠️ Backfill skipped for {ticker}; timestamp column missing. Columns: {list(raw.columns)}",
                    level=logging.WARNING,
                )
                continue

            raw["symbol"] = raw.get("symbol", ticker)
            frame = raw[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame.dropna(subset=["timestamp"], inplace=True)
            float_cols = ["open", "high", "low", "close", "volume"]
            for col in float_cols:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame["volume"] = frame["volume"].fillna(0.0)
            frame.dropna(subset=["open", "high", "low", "close"], inplace=True)

            if frame.empty:
                self._log(f"ℹ️ No valid bars returned for {ticker}.")
                continue

            inserted, skipped = self.db.upsert_bars_multi(frame, update=False)
            total_inserted += inserted
            self._log(f"✅ Inserted {inserted} new rows for {ticker} (skipped {skipped}).")

            latest_ts = frame["timestamp"].max()
            self.db.set_last_sync(f"yfinance_{interval}", self._naive(latest_ts))

        if total_inserted:
            self._log(f"💾 yfinance pulse stored {total_inserted} bars ({interval}).")
        else:
            self._log("ℹ️ yfinance pulse produced no new bars.")

    # --------------------------------------------------------------------- #
    # TwelveData                                                            #
    # --------------------------------------------------------------------- #
    def pulse_twelvedata_backup(self, tickers: Optional[List[str]] = None) -> None:
        provider = self.providers.get("twelvedata")
        if not provider or not provider.enabled:
            self._log("ℹ️ TwelveData disabled; skipping pulse.")
            return

        token = provider.api_key
        if not token:
            self._log("⚠️ TwelveData API key missing.")
            return

        tickers = tickers or (self.universe or [])
        interval = self._provider_cfg("twelvedata").get("interval", "1h")
        outputsize = int(self._provider_cfg("twelvedata").get("outputsize", 120))

        rows_total = 0
        latest_ts: Optional[datetime] = None
        for ticker in tickers:
            params = {
                "symbol": ticker,
                "interval": interval,
                "apikey": token,
                "outputsize": outputsize,
            }
            payload = self._http_get(TWELVEDATA_SERIES_URL, params=params)
            if not payload or "values" not in payload:
                continue

            df = pd.DataFrame(payload["values"])
            if df.empty:
                continue

            df.rename(
                columns={"datetime": "ts", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"},
                inplace=True,
            )
            frame = pd.DataFrame(
                {
                    "ts": pd.to_datetime(df["ts"], utc=True),
                    "ticker": ticker,
                    "open": df["open"].astype(float),
                    "high": df["high"].astype(float),
                    "low": df["low"].astype(float),
                    "close": df["close"].astype(float),
                    "volume": df.get("volume", 0).fillna(0).astype(float),
                }
            )
            if frame.empty:
                continue

            self.db.upsert_bars(frame, table="hour_bars")
            rows_total += len(frame)
            max_ts = frame["ts"].max()
            if latest_ts is None or max_ts > latest_ts:
                latest_ts = max_ts

        if latest_ts:
            self.db.set_last_sync("twelvedata", self._naive(latest_ts))
        self._log(f"💾 TwelveData pulse inserted {rows_total} rows.")

    # --------------------------------------------------------------------- #
    # Financial Modeling Prep                                               #
    # --------------------------------------------------------------------- #
    def pulse_fmp_fundamentals(self, tickers: Optional[List[str]] = None) -> None:
        provider = self.providers.get("fmp")
        if not provider or not provider.enabled:
            self._log("ℹ️ FMP disabled; skipping fundamentals pulse.")
            return

        token = provider.api_key
        if not token:
            self._log("⚠️ FMP API key missing.")
            return

        tickers = tickers or (self.universe or [])
        for ticker in tickers:
            url = FMP_PROFILE_URL.format(ticker=ticker)
            params = {"apikey": token}
            payload = self._http_get(url, params=params)
            if not payload:
                continue
            ts = self._naive(datetime.now(timezone.utc))
            self.db.upsert_payload("fmp", ticker, "fundamentals", ts, payload)
            self._log(f"📄 Stored FMP fundamentals for {ticker}.")

        self.db.set_last_sync("fmp_fundamentals", self._naive(datetime.now(timezone.utc)))

    # --------------------------------------------------------------------- #
    # MarketStack                                                           #
    # --------------------------------------------------------------------- #
    def pulse_marketstack_eod(self, tickers: Optional[List[str]] = None) -> None:
        provider = self.providers.get("marketstack")
        if not provider or not provider.enabled:
            self._log("ℹ️ MarketStack disabled; skipping.")
            return

        token = provider.api_key
        if not token:
            self._log("⚠️ MarketStack access key missing.")
            return

        tickers = tickers or (self.universe or [])
        for batch in self._chunks(tickers, size=20):
            params = {"access_key": token, "symbols": ",".join(batch), "limit": 500}
            payload = self._http_get(MARKETSTACK_EOD_URL, params=params, timeout=30)
            if not payload or "data" not in payload:
                continue

            df = pd.DataFrame(payload["data"])
            if df.empty:
                continue

            frame = pd.DataFrame(
                {
                    "ts": pd.to_datetime(df["date"], utc=True),
                    "ticker": df["symbol"],
                    "open": df["open"].astype(float),
                    "high": df["high"].astype(float),
                    "low": df["low"].astype(float),
                    "close": df["close"].astype(float),
                    "volume": df.get("volume", 0).fillna(0).astype(float),
                }
            )
            self.db.upsert_bars(frame, table="daily_bars")

        self.db.set_last_sync("marketstack", self._naive(datetime.now(timezone.utc)))
        self._log("🌅 MarketStack EOD pulse complete.")

    # --------------------------------------------------------------------- #
    # Compatibility helpers (legacy)                                        #
    # --------------------------------------------------------------------- #
    def backfill_yf(self, interval: str = "1h", max_days: int = 60) -> None:
        """Legacy helper retained for backwards compatibility."""
        tickers = self.universe or ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]
        self._log(f"📡 Historical backfill from yfinance ({interval}) for {len(tickers)} tickers.")
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max_days)
        total = 0
        for ticker in tickers:
            raw = safe_download(ticker, start=start, end=end, interval=interval)
            if raw is None or raw.empty:
                print(f"[MarketData] No data for {ticker}, skipping.")
                continue

            raw = raw.reset_index()
            if isinstance(raw.columns, pd.MultiIndex):
                flattened_cols = []
                for col in raw.columns:
                    parts = [str(part).strip() for part in col if part not in (None, "")]
                    flattened_cols.append("_".join(parts) if parts else "")
                raw.columns = flattened_cols
            raw.columns = [str(c).lower() for c in raw.columns]

            if "datetime" in raw.columns:
                ts_series = raw["datetime"]
            elif "date" in raw.columns:
                ts_series = raw["date"]
            elif "index" in raw.columns:
                ts_series = raw["index"]
            else:
                self._log(
                    f"⚠️  Backfill skipped for {ticker}; timestamp column missing. Columns: {list(raw.columns)}"
                )
                continue

            def _first_available(column_name: str) -> pd.Series:
                if column_name in raw.columns:
                    return raw[column_name]
                matches = [
                    col
                    for col in raw.columns
                    if col.endswith(f"_{column_name}") or col.startswith(f"{column_name}_")
                ]
                if matches:
                    return raw[matches[0]]
                for col in raw.columns:
                    parts = [part.strip() for part in col.split("_") if part.strip()]
                    if column_name in parts:
                        return raw[col]
                raise KeyError(column_name)

            frame = pd.DataFrame(
                {
                    "ts": pd.to_datetime(ts_series, utc=True),
                    "ticker": ticker,
                    "open": _first_available("open").astype(float),
                    "high": _first_available("high").astype(float),
                    "low": _first_available("low").astype(float),
                    "close": _first_available("close").astype(float),
                    "volume": _first_available("volume").fillna(0).astype(float),
                }
            )
            self.db.upsert_bars(frame)
            total += len(frame)

        self._log(f"💾 Backfill complete — {total} rows inserted.")

    def update_from_alpaca(self, timeframe: str = "1Min") -> None:
        """Incrementally refresh market bars using alpaca-py."""
        if self.alpaca_data_client is None:
            self._log("⚠️ Alpaca credentials missing — skipping incremental updates.")
            return

        end = datetime.now(timezone.utc)
        since = self.db.get_last_sync(f"alpaca_{timeframe}") or (end - timedelta(days=1))
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

        tickers = self.universe or ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]
        timeframe_obj = self._timeframe_from_label(timeframe)

        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            start=since,
            end=end,
            timeframe=timeframe_obj,
            feed=self._alpaca_feed,
            limit=10000,
        )
        try:
            bars = self.alpaca_data_client.get_stock_bars(request)
        except Exception as exc:  # noqa: BLE001
            self._log(f"⚠️ Alpaca bars fetch failed: {exc}")
            return

        df = getattr(bars, "df", pd.DataFrame())
        if df.empty:
            self._log("ℹ️ Alpaca returned no new bars.")
            return

        df = df.reset_index().rename(columns={"symbol": "symbol", "timestamp": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]

        total = 0
        latest_ts: Optional[datetime] = None
        for symbol, chunk in df.groupby("symbol"):
            if chunk.empty:
                continue
            self.db.upsert_bars_multi(chunk)
            total += len(chunk)
            symbol_max = chunk["timestamp"].max()
            if latest_ts is None or symbol_max > latest_ts:
                latest_ts = symbol_max

        if latest_ts:
            self.db.set_last_sync(f"alpaca_{timeframe}", self._naive(latest_ts))
        self._log(f"💾 Alpaca incremental update inserted {total} rows.")
