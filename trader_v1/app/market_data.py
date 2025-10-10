"""Unified market data fetching utilities."""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def _resolve_timeframe(interval: str) -> TimeFrame:
    """Map string intervals to Alpaca TimeFrame objects."""
    cleaned = (interval or "").strip().lower()
    if cleaned.endswith("m"):
        amount = int(cleaned[:-1] or 1)
        return TimeFrame(amount, TimeFrameUnit.Minute)
    if cleaned.endswith("h"):
        amount = int(cleaned[:-1] or 1)
        return TimeFrame(amount, TimeFrameUnit.Hour)
    if cleaned.endswith("d"):
        amount = int(cleaned[:-1] or 1)
        return TimeFrame(amount, TimeFrameUnit.Day)
    return TimeFrame(1, TimeFrameUnit.Hour)


def fetch_bars(
    symbol: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = "1h",
    retries: int = 3,
) -> pd.DataFrame:
    """
    Unified data fetcher:
    - Alpaca for live/incremental bars
    - yfinance for backfill or fallback
    """

    # --- Normalize ticker for Alpaca ---
    # Alpaca expects BRK.B instead of BRK-B, etc.
    symbol_alpaca = symbol.replace("-", ".")

    def fetch_alpaca() -> pd.DataFrame:
        key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            print(f"[MarketData] ⚠️ Alpaca credentials missing; skipping live fetch for {symbol}")
            return pd.DataFrame()

        try:
            client = StockHistoricalDataClient(key, secret)
            req = StockBarsRequest(
                symbol_or_symbols=symbol_alpaca,  # use normalized ticker
                timeframe=_resolve_timeframe(interval),
                start=start,
                end=end,
                limit=10000,
                feed=DataFeed.IEX,
            )
            bars = client.get_stock_bars(req).df
            if bars is None or bars.empty:
                return pd.DataFrame()
            bars = bars.reset_index()
            if "timestamp" not in bars.columns:
                bars.rename(columns={"time": "timestamp"}, inplace=True, errors="ignore")
            bars["symbol"] = bars.get("symbol", symbol)
            for column in ("open", "high", "low", "close", "volume"):
                if column not in bars.columns:
                    bars[column] = 0.0 if column == "volume" else None
            frame = bars[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame.dropna(subset=["timestamp"], inplace=True)
            print(f"[MarketData] ✅ Loaded {len(frame)} bars for {symbol_alpaca} from Alpaca IEX")
            return frame
        except Exception as exc:  # noqa: BLE001
            print(f"[MarketData] ⚠️ Alpaca fetch failed for {symbol_alpaca}: {exc}")
            return pd.DataFrame()

    def fetch_yf() -> pd.DataFrame:
        def _normalize(df: pd.DataFrame) -> pd.DataFrame:
            df = df.reset_index()
            if "Datetime" in df.columns:
                df.rename(columns={"Datetime": "timestamp"}, inplace=True)
            elif "Date" in df.columns:
                df.rename(columns={"Date": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["symbol"] = symbol
            frame = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            for column in ("open", "high", "low", "close", "volume"):
                if column not in frame.columns:
                    frame[column] = 0.0 if column == "volume" else None
            frame = frame[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
            frame.dropna(subset=["timestamp"], inplace=True)
            for column in ("open", "high", "low", "close", "volume"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame["volume"] = frame["volume"].fillna(0.0)
            frame.dropna(subset=["open", "high", "low", "close"], inplace=True)
            return frame

        for attempt in range(retries):
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                )
                if df is not None and not df.empty:
                    frame = _normalize(df)
                    print(f"[MarketData] ✅ Loaded {len(frame)} bars for {symbol} from yfinance backfill")
                    return frame
            except Exception as exc:  # noqa: BLE001
                if attempt < retries - 1:
                    time.sleep(2**attempt)
                else:
                    print(f"[MarketData] ⚠️ yfinance failed for {symbol}: {exc}")

        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            if df is not None and not df.empty:
                frame = _normalize(df)
                print(f"[MarketData] ✅ Fallback daily bars for {symbol}")
                return frame
        except Exception:
            pass

        print(f"[MarketData] ⚠️ No data retrieved for {symbol} via yfinance.")
        return pd.DataFrame()

    bars = fetch_alpaca()
    if not bars.empty:
        return bars

    print(f"[MarketData] ⚠️ Falling back to yfinance for {symbol} (empty response from Alpaca)")
    fallback = fetch_yf()
    if fallback.empty:
        print(f"[MarketData] ⚠️ No market data available for {symbol}.")
    return fallback
