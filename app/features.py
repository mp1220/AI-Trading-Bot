from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_z(series: pd.Series) -> pd.Series:
    """Safely compute z-score normalization for a pandas Series."""
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s), dtype="float64"), index=s.index)
    z = (s - mean) / std
    return pd.Series(z.to_numpy(dtype="float64", na_value=np.nan), index=s.index)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `frame` with normalized and derived feature columns."""
    if frame is None or frame.empty:
        return frame

    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index(drop=False)

    frame = frame.copy()
    frame.reset_index(drop=True, inplace=True)

    # Convert to numeric safely
    for col in ("open", "high", "low", "close", "volume"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Volume z-score
    if "volume" in frame.columns:
        frame["volume_z"] = _safe_z(frame["volume"]).to_numpy(dtype="float64")

    # High-low range and returns
    if {"high", "low", "close"}.issubset(frame.columns):
        hl_range = (frame["high"] - frame["low"]).abs()
        frame["hl_range"] = pd.to_numeric(hl_range, errors="coerce").to_numpy(dtype="float64")
        returns = frame["close"].pct_change()
        frame["ret_1"] = pd.to_numeric(returns, errors="coerce").to_numpy(dtype="float64")

    # Ensure derived columns always exist
    for col in ["volume_z", "hl_range", "ret_1"]:
        if col not in frame.columns:
            frame[col] = 0.0

    # Standardize index
    if {"symbol", "timestamp"}.issubset(frame.columns):
        frame["symbol"] = frame["symbol"].astype("object")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.set_index(["symbol", "timestamp"])

    return frame


# === Legacy indicator functions for compatibility ===
def build_supervised_dataset(frame: pd.DataFrame, shift: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Construct feature/target matrices using ``build_features`` pipeline."""
    frame = build_features(frame)
    feature_cols = ["ret_1", "hl_range", "volume_z"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=feature_cols), pd.Series([], dtype="float64")

    if {"close"}.issubset(frame.columns):
        working = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
        if "symbol" not in working.columns:
            working["symbol"] = "UNKNOWN"
        if "timestamp" not in working.columns:
            if "time" in working.columns:
                working.rename(columns={"time": "timestamp"}, inplace=True)
            else:
                working["timestamp"] = pd.NaT
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
        sorted_frame = working.sort_values(["symbol", "timestamp"], ascending=True)
        future_close = sorted_frame.groupby("symbol")["close"].shift(-shift)
        sorted_frame["future_close"] = future_close
        merged = sorted_frame.dropna(subset=["future_close"]).copy()
        merged["target"] = (merged["future_close"] > merged["close"]).astype(int)
        merged.set_index(["symbol", "timestamp"], inplace=True)
        frame = merged
    else:
        frame = frame.copy()

    for column in feature_cols:
        if column not in frame.columns:
            frame[column] = 0.0

    features = frame[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    target = frame.get("target")
    if target is None:
        target = pd.Series([], dtype="float64")
    else:
        target = target.astype(int)
    return features, target


def ema(series: pd.Series, span: int = 14) -> pd.Series:
    """Exponential Moving Average."""
    if series is None or series.empty:
        return pd.Series([], dtype="float64")
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    if series is None or series.empty:
        return pd.Series([], dtype="float64")
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def realized_vol(series: pd.Series, window: int = 14) -> pd.Series:
    """Realized volatility based on log returns."""
    if series is None or series.empty:
        return pd.Series([], dtype="float64")
    returns = np.log(series / series.shift(1))
    return returns.rolling(window=window).std() * np.sqrt(window)
