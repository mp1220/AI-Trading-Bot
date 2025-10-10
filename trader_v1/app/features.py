from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s), dtype="float64"), index=s.index)
    z = (s - mean) / std
    return pd.Series(z.to_numpy(dtype="float64", na_value=np.nan), index=s.index)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with aligned derived feature columns."""

    if frame is None or frame.empty:
        return frame

    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index(drop=False)

    frame = frame.copy()
    frame.reset_index(drop=True, inplace=True)

    for col in ("open", "high", "low", "close", "volume"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "volume" in frame.columns:
        frame["volume_z"] = _safe_z(frame["volume"]).to_numpy(dtype="float64")

    if {"high", "low", "close"}.issubset(frame.columns):
        hl_range = (frame["high"] - frame["low"]).abs()
        frame["hl_range"] = pd.to_numeric(hl_range, errors="coerce").to_numpy(dtype="float64")
        returns = frame["close"].pct_change()
        frame["ret_1"] = pd.to_numeric(returns, errors="coerce").to_numpy(dtype="float64")

    # --- Ensure derived columns always exist ---
    for col in ["volume_z", "hl_range", "ret_1"]:
        if col not in frame.columns:
            frame[col] = 0.0

    if {"symbol", "timestamp"}.issubset(frame.columns):
        frame["symbol"] = frame["symbol"].astype("object")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.set_index(["symbol", "timestamp"])

    return frame
