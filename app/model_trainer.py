"""Asynchronous model training pipeline."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - optional joblib fallback
    import joblib
except ImportError:  # pragma: no cover
    joblib = None
import pickle

from app.features import build_features
from app.storage import DB


class ModelTrainer:
    """Coordinates loading data, training, and persisting the latest model."""

    def __init__(self, db: DB, models_dir: Path) -> None:
        self.db = db
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ModelTrainer")
        self._latest_path: Optional[Path] = None

    async def train(self) -> Optional[Tuple[Path, float]]:
        """Run the full training cycle asynchronously."""

        return await asyncio.to_thread(self._train_sync)

    def latest_model_path(self) -> Optional[Path]:
        if self._latest_path and self._latest_path.exists():
            return self._latest_path
        candidates = list(self.models_dir.glob("model_*.pkl"))
        return max(candidates, default=None) if candidates else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _train_sync(self) -> Optional[Tuple[Path, float]]:
        frame = self._load_training_frame()
        if frame is None or frame.empty:
            self.logger.warning("Training skipped — insufficient market data")
            return None

        features, target = self._build_features(frame)
        if features.empty or target.empty:
            self.logger.warning("Training skipped — no features available")
            return None

        x_train, x_test, y_train, y_test = train_test_split(
            features.values,
            target.values,
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", SGDClassifier(loss="log_loss", max_iter=1000, n_jobs=None)),
        ])

        pipeline.fit(x_train, y_train)
        accuracy = float(pipeline.score(x_test, y_test))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        model_path = self.models_dir / f"model_{timestamp}.pkl"

        if joblib is not None:
            joblib.dump(pipeline, model_path)
        else:  # pragma: no cover
            with model_path.open("wb") as fh:
                pickle.dump(pipeline, fh)

        self.logger.info("✅ Model trained (accuracy=%.3f) → %s", accuracy, model_path.name)
        self._latest_path = model_path
        return model_path, accuracy

    def _load_training_frame(self) -> Optional[pd.DataFrame]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        sql = """
            SELECT symbol AS ticker, timestamp AS ts, close, volume
            FROM market_bars
            WHERE timestamp >= %(cutoff)s
            ORDER BY ts ASC
        """
        try:
            frame = self.db.to_df(sql, {"cutoff": cutoff})
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to load training data: %s", exc)
            return None
        return frame

    def _build_features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        frame = build_features(frame)

        if {"close"}.issubset(frame.columns):
            future_close = (
                frame.sort_values(["ticker", "ts"])
                .groupby("ticker")["close"]
                .shift(-1)
            )
            frame["future_close"] = pd.Series(future_close, index=frame.index, dtype="float64")

        if {"future_close", "close"}.issubset(frame.columns):
            frame.dropna(subset=["future_close"], inplace=True)
            frame["target"] = (frame["future_close"] > frame["close"]).astype(int)

        features = frame[["ret_1", "hl_range", "volume_z"]].replace([np.inf, -np.inf], 0.0)
        target = frame["target"].astype(int)
        return features, target
