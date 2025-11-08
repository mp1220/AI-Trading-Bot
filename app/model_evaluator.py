"""Model evaluation utilities."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:  # pragma: no cover - joblib optional at runtime
    import joblib
except ImportError:  # pragma: no cover
    joblib = None
import pickle

from app.features import build_features
from app.storage import DB


class ModelEvaluator:
    def __init__(self, db: DB, models_dir: Path) -> None:
        self.db = db
        self.models_dir = models_dir
        self.logger = logging.getLogger("ModelEvaluator")

    async def evaluate(self) -> Optional[dict[str, float]]:
        return await asyncio.to_thread(self._evaluate_sync)

    def _evaluate_sync(self) -> Optional[dict[str, float]]:
        model_path = self._latest_model()
        if model_path is None:
            self.logger.warning("Skipping evaluation — no trained model found")
            return None

        model = self._load_model(model_path)
        if model is None:
            return None

        frame = self._load_eval_frame()
        if frame is None or frame.empty:
            self.logger.warning("Skipping evaluation — insufficient data")
            return None

        features, target = self._build_features(frame)
        if features.empty:
            self.logger.warning("Skipping evaluation — feature matrix empty")
            return None

        preds = model.predict(features.values)
        probas = model.predict_proba(features.values)[:, 1] if hasattr(model, "predict_proba") else preds
        accuracy = float(accuracy_score(target, preds))
        precision = float(precision_score(target, preds, zero_division=0))
        recall = float(recall_score(target, preds, zero_division=0))
        f1 = float(f1_score(target, preds, zero_division=0))
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "records": float(len(target)),
            "avg_proba": float(np.mean(probas)),
        }
        self.logger.info(
            "📊 Model evaluation (%s): acc=%.3f precision=%.3f recall=%.3f f1=%.3f",
            model_path.name,
            accuracy,
            precision,
            recall,
            f1,
        )
        return metrics

    def _latest_model(self) -> Optional[Path]:
        candidates = list(self.models_dir.glob("model_*.pkl"))
        return max(candidates, default=None) if candidates else None

    def _load_model(self, path: Path):
        try:
            if joblib is not None:
                return joblib.load(path)
            with path.open("rb") as fh:  # pragma: no cover
                return pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to load model %s: %s", path, exc)
            return None

    def _load_eval_frame(self) -> Optional[pd.DataFrame]:
        sql = """
            SELECT symbol AS ticker, timestamp AS ts, close, volume
            FROM market_bars
            ORDER BY ts DESC
            LIMIT 2000
        """
        try:
            return self.db.to_df(sql)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to load evaluation frame: %s", exc)
            return None

    def _build_features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        frame = build_features(frame)
        future_close = (
            frame.sort_values(["ticker", "ts"])
            .groupby("ticker")["close"]
            .shift(-1)
        )
        frame["future_close"] = pd.Series(future_close, index=frame.index, dtype="float64")
        frame.dropna(subset=["future_close"], inplace=True)
        frame["target"] = (frame["future_close"] > frame["close"]).astype(int)
        features = frame[["ret_1", "hl_range", "volume_z"]].replace([np.inf, -np.inf], 0.0)
        target = frame["target"].astype(int)
        return features, target
