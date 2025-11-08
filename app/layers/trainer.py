"""Online and batch training layer for Trader_Model."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover
    RandomForestClassifier = None
    SGDClassifier = None

    def accuracy_score(y_true, y_pred):
        return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0

    def classification_report(y_true, y_pred, output_dict=False):
        if not len(y_true):
            metrics = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}
        else:
            positives = (y_true == 1)
            tp = float(np.sum((y_pred == 1) & positives))
            fp = float(np.sum((y_pred == 1) & (~positives)))
            fn = float(np.sum((y_pred == 0) & positives))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            metrics = {"precision": precision, "recall": recall, "f1-score": f1}
        return {"1": metrics} if output_dict else metrics

    def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
        split_idx = max(1, int(len(X) * (1 - test_size)))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    class _DummySGDClassifier:
        def __init__(self, *args, **kwargs):
            self._prob = 0.5

        def partial_fit(self, X, y, classes=None):
            if len(y):
                self._prob = float(np.clip(np.mean(y), 0.01, 0.99))
            return self

        def predict_proba(self, X):
            probs = np.full((len(X), 2), [1 - self._prob, self._prob], dtype=float)
            return probs

    class _DummyRandomForestClassifier:
        def __init__(self, *args, **kwargs):
            self._majority = 0

        def fit(self, X, y):
            if len(y):
                self._majority = int(np.round(np.mean(y)))
            return self

        def predict(self, X):
            return np.full(len(X), self._majority, dtype=int)

    SGDClassifier = _DummySGDClassifier
    RandomForestClassifier = _DummyRandomForestClassifier

from app.features import ema, realized_vol, rsi
from app.layers.base import BaseLayer
from app.utils.model_utils import ModelManager


class TrainerLayer(BaseLayer):
    """Handles both online incremental updates and end-of-day retraining."""

    def __init__(
        self,
        db,
        config: Dict[str, Any],
        error_manager,
        models_dir: Path,
        timezone: dt.tzinfo,
        model_prefix: str,
    ) -> None:
        super().__init__("Trainer", db, config, error_manager)
        self._tz = timezone
        self._model_manager = ModelManager(
            models_dir,
            db,
            timezone,
            model_prefix=model_prefix,
        )
        self._online_model: Optional[SGDClassifier] = None
        self._classes = np.array([0, 1])
        self._online_window_hours = int(config.get("training", {}).get("online_window_hours", 6))
        self._daily_window_days = int(config.get("training", {}).get("daily_window_days", 7))
        data_sources = config.get("data_sources", {})
        self._sources = [name for name, enabled in data_sources.items() if enabled]

    async def run(self) -> Dict[str, Any] | None:
        """Default entrypoint for scheduler; delegates to online routine."""
        return await self.run_online()

    async def run_online(self) -> Dict[str, Any]:
        await self._update_status("🟡")
        try:
            metrics = await self._guard("TrainerOnline", self._train_online)
            self._info("🧠 Online training metrics: %s", json.dumps(metrics))
            await self._update_status("🟢")
            return metrics
        except Exception as exc:  # noqa: BLE001
            await self._update_status("🔴")
            self.logger.error("⚠️  Online training failed: %s", exc, exc_info=False)
            return {"status": "error", "error": str(exc)}

    async def run_daily(self) -> Dict[str, Any]:
        await self._update_status("🟡")
        try:
            metrics = await self._guard("TrainerDaily", self._train_daily)
            self._info("🧠 Daily retrain metrics: %s", json.dumps(metrics))
            await self._update_status("🟢")
            return metrics
        except Exception as exc:  # noqa: BLE001
            await self._update_status("🔴")
            self.logger.error("⚠️  Daily retrain failed: %s", exc, exc_info=False)
            return {"status": "error", "error": str(exc)}

    async def _train_online(self) -> Dict[str, Any]:
        window_start = dt.datetime.now(tz=self._tz) - dt.timedelta(hours=self._online_window_hours)
        dataset = await self._prepare_dataset(window_start)
        if dataset is None:
            return {"status": "empty"}

        X, y = dataset
        if self._online_model is None:
            self._online_model = SGDClassifier(loss="log_loss", max_iter=1000, learning_rate="optimal")
            self._online_model.partial_fit(X, y, classes=self._classes)
        else:
            self._online_model.partial_fit(X, y)

        probs = self._online_model.predict_proba(X)[:, 1]
        preds = (probs > 0.5).astype(int)
        acc = float(accuracy_score(y, preds))

        self._model_manager.save_model(
            self._online_model,
            mode="online",
            metrics={"accuracy": acc, "window_hours": self._online_window_hours},
            extra={
                "model_type": type(self._online_model).__name__,
                "training_samples": int(len(y)),
                "sources": self._sources,
            },
        )
        return {"accuracy": acc, "samples": int(len(y))}

    async def _train_daily(self) -> Dict[str, Any]:
        window_start = dt.datetime.now(tz=self._tz) - dt.timedelta(days=self._daily_window_days)
        dataset = await self._prepare_dataset(window_start)
        if dataset is None:
            return {"status": "empty"}
        X, y = dataset

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))

        previous = self._model_manager.latest_metrics("daily")
        prev_acc = float(previous.get("accuracy", 0.0)) if previous else 0.0
        if previous and acc < prev_acc:
            self.logger.warning(
                "⚠️  Daily retrain accuracy %.3f < previous %.3f. Keeping prior model.",
                acc,
                prev_acc,
            )
            self._model_manager.save_model(
                model,
                mode="backup",
                metrics={"accuracy": acc, "note": "candidate rejected"},
            )
            return {"accuracy": acc, "status": "rejected"}

        path = self._model_manager.save_model(
            model,
            mode="daily",
            metrics={"accuracy": acc, "window_days": self._daily_window_days},
            extra={
                "model_type": type(model).__name__,
                "training_samples": int(len(y)),
                "sources": self._sources,
            },
        )
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics_pos = report.get("1") or {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}
        await asyncio.to_thread(
            self.db.executemany,
            self._performance_sql(),
            [
                {
                    "recorded_at": dt.datetime.now(tz=self._tz).replace(tzinfo=None),
                    "mode": "daily",
                    "accuracy": acc,
                    "precision": float(metrics_pos.get("precision", 0.0)),
                    "recall": float(metrics_pos.get("recall", 0.0)),
                    "f1": float(metrics_pos.get("f1-score", 0.0)),
                    "notes": f"path={path}",
                }
            ],
        )
        return {"accuracy": acc, "samples": int(len(y)), "path": str(path)}

    async def _prepare_dataset(self, window_start: dt.datetime) -> Optional[tuple[np.ndarray, np.ndarray]]:
        sql = """
            SELECT timestamp AS ts, symbol AS ticker, close, volume
            FROM market_bars
            WHERE timestamp >= %(start)s
            ORDER BY symbol, timestamp
        """
        bars = await asyncio.to_thread(self.db.to_df, sql, {"start": window_start})
        if bars.empty:
            self._info("ℹ️  No bars available for training window starting %s", window_start)
            return None

        frames = []
        for _, group in bars.groupby("ticker"):
            group = group.sort_values("ts").copy()
            group["ema_fast"] = ema(group["close"], 12)
            group["ema_slow"] = ema(group["close"], 26)
            group["rsi"] = rsi(group["close"], 14)
            group["vol"] = realized_vol(group["close"], 20)
            group["return_fwd"] = group["close"].shift(-3) / group["close"] - 1
            frames.append(group)

        dataset = pd.concat(frames, ignore_index=True).dropna()
        if dataset.empty:
            return None

        y = (dataset["return_fwd"] > 0).astype(int).values
        feature_cols = ["ema_fast", "ema_slow", "rsi", "vol", "volume"]
        X = dataset[feature_cols].values
        return X, y

    @staticmethod
    def _performance_sql() -> str:
        return """
            INSERT INTO performance_metrics (recorded_at, mode, accuracy, precision, recall, f1, notes)
            VALUES (%(recorded_at)s, %(mode)s, %(accuracy)s, %(precision)s, %(recall)s, %(f1)s, %(notes)s)
        """
    
# ----------------------------------------------------------------------
# CLI ENTRYPOINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import asyncio
    import pytz
    from app.config import load_config
    from app.storage import DB
    from app.errors import ErrorManager

    parser = argparse.ArgumentParser(description="Manual training run for Trader_Model.")
    parser.add_argument("--start", required=False, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=False, help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--validate", required=False, help="Validation date (YYYY-MM-DD)")
    parser.add_argument("--save", action="store_true", help="Save model after training")
    args = parser.parse_args()

    # Initialize environment
    cfg = load_config()
    db = DB()
    tz = pytz.UTC
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    trainer = TrainerLayer(
        db=db,
        config=cfg,
        error_manager=ErrorManager(),
        models_dir=models_dir,
        timezone=tz,
        model_prefix="trader_model",
    )

    async def run_manual():
        print("📂 Loading training dataset...")
        metrics = await trainer.run_daily()
        print("\n✅ Training complete.")
        print(json.dumps(metrics, indent=2))

    asyncio.run(run_manual())

