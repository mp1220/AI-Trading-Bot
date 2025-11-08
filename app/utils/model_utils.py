"""Utilities for persisting and rotating Trader_Model artifacts."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None
import pickle

from .logging import get_logger


class ModelManager:
    """Handles saving, rotating, and recording metadata for models."""

    def __init__(
        self,
        base_dir: Path,
        db,
        timezone: dt.tzinfo,
        keep_online_hours: int = 48,
        keep_daily_days: int = 30,
        model_prefix: str = "trader_model",
    ) -> None:
        self._logger = get_logger("ModelManager")
        self._db = db
        self._tz = timezone
        self.base_dir = base_dir
        self.daily_dir = base_dir / "daily"
        self.online_dir = base_dir / "online"
        self.backup_dir = base_dir / "backups"
        self._prefix = model_prefix
        for path in (self.base_dir, self.daily_dir, self.online_dir, self.backup_dir):
            path.mkdir(parents=True, exist_ok=True)

        self._keep_online_hours = keep_online_hours
        self._keep_daily_days = keep_daily_days

    def save_model(
        self,
        model: Any,
        mode: str,
        metrics: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        now = dt.datetime.now(tz=self._tz)
        suffix = now.strftime("%Y%m%dT%H%M%S")
        directory = self._resolve_dir(mode)
        path = directory / f"{self._prefix}-{mode}_{suffix}.pkl"
        if joblib:
            joblib.dump(model, path)
        else:  # pragma: no cover
            with path.open("wb") as fh:
                pickle.dump(model, fh)
        self._logger.info("💾 Saved %s model → %s", mode, path)

        payload = {
            "mode": mode,
            "path": str(path),
            "created_at": now,
            **metrics,
            **(extra or {}),
        }
        self._record_metadata(payload)
        self._rotate(mode)
        self._write_meta(now, mode, metrics, extra or {})
        return path

    def _resolve_dir(self, mode: str) -> Path:
        mode = mode.lower()
        if mode == "online":
            return self.online_dir
        if mode == "daily":
            return self.daily_dir
        if mode == "backup":
            return self.backup_dir
        raise ValueError(f"Unknown model mode {mode}")

    def _record_metadata(self, payload: Dict[str, Any]) -> None:
        sql = (
            "INSERT INTO model_versions (mode, path, created_at, metrics) "
            "VALUES (%(mode)s, %(path)s, %(created_at)s, CAST(%(metrics)s AS JSONB))"
        )
        metrics = {
            k: v
            for k, v in payload.items()
            if k not in ("mode", "path", "created_at")
        }
        metrics_json = json.dumps(metrics, default=str)
        params = {
            "mode": payload["mode"],
            "path": payload["path"],
            "created_at": payload["created_at"],
            "metrics": metrics_json,
        }
        try:
            self._db.execute(sql, params)
        except Exception as exc:  # noqa: BLE001
            self._logger.error("⚠️  Failed to record model metadata: %s", exc)

    def _rotate(self, mode: str) -> None:
        now = dt.datetime.now(tz=self._tz)
        directory = self._resolve_dir(mode)
        keep_delta = dt.timedelta(hours=self._keep_online_hours) if mode == "online" else dt.timedelta(days=self._keep_daily_days)
        pattern = f"{self._prefix}-{mode}_*.pkl"
        for path in sorted(directory.glob(pattern)):
            created = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=self._tz)
            if now - created > keep_delta:
                path.unlink(missing_ok=True)
                self._logger.info("🧹 Removed expired %s model %s", mode, path.name)

    def latest_metrics(self, mode: str) -> Optional[Dict[str, Any]]:
        sql = (
            "SELECT metrics FROM model_versions WHERE mode=%(mode)s "
            "ORDER BY created_at DESC LIMIT 1"
        )
        df = self._db.to_df(sql, {"mode": mode})
        if df.empty:
            return None
        metrics = df.iloc[0]["metrics"]
        if isinstance(metrics, dict):
            return metrics
        return None

    def _write_meta(
        self,
        timestamp: dt.datetime,
        mode: str,
        metrics: Dict[str, Any],
        extra: Dict[str, Any],
    ) -> None:
        folder = self.base_dir / f"{self._prefix}_{timestamp.strftime('%Y%m%d_%H%M')}"
        folder.mkdir(parents=True, exist_ok=True)
        meta_path = folder / "meta.json"
        meta_payload = {
            "timestamp": timestamp.isoformat(),
            "mode": mode,
            "model_type": extra.get("model_type", metrics.get("model_type", "unknown")),
            "training_samples": extra.get("training_samples", metrics.get("training_samples")),
            "accuracy": metrics.get("accuracy"),
            "sources": extra.get("sources", []),
        }
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta_payload, fh, indent=2)
        self._logger.info("🗃️  Wrote model metadata %s", meta_path.name)
