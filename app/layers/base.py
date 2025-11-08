"""Base layer for orchestrated jobs."""
from __future__ import annotations

import asyncio
import pandas as pd
from typing import Any, Dict, Iterable, Optional

from app.utils.error_manager import ErrorManager
from app.utils.logging import get_logger


class BaseLayer:
    """Shared functionality across ingestion/analysis layers."""

    def __init__(
        self,
        name: str,
        db,
        config: Dict[str, Any],
        error_manager: ErrorManager,
    ) -> None:
        self.name = name
        self.db = db
        self.config = config
        self.errors = error_manager
        self.logger = get_logger(f"Layer.{name}")
        self._status = "🟡"
        self._lock = asyncio.Lock()
        self._quiet = bool(config.get("runtime", {}).get("quiet_mode", False))

    async def run(self) -> None:
        raise NotImplementedError

    async def _guard(self, api: str, func, *args, **kwargs):
        return await self.errors.guard(api, func, *args, **kwargs)

    async def _update_status(self, status: str) -> None:
        async with self._lock:
            self._status = status

    async def status(self) -> str:
        async with self._lock:
            return self._status

    def validate_frame(self, frame: pd.DataFrame, required: Iterable[str]) -> bool:
        if frame is None or frame.empty:
            self.logger.warning("⚠️  %s produced empty frame", self.name)
            return False
        missing = [col for col in required if col not in frame.columns]
        if missing:
            self.logger.warning("⚠️  %s missing columns %s", self.name, missing)
            return False
        if frame[required].isna().any().any():
            self.logger.warning("⚠️  %s contains NaN values in critical columns", self.name)
            return False
        return True

    def _store_dataframe(self, table: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        cols = ",".join(df.columns)
        placeholders = ",".join(f":{col}" for col in df.columns)
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        rows = df.to_dict(orient="records")
        self.db.executemany(sql, rows)

    def _info(self, message: str, *args) -> None:
        if not self._quiet:
            self.logger.info(message, *args)
