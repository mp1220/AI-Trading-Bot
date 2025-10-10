"""Centralised error handling utilities."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, Optional

from .logging import get_logger

Retryable = Callable[..., Awaitable[Any]] | Callable[..., Any]


@dataclass
class FailureState:
    count: int = 0
    disabled_until: float = 0.0
    history: Deque[str] = field(default_factory=lambda: deque(maxlen=50))


class ErrorManager:
    """Tracks API health, applies retries, and logs persistent failures."""

    def __init__(
        self,
        log_dir: Path,
        max_retries: int = 3,
        failure_threshold: int = 5,
        disable_minutes: int = 30,
    ) -> None:
        self._logger = get_logger("ErrorManager")
        self._max_retries = max_retries
        self._failure_threshold = failure_threshold
        self._disable_interval = disable_minutes * 60

        log_dir.mkdir(parents=True, exist_ok=True)
        self._error_log_path = log_dir / "errors.log"

        self._failures: Dict[str, FailureState] = defaultdict(FailureState)
        self._lock = asyncio.Lock()

    def is_enabled(self, api: str) -> bool:
        state = self._failures[api]
        if state.disabled_until and time.time() < state.disabled_until:
            return False
        return True

    async def _record_failure(self, api: str, message: str) -> None:
        async with self._lock:
            state = self._failures[api]
            state.count += 1
            state.history.append(message)
            self._write_error_log(api, message)

    async def _record_success(self, api: str) -> None:
        async with self._lock:
            state = self._failures[api]
            if state.count:
                self._logger.info("🟢 %s recovered after %d failures", api, state.count)
            self._failures[api] = FailureState()

    def _write_error_log(self, api: str, message: str) -> None:
        with self._error_log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {api} | {message}\n")

    async def guard(
        self,
        api: str,
        func: Retryable,
        *args: Any,
        retries: Optional[int] = None,
        backoff_seconds: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """Execute `func` with retries and state tracking."""

        attempts = (retries if retries is not None else self._max_retries) + 1
        last_err: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            if not self.is_enabled(api):
                raise RuntimeError(f"{api} temporarily disabled due to repeated failures.")
            try:
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                await self._record_success(api)
                return result
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                await self._record_failure(api, f"{exc.__class__.__name__}: {exc}")
                self._logger.info("%s attempt %d failed: %s", api, attempt, exc)
                if attempt < attempts:
                    await asyncio.sleep(backoff_seconds * (2 ** (attempt - 1)))

        return None

    def summarize(self) -> str:
        """Return a human-readable error summary."""

        parts: list[str] = []
        for api, state in self._failures.items():
            status = "DISABLED" if not self.is_enabled(api) else "DEGRADED" if state.count else "OK"
            parts.append(f"{api}: {status} (failures={state.count})")
        return "; ".join(parts) if parts else "All systems nominal."
