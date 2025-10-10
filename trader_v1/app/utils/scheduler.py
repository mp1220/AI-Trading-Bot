"""Asynchronous scheduler for orchestrating independent layers."""
from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from zoneinfo import ZoneInfo

from .logging import get_logger

JobCallable = Callable[[], Awaitable[None]] | Callable[[], None]


@dataclass
class ScheduledJob:
    name: str
    interval: dt.timedelta
    callback: JobCallable
    window: str = "all"
    immediate: bool = True
    last_run: Optional[dt.datetime] = None
    task: Optional[asyncio.Task] = field(default=None, compare=False)


class AsyncScheduler:
    """Lightweight asyncio scheduler with market-aware windows."""

    def __init__(
        self,
        tz: ZoneInfo,
        windows: Dict[str, tuple[dt.time, dt.time]],
        heartbeat_seconds: int = 60,
    ) -> None:
        self._logger = get_logger("Scheduler")
        self._jobs: List[ScheduledJob] = []
        self._tz = tz
        self._windows = windows
        self._heartbeat_seconds = heartbeat_seconds
        self._running = False

    def add_job(
        self,
        name: str,
        interval_seconds: int,
        callback: JobCallable,
        window: str = "all",
        immediate: bool = True,
    ) -> None:
        job = ScheduledJob(
            name=name,
            interval=dt.timedelta(seconds=interval_seconds),
            callback=callback,
            window=window,
            immediate=immediate,
        )
        self._jobs.append(job)
        self._logger.info("📆 Registered job %s (%s)", name, window)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for job in self._jobs:
            job.task = asyncio.create_task(self._run_job(job))
        asyncio.create_task(self._heartbeat())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for job in self._jobs:
            if job.task:
                job.task.cancel()

    async def _run_job(self, job: ScheduledJob) -> None:
        if job.immediate:
            await self._execute(job)
        while self._running:
            await asyncio.sleep(job.interval.total_seconds())
            if await self._should_run(job):
                await self._execute(job)

    async def _execute(self, job: ScheduledJob) -> None:
        try:
            result = job.callback()
            if asyncio.iscoroutine(result):
                await result
            job.last_run = dt.datetime.now(tz=self._tz)
            self._logger.info("✅ Job %s completed", job.name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._logger.error("⚠️  Job %s failed: %s", job.name, exc, exc_info=False)

    async def _should_run(self, job: ScheduledJob) -> bool:
        now = dt.datetime.now(tz=self._tz)
        window = job.window.lower()
        return self._window_active(window, now)

    def _window_active(self, window: str, now: dt.datetime) -> bool:
        if window == "all":
            return True
        if window not in self._windows:
            return True

        start, end = self._windows[window]
        now_time = now.timetz()

        if start <= end:
            return start <= now_time <= end
        # Overnight window (e.g. after-hours)
        return now_time >= start or now_time <= end

    async def _heartbeat(self) -> None:
        while self._running:
            await asyncio.sleep(self._heartbeat_seconds)
            snapshot = [
                f"{job.name}:{'✅' if job.last_run else '⏳'}"
                for job in self._jobs
            ]
            self._logger.info("💓 Heartbeat %s", " | ".join(snapshot))

