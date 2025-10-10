"""Asynchronous job scheduler for orchestrating subsystem cadences."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Iterable, Optional

JobCallable = Callable[[], Awaitable[None]] | Callable[[], Awaitable]

logger = logging.getLogger("Scheduler")


@dataclass
class ScheduledJob:
    name: str
    interval: float
    callback: JobCallable
    enabled: bool = True
    task: Optional[asyncio.Task] = field(default=None, compare=False)


class AsyncScheduler:
    """Lightweight asynchronous scheduler with pause/resume support."""

    def __init__(
        self,
        *,
        error_handler: Optional[Callable[[ScheduledJob, Exception], Awaitable[None] | None]] = None,
    ) -> None:
        self._jobs: Dict[str, ScheduledJob] = {}
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._error_handler = error_handler
        self._runner_tasks: list[asyncio.Task] = []
        self._lock = asyncio.Lock()

    def add_job(self, name: str, interval_seconds: float, callback: JobCallable, *, enabled: bool = True) -> None:
        if name in self._jobs:
            raise ValueError(f"Job '{name}' already registered.")
        self._jobs[name] = ScheduledJob(name=name, interval=interval_seconds, callback=callback, enabled=enabled)
        logger.info("📆 Registered job %s (interval=%ss)", name, interval_seconds)

    def remove_job(self, name: str) -> None:
        job = self._jobs.pop(name, None)
        if job and job.task:
            job.task.cancel()

    def jobs(self) -> Iterable[ScheduledJob]:
        return tuple(self._jobs.values())

    async def start_forever(self) -> None:
        if self._runner_tasks:
            raise RuntimeError("Scheduler already running.")
        self._stop_event.clear()
        for job in self._jobs.values():
            job.task = asyncio.create_task(self._job_loop(job), name=f"scheduler:{job.name}")
            self._runner_tasks.append(job.task)
        logger.info("✅ Scheduler started (%d jobs active)", len(self._jobs))
        try:
            await self._stop_event.wait()
        finally:
            await self._shutdown_tasks()
            logger.info("🛑 Scheduler stopped.")

    async def _job_loop(self, job: ScheduledJob) -> None:
        loop = asyncio.get_running_loop()
        next_run = loop.time()
        while not self._stop_event.is_set():
            await self._pause_event.wait()
            now = loop.time()
            delay = max(0.0, next_run - now)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                pass

            if not job.enabled:
                next_run = loop.time() + job.interval
                continue

            started = time.time()
            try:
                result = job.callback()
                if asyncio.iscoroutine(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("⚠️ Job %s failed: %s", job.name, exc)
                await self._handle_error(job, exc)
            finally:
                duration = time.time() - started
                logger.debug("⏱️ Job %s completed in %.2fs", job.name, duration)
                next_run = loop.time() + job.interval

    async def _handle_error(self, job: ScheduledJob, exc: Exception) -> None:
        if not self._error_handler:
            return
        try:
            result = self._error_handler(job, exc)
            if asyncio.iscoroutine(result):
                await result
        except Exception as callback_exc:  # noqa: BLE001
            logger.error("Scheduler error handler raised: %s", callback_exc, exc_info=True)

    async def pause_job(self, name: str) -> None:
        job = self._jobs.get(name)
        if job:
            job.enabled = False
            logger.info("⏸️ Job %s paused", name)

    async def resume_job(self, name: str) -> None:
        job = self._jobs.get(name)
        if job:
            job.enabled = True
            logger.info("▶️ Job %s resumed", name)

    def pause_all(self) -> None:
        self._pause_event.clear()
        logger.info("⏸️ Scheduler paused")

    def resume_all(self) -> None:
        if not self._pause_event.is_set():
            self._pause_event.set()
            logger.info("▶️ Scheduler resumed")

    async def stop(self) -> None:
        self._stop_event.set()
        await self._shutdown_tasks()

    async def _shutdown_tasks(self) -> None:
        if not self._runner_tasks:
            return
        for task in self._runner_tasks:
            task.cancel()
        for task in self._runner_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._runner_tasks.clear()
        for job in self._jobs.values():
            job.task = None
