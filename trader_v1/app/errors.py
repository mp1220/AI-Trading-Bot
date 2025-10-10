"""Lightweight error manager placeholder with async guard."""

import asyncio
import traceback
import logging

logger = logging.getLogger("ErrorManager")


class ErrorManager:
    """Minimal error manager that supports async guard execution."""

    def __init__(self):
        self.errors = []

    async def guard(self, name: str, coro_or_func, *args, **kwargs):
        """Safely execute an async or sync function, capturing exceptions."""
        try:
            if asyncio.iscoroutinefunction(coro_or_func):
                return await coro_or_func(*args, **kwargs)
            else:
                # Support sync fallback
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, coro_or_func, *args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            msg = f"{name} failed: {e}\n{tb}"
            logger.error(msg)
            self.errors.append(msg)
            raise

    def capture(self, message: str):
        """Store a non-fatal error message."""
        self.errors.append(message)
        logger.warning(f"Captured error: {message}")

    def latest(self):
        """Return latest error."""
        return self.errors[-1] if self.errors else None

    def clear(self):
        """Clear all stored errors."""
        self.errors.clear()
