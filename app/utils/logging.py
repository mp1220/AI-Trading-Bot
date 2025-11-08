"""Logging utilities for Trader_Model."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColorFormatter(logging.Formatter):
    """Simple ANSI color formatter for terminal readability."""

    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        color = self.COLORS.get(record.levelno)
        if color and os.isatty(1):
            return f"{color}{base}{self.RESET}"
        return base


def init_logging(log_dir: Path, level: str = "INFO", quiet: bool = False) -> None:
    """Initialise root logging handlers.

    Parameters
    ----------
    log_dir: Path
        Directory where log files reside.
    level: str
        Desired root logging level name.
    quiet: bool
        When true, suppress console logs below WARNING.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "trader_model.log"

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(numeric_level)
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and quiet:
                handler.setLevel(logging.WARNING)
        return

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING if quiet else numeric_level)
    stream_handler.setFormatter(ColorFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    logging.basicConfig(level=numeric_level, handlers=[stream_handler, file_handler])


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module logger."""

    return logging.getLogger(name or "TraderModel")
