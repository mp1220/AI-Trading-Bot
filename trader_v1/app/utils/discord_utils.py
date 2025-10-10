"""Discord helpers for messaging and attachments."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import requests

from .logging import get_logger

ALERT_LEVELS = {
    "debug": 10,
    "info": 20,
    "warn": 30,
    "warning": 30,
    "error": 40,
    "critical": 50,
}


def _normalize_level(level: Optional[str]) -> str:
    if not level:
        return "info"
    return str(level).strip().lower()


def _should_send(event_level: str, minimum: str) -> bool:
    event_score = ALERT_LEVELS.get(event_level, ALERT_LEVELS["info"])
    min_score = ALERT_LEVELS.get(minimum, ALERT_LEVELS["info"])
    return event_score >= min_score


def send_discord_message(
    content: str,
    file_path: str | Path | None = None,
) -> bool:
    webhook = os.getenv("DISCORD_WEBHOOK")
    logger = get_logger("Discord")
    if not webhook:
        logger.debug("Discord webhook missing; message dropped.")
        return False

    payload = {"content": content}

    try:
        if file_path:
            path = Path(file_path)
            if not path.exists():
                logger.warning("⚠️  Discord attachment missing: %s", path)
                return False
            with path.open("rb") as fh:
                files = {"file": (path.name, fh)}
                data = {"payload_json": json.dumps(payload)}
                resp = requests.post(webhook, data=data, files=files, timeout=15)
        else:
            headers = {"Content-Type": "application/json"}
            resp = requests.post(webhook, headers=headers, data=json.dumps(payload), timeout=15)

        if resp.status_code >= 400:
            logger.warning("⚠️  Discord webhook failed: %s %s", resp.status_code, resp.text)
            return False
        return True
    except requests.RequestException as exc:  # noqa: BLE001
        logger.warning("⚠️  Discord notification error: %s", exc)
        return False


def notify_discord(
    event: str,
    content: str,
    level: str = "info",
    file_path: str | Path | None = None,
    minimum_level: Optional[str] = None,
) -> bool:
    logger = get_logger("Discord")
    event_level = _normalize_level(level)
    configured = _normalize_level(minimum_level or os.getenv("DISCORD_ALERT_LEVEL"))

    if not _should_send(event_level, configured):
        logger.debug(
            "Skipping Discord event '%s' at level '%s' (threshold '%s')",
            event,
            event_level,
            configured,
        )
        return False

    prefix = f"[{event.upper()}] " if event else ""
    return send_discord_message(f"{prefix}{content}", file_path=file_path)
