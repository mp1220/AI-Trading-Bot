"""Discord helpers for messaging and attachments."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import requests
from .logging import get_logger

# ────────────────────────────────────────────────
# ⚙️ CONFIG CONSTANTS
# ────────────────────────────────────────────────
ALERT_LEVELS = {
    "debug": 10,
    "info": 20,
    "warn": 30,
    "warning": 30,
    "error": 40,
    "critical": 50,
}

STATUS_COLORS = {
    "ok": 0x2ECC71,
    "success": 0x2ECC71,
    "info": 0x3498DB,
    "warn": 0xF1C40F,
    "warning": 0xF1C40F,
    "error": 0xE74C3C,
    "critical": 0xE74C3C,
    "startup": 0x7289DA,
    "trade": 0x1ABC9C,
}

DEFAULT_WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"
MAX_EMBED_CHARS = 2000
MAX_EMBED_CHUNKS = 3
CONT_SUFFIX = " (cont.)"


# ────────────────────────────────────────────────
# 🔧 LEVEL + STRING HELPERS
# ────────────────────────────────────────────────
def _normalize_level(level: Optional[str]) -> str:
    return str(level or "info").strip().lower()


def _should_send(event_level: str, minimum: str) -> bool:
    event_score = ALERT_LEVELS.get(event_level, ALERT_LEVELS["info"])
    min_score = ALERT_LEVELS.get(minimum, ALERT_LEVELS["info"])
    return event_score >= min_score


def _sanitize_field_value(value: str) -> str:
    return value if len(value) <= MAX_EMBED_CHARS else value[: MAX_EMBED_CHARS - 3] + "..."


# ────────────────────────────────────────────────
# 🔍 DYNAMIC WEBHOOK RESOLUTION
# ────────────────────────────────────────────────
def _resolve_webhook(channel: Optional[str], webhook_url: Optional[str]) -> Optional[str]:
    """
    Dynamically resolve Discord webhook based on channel name or explicit URL.
    This allows real-time .env reloads and clean multi-channel routing.
    """
    if webhook_url:
        return webhook_url

    # Dynamically build environment mapping
    env_map = {
        "market-updates": os.getenv("DISCORD_WEBHOOK_UPDATES"),
        "backend": os.getenv("DISCORD_WEBHOOK_BACKEND"),
        "trades": os.getenv("DISCORD_WEBHOOK_TRADES"),
        # add any custom naming if you changed .env vars
        "general": os.getenv("DISCORD_WEBHOOK_GENERAL"),
        "alerts": os.getenv("DISCORD_WEBHOOK_ALERTS"),
    }

    # Direct channel match
    if channel:
        resolved = env_map.get(channel)
        if resolved:
            return resolved

    # Fallbacks
    fallback = os.getenv(DEFAULT_WEBHOOK_ENV)
    if fallback:
        return fallback

    # Safety: default to backend webhook if set
    backend = os.getenv("DISCORD_WEBHOOK_BACKEND")
    if backend:
        return backend

    return None


# ────────────────────────────────────────────────
# 🧩 EMBED BUILDING
# ────────────────────────────────────────────────
def _chunk_description(text: Optional[str]) -> list[str]:
    if not text:
        return [""]
    chunks: list[str] = []
    remaining = text
    for index in range(MAX_EMBED_CHUNKS):
        if len(remaining) <= MAX_EMBED_CHARS:
            chunks.append(remaining)
            break
        if index == MAX_EMBED_CHUNKS - 1:
            chunks.append(remaining[: MAX_EMBED_CHARS - 3] + "...")
            break
        take = MAX_EMBED_CHARS - len(CONT_SUFFIX)
        chunks.append(remaining[:take] + CONT_SUFFIX)
        remaining = remaining[take:]
    return chunks


# ────────────────────────────────────────────────
# 💬 MAIN EMBED FUNCTION
# ────────────────────────────────────────────────
def discord_embed(
    title: str,
    description: Optional[str] = None,
    *,
    status: Optional[str] = None,
    color: Optional[int] = None,
    fields: Optional[Iterable[Sequence[str]]] = None,
    mention_role_id: Optional[str] = None,
    channel: Optional[str] = None,
    webhook_url: Optional[str] = None,
    username: Optional[str] = None,
    avatar_url: Optional[str] = None,
) -> bool:
    """
    Send a Discord embed message with smart routing and rate-limit handling.
    Supports multi-part splitting, color-coded statuses, and role mentions.
    """

    logger = get_logger("Discord")
    target_webhook = _resolve_webhook(channel, webhook_url)

    if not target_webhook:
        logger.error("Discord embed aborted: No webhook configured (channel='%s', env missing).", channel)
        return False

    resolved_color = color or STATUS_COLORS.get((status or "info").lower(), STATUS_COLORS["info"])
    description_chunks = _chunk_description(description)
    content = f"<@&{mention_role_id}>" if mention_role_id else None
    if content and len(content) > 1900:
        content = content[:1900] + "...[truncated]"

    # Prepare field blocks
    sanitized_fields = []
    if fields:
        for field in fields:
            if isinstance(field, Sequence) and len(field) >= 2:
                sanitized_fields.append(
                    {
                        "name": str(field[0])[:256],
                        "value": _sanitize_field_value(str(field[1])),
                        "inline": True,
                    }
                )

    def _post(payload: dict) -> bool:
        """Post JSON payload to Discord with retry + gentle pacing."""
        try:
            resp = requests.post(target_webhook, json=payload, timeout=10)
        except requests.RequestException as exc:
            logger.warning("Discord network error: %s", exc)
            return False

        # Handle rate-limiting
        if resp.status_code == 429:
            try:
                retry_after = float(resp.json().get("retry_after", 1.0))
            except Exception:
                retry_after = 1.0
            logger.warning("Discord rate-limited. Retrying after %.2fs.", retry_after)
            time.sleep(retry_after + 0.2)
            return _post(payload)

        if resp.status_code >= 400:
            logger.warning("Discord webhook error %s: %s", resp.status_code, resp.text[:200])
            return False

        # Prevent burst spam
        time.sleep(0.25)
        return True

    # Iterate through chunks to obey 2000-char limit
    for idx, chunk in enumerate(description_chunks, start=1):
        embed_title = title if idx == 1 else f"{title} (cont. {idx})"
        embed = {"title": embed_title, "description": chunk, "color": resolved_color}

        if sanitized_fields and idx == 1:
            embed["fields"] = sanitized_fields

        payload = {"embeds": [embed]}
        if content and idx == 1:
            payload["content"] = content
        if username:
            payload["username"] = username
        if avatar_url:
            payload["avatar_url"] = avatar_url

        if not _post(payload):
            return False

    return True


# ────────────────────────────────────────────────
# 📎 SIMPLE MESSAGE + FILE SENDERS
# ────────────────────────────────────────────────
def send_discord_message(
    content: str,
    file_path: str | Path | None = None,
) -> bool:
    """Send a plain text message (and optional file) to the default webhook."""
    logger = get_logger("Discord")
    webhook = os.getenv("DISCORD_WEBHOOK") or os.getenv(DEFAULT_WEBHOOK_ENV) or os.getenv("DISCORD_WEBHOOK_BACKEND")
    if not webhook:
        logger.debug("Discord webhook missing; message dropped.")
        return False

    payload = {"content": content}
    if payload["content"] and len(payload["content"]) > 1900:
        payload["content"] = payload["content"][:1900] + "...[truncated]"
    try:
        if file_path:
            path = Path(file_path)
            if not path.exists():
                logger.warning("⚠️ Discord attachment missing: %s", path)
                return False
            with path.open("rb") as fh:
                files = {"file": (path.name, fh)}
                data = {"payload_json": json.dumps(payload)}
                resp = requests.post(webhook, data=data, files=files, timeout=15)
        else:
            resp = requests.post(webhook, json=payload, timeout=10)

        if resp.status_code >= 400:
            logger.warning("⚠️ Discord webhook failed: %s %s", resp.status_code, resp.text)
            return False
        return True
    except requests.RequestException as exc:
        logger.warning("⚠️ Discord send error: %s", exc)
        return False


# ────────────────────────────────────────────────
# 🔔 FILTERED ALERT SENDER
# ────────────────────────────────────────────────
def notify_discord(
    event: str,
    content: str,
    level: str = "info",
    file_path: str | Path | None = None,
    minimum_level: Optional[str] = None,
) -> bool:
    """Simplified filtered notifier respecting DISCORD_ALERT_LEVEL."""
    logger = get_logger("Discord")
    event_level = _normalize_level(level)
    configured = _normalize_level(minimum_level or os.getenv("DISCORD_ALERT_LEVEL"))

    if not _should_send(event_level, configured):
        logger.debug("Skipping Discord event '%s' at level '%s' (threshold '%s')", event, event_level, configured)
        return False

    prefix = f"[{event.upper()}] " if event else ""
    return send_discord_message(f"{prefix}{content}", file_path=file_path)
