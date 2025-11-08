"""Cooldown manager for throttling API requests."""
from __future__ import annotations

import time
from typing import Dict

_last_calls: Dict[str, float] = {}


def enforce(service: str, seconds: int) -> None:
    now = time.time()
    last = _last_calls.get(service, 0.0)
    if now - last < seconds:
        time.sleep(seconds - (now - last))
    _last_calls[service] = time.time()
