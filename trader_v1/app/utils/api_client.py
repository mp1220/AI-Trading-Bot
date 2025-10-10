"""HTTP client helpers with retry, diagnostics, and logging."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests

from .logging import get_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTICS_PATH = PROJECT_ROOT / "logs" / "api_diagnostics.csv"


def _ensure_diagnostics_header() -> None:
    if DIAGNOSTICS_PATH.exists():
        return
    DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DIAGNOSTICS_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "service", "success", "latency", "error_code"])


def _record(service: str, success: bool, latency: float, error_code: Optional[int]) -> None:
    _ensure_diagnostics_header()
    with DIAGNOSTICS_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            service,
            int(success),
            f"{latency:.3f}",
            "" if error_code is None else error_code,
        ])


def request_json(
    service: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    method: str = "GET",
    timeout: int = 15,
    backoff_schedule: Iterable[int] | None = None,
    raise_for_status: bool = True,
) -> Any:
    logger = get_logger(f"API.{service}")
    schedule = list(backoff_schedule or [0, 2, 4, 8, 16])

    for attempt, wait_time in enumerate(schedule, start=1):
        if wait_time:
            logger.info("[%s] Rate limit reached, retrying in %ss...", service, wait_time)
            time.sleep(wait_time)

        start = time.perf_counter()
        error_code: Optional[int] = None
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            latency = time.perf_counter() - start
            error_code = response.status_code if not response.ok else None
            if response.ok:
                _record(service, True, latency, None)
                if not response.content:
                    return {}
                if "application/json" in response.headers.get("Content-Type", ""):
                    return response.json()
                return json.loads(response.text)

            if response.status_code in {429, 500, 502, 503, 504} and attempt < len(schedule):
                _record(service, False, latency, response.status_code)
                continue
            _record(service, False, latency, response.status_code)
            return {}
        except requests.RequestException as exc:
            latency = time.perf_counter() - start
            _record(service, False, latency, error_code)
            logger.info("[%s] Request error: %s", service, exc)

    return {}
