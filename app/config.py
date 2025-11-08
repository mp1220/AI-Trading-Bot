"""Configuration utilities for Trader_Model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import re
import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def normalize_alpaca_env() -> None:
    """Ensure canonical ALPACA_* environment variables are populated."""

    key_id = os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    if key_id:
        os.environ["ALPACA_API_KEY_ID"] = key_id
        os.environ.setdefault("ALPACA_API_KEY", key_id)

    secret_key = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if secret_key:
        os.environ["ALPACA_API_SECRET_KEY"] = secret_key
        os.environ.setdefault("ALPACA_SECRET_KEY", secret_key)

    base_url = os.getenv("ALPACA_API_BASE_URL") or os.getenv("ALPACA_BASE_URL")
    if base_url:
        os.environ["ALPACA_API_BASE_URL"] = base_url
        os.environ.setdefault("ALPACA_BASE_URL", base_url)


normalize_alpaca_env()


class ConfigError(Exception):
    """Raised when the configuration file cannot be loaded or validated."""


_DURATION_PATTERN = re.compile(r"^(?P<value>\d+)(?P<unit>[smhd])$")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError("Root of config.yaml must be a mapping/object.")
    return data


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    cfg_path = Path(path) if path else root / "config.yaml"
    return _load_yaml(cfg_path)


def get_alpaca_settings(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    alpaca_cfg = cfg.get("alpaca") or {}
    mode = alpaca_cfg.get("mode", "paper").lower()
    base_url = (
        os.getenv("ALPACA_API_BASE_URL")
        or os.getenv("ALPACA_BASE_URL")
        or (
        alpaca_cfg.get("base_url_live")
        if mode == "live"
        else alpaca_cfg.get("base_url_paper")
        )
    )
    return {
        "mode": mode,
        "feed": alpaca_cfg.get("data_feed", "IEX"),
        "base_url": base_url,
        "trade_enabled": bool(alpaca_cfg.get("trade_enabled", False)),
        "focus": focus_tickers(cfg),
    }


def parse_duration(value: str | int | float) -> int:
    """Convert config duration strings (e.g. ``15m``) to seconds."""

    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        raise ConfigError(f"Unsupported duration type: {value!r}")
    match = _DURATION_PATTERN.match(value.strip().lower())
    if not match:
        raise ConfigError(f"Invalid duration expression: {value}")
    magnitude = int(match.group("value"))
    unit = match.group("unit")
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return magnitude * multipliers[unit]


def data_source_enabled(cfg: Dict[str, Any], source: str) -> bool:
    return bool(cfg.get("data_sources", {}).get(source, False))


def runtime_setting(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    return cfg.get("runtime", {}).get(key, default)


def scheduler_interval(cfg: Dict[str, Any], key: str, default: str | int) -> int:
    value = cfg.get("scheduler", {}).get(key, default)
    return parse_duration(value)


def database_dsn(cfg: Dict[str, Any]) -> Optional[str]:
    dsn_env = cfg.get("database", {}).get("dsn_env")
    if not dsn_env:
        return None
    return os.getenv(dsn_env)


@dataclass(frozen=True)
class ProviderCredentials:
    name: str
    enabled: bool
    api_key_env: Optional[str]

    @property
    def api_key(self) -> Optional[str]:
        if not self.enabled or not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)


def get_provider(cfg: Dict[str, Any], name: str) -> ProviderCredentials:
    provider = (cfg.get("providers") or {}).get(name, {})
    enabled = bool(provider.get("enabled", False))
    api_key_env = provider.get("api_key_env")
    return ProviderCredentials(name=name, enabled=enabled, api_key_env=api_key_env)


@dataclass(frozen=True)
class DiscordConfig:
    enabled: bool
    send_attachments: bool
    report_path: Path
    alert_level: str


def discord_config(cfg: Dict[str, Any]) -> DiscordConfig:
    runtime_enabled = bool(runtime_setting(cfg, "discord_reports", True))
    discord_cfg = cfg.get("discord", {})
    send_attachments = bool(discord_cfg.get("send_attachments", False))
    report_path = Path(discord_cfg.get("report_path", "reports"))
    env_var = runtime_setting(cfg, "discord_alert_level_env", "DISCORD_ALERT_LEVEL")
    alert_level = os.getenv(env_var) or discord_cfg.get("alert_level") or "info"
    return DiscordConfig(runtime_enabled, send_attachments, report_path, alert_level.lower())


def focus_tickers(cfg: Dict[str, Any]) -> List[str]:
    env_var = runtime_setting(cfg, "focus_tickers_env", "FOCUS_TICKERS")
    raw_env = os.getenv(env_var, "")
    if raw_env:
        focus = [sym.strip().upper() for sym in raw_env.split(",") if sym.strip()]
    else:
        focus = cfg.get("alpaca", {}).get("focus_universe") or []
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: List[str] = []
    for ticker in focus:
        ticker_up = ticker.upper()
        if ticker_up and ticker_up not in seen:
            ordered.append(ticker_up)
            seen.add(ticker_up)
    return ordered
