import os
import yaml


def load_config():
    path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
SP100 = sum([s.replace(" ", "").split(";") for s in CONFIG["universe"]["tickers"]], [])
FOCUS_TICKERS = CONFIG["focus_universe"]["tickers"]
SCHED = CONFIG["scheduling"]
RISK = CONFIG["risk"]
NOTIFY = CONFIG["notifications"]
THRESHOLDS = CONFIG["thresholds"]
ALP = CONFIG.get("alpaca", {})
