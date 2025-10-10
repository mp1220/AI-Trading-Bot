import os
from dotenv import load_dotenv
import yaml

from app.config import normalize_alpaca_env

# Force dotenv to load from project root
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
normalize_alpaca_env()

def load_config(path: str = "config.yaml") -> dict:
    load_dotenv()
    normalize_alpaca_env()
    with open(path, "r") as f:
        return yaml.safe_load(f)
