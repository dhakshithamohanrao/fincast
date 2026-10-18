from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Return the project root folder (the one that contains config/ and src/)."""
    return Path(__file__).resolve().parents[2]


def load_config(path=None) -> dict:
    """Load settings from config/config.yaml and return them as a dict."""
    config_path = Path(path) if path else get_project_root() / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)