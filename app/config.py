"""
Central config loader: loads .env and configs/settings.yaml, returns a single settings dict.
Prevents hardcoding settings across ingestion, retrieval, and app.
"""

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

_settings_cache: dict[str, Any] | None = None


def load_env() -> None:
    """Load .env into os.environ if python-dotenv is available and .env exists."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass


def load_settings_yaml() -> dict[str, Any]:
    """Load configs/settings.yaml. Returns empty dict on missing or error."""
    path = ROOT / "configs" / "settings.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get_settings(reload: bool = False) -> dict[str, Any]:
    """
    Return merged settings: .env loaded into os.environ, then settings.yaml as dict.
    Cached; set reload=True to force re-read.
    """
    global _settings_cache
    if _settings_cache is not None and not reload:
        return _settings_cache
    load_env()
    _settings_cache = load_settings_yaml()
    return _settings_cache
