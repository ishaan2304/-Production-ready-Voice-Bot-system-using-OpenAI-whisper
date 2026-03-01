"""
Configuration loader for VoiceBot system.
Loads and validates settings from YAML config files.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    full_path = BASE_DIR / path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


class Config:
    """Centralized configuration manager."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._data = load_yaml(config_path)
        self._response_templates = None
        logger.info(f"Configuration loaded from {config_path}")

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a nested config value using dot-path keys."""
        data = self._data
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key)
            else:
                return default
            if data is None:
                return default
        return data

    @property
    def app(self) -> Dict:
        return self._data.get("app", {})

    @property
    def asr(self) -> Dict:
        return self._data.get("asr", {})

    @property
    def intent(self) -> Dict:
        return self._data.get("intent", {})

    @property
    def response(self) -> Dict:
        return self._data.get("response", {})

    @property
    def tts(self) -> Dict:
        return self._data.get("tts", {})

    @property
    def training(self) -> Dict:
        return self._data.get("training", {})

    @property
    def evaluation(self) -> Dict:
        return self._data.get("evaluation", {})

    @property
    def response_templates(self) -> Dict:
        if self._response_templates is None:
            templates_path = self.response.get("templates_path", "config/response_templates.yaml")
            self._response_templates = load_yaml(templates_path)
        return self._response_templates

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a path relative to the project base directory."""
        return BASE_DIR / relative_path


# Singleton config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config
