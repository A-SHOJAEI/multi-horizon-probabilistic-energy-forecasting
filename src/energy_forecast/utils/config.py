"""Configuration utilities."""

import yaml
from pathlib import Path
from typing import Optional


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    default_path = Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"
    path = Path(config_path) if config_path else default_path

    with open(path) as f:
        config = yaml.safe_load(f)
    return config
