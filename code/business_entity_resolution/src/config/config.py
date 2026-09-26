"""
config.py -- Single YAML loader for all pipeline parameters.

All src/ modules import from here. No hardcoded paths, thresholds,
or hyperparameters anywhere in pipeline code.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Any

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML is required: pip install pyyaml"
    ) from e

_DEFAULT_CONFIG = pathlib.Path(__file__).parent / "pipeline.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | pathlib.Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and cache the pipeline YAML config.

    Assumption: config is read-only after first load; no hot-reload.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get(key: str, default: Any = None) -> Any:
    """Convenience accessor for top-level keys."""
    return load_config().get(key, default)
