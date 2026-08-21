"""Configuration loader."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var = value[2:-1]
        return os.environ.get(var, "")
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load YAML config.  Relative paths are resolved against the repo root."""
    if path is None:
        path = ROOT / "configs" / "default.yaml"
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve(cfg)
    return cfg


def repo_root() -> Path:
    return ROOT
