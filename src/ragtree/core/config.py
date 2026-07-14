# ragtree/core/config.py
"""Configuration loading for RAGTree.

Behavior preserved from the research codebase: values under ``paths`` are
resolved to absolute paths. The resolution base is the directory that
contains the ``configs/`` folder when the file lives in one (this matches
the historical repo-root behavior), otherwise the config file's own folder.

Lookup order when no path is given:

1. the ``RAGTREE_CONFIG`` environment variable;
2. ``configs/default.yaml`` searched from the current directory upward.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError

__all__ = ["load_config", "DEFAULT_CONFIG_RELPATH"]

DEFAULT_CONFIG_RELPATH = Path("configs") / "default.yaml"


def _find_default_config(start: Path) -> Path | None:
    current = start.resolve()
    for candidate_root in (current, *current.parents):
        candidate = candidate_root / DEFAULT_CONFIG_RELPATH
        if candidate.is_file():
            return candidate
    return None


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load a YAML config and resolve its ``paths`` section to absolute paths."""
    if path is not None:
        cfg_path = Path(path)
        if not cfg_path.is_file():
            raise ConfigurationError(f"Config file not found: {cfg_path}")
    else:
        env_path = os.getenv("RAGTREE_CONFIG")
        if env_path:
            cfg_path = Path(env_path)
            if not cfg_path.is_file():
                raise ConfigurationError(
                    f"RAGTREE_CONFIG points to a missing file: {cfg_path}"
                )
        else:
            found = _find_default_config(Path.cwd())
            if found is None:
                raise ConfigurationError(
                    "No config path given and no configs/default.yaml found from "
                    "the current directory upward. Pass an explicit path or set "
                    "the RAGTREE_CONFIG environment variable."
                )
            cfg_path = found

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ConfigurationError(f"Top level of {cfg_path} must be a mapping.")

    cfg_path = cfg_path.resolve()
    root = cfg_path.parent.parent if cfg_path.parent.name == "configs" else cfg_path.parent
    paths = cfg.get("paths") or {}
    for key, value in paths.items():
        paths[key] = str((root / value).resolve())

    return cfg
