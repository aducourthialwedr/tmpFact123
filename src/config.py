"""Lecture des fichiers de configuration YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA_PATH = REPO_ROOT / "config" / "schema.yaml"
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    return content or {}


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    """Chemin absolu : relatif à `base` (racine du dépôt par défaut)."""
    p = Path(path)
    return p if p.is_absolute() else base / p
