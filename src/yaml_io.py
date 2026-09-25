"""Écriture YAML commentée à partir de modèles pydantic.

Les fichiers de config sont édités à la fois à la main et par l'UI : on les
régénère avec les descriptions des champs en commentaires, pour que l'écriture
par l'UI ne perde pas la documentation.
"""

from __future__ import annotations

import textwrap
from datetime import date
from typing import Any

import yaml
from pydantic import BaseModel


def scalar(value: Any) -> str:
    """Représentation YAML d'une valeur scalaire, liste ou dict courts (style flow)."""
    if isinstance(value, date):
        return value.isoformat()
    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, sort_keys=False, width=10_000)
    return text.removesuffix("\n...\n").strip()


def _comment(text: str | None, pad: str) -> list[str]:
    if not text:
        return []
    return [f"{pad}# {line}" for line in textwrap.wrap(text, width=96 - len(pad))]


def model_lines(model: BaseModel, indent: int = 0, comments: bool = True) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    for name, info in type(model).model_fields.items():
        value = getattr(model, name)
        if comments:
            lines += _comment(info.description, pad)
        if isinstance(value, BaseModel):
            lines.append(f"{pad}{name}:")
            lines += model_lines(value, indent + 2, comments)
        elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
            lines.append(f"{pad}{name}:")
            for item in value:
                item_lines = model_lines(item, indent + 4, comments=False)
                first = item_lines[0].lstrip()
                lines.append(f"{pad}  - {first}")
                lines += item_lines[1:]
        elif isinstance(value, dict) and value:
            lines.append(f"{pad}{name}:")
            lines += [f"{pad}  {k}: {scalar(v)}" for k, v in value.items()]
        else:
            lines.append(f"{pad}{name}: {scalar(value)}")
        if comments and indent == 0:
            lines.append("")
    return lines


def dump_model(model: BaseModel, header: str = "") -> str:
    head = [f"# {line}" if line else "#" for line in header.strip().splitlines()] if header else []
    body = model_lines(model)
    return "\n".join(head + ([""] if head else []) + body).rstrip() + "\n"
