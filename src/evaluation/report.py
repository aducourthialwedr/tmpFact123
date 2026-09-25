"""Rendu markdown d'une évaluation (rapport reproductible, brief §8)."""

from __future__ import annotations

import math

import pandas as pd


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.2%}" if value <= 1 else f"{value:.2f}"
    return str(value)


def render_markdown(summary: dict, tables: dict[str, pd.DataFrame], context: dict) -> str:
    lines = [f"# Évaluation — rapprocheur « {context['matcher']} », période {context['period']}", "",
             f"Du {context['start']} au {context['end']} · journal `{context['journal_sha256'][:16]}…`", "",
             "| indicateur | valeur |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in summary.items()]
    for name in ("cascade", "by_group", "by_step", "by_rule", "by_rule_alone", "by_client_file", "ml_calibration",
                 "by_month"):
        if name not in tables:
            continue
        df = tables[name]
        lines += ["", f"## {name}", ""]
        if df.empty:
            lines.append("_(vide)_")
            continue
        lines += ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"
