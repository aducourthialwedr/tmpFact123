"""Profil des données chargées : volumes, plages de dates, champs manquants,
anomalies et résumé du journal (critère « fini quand » de l'étape 1)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.load.canonical import TABLES, FieldType
from src.load.loader import Issue, LoadedData


def build_profile(data: LoadedData, journal: pd.DataFrame, issues: list[Issue]) -> dict[str, pd.DataFrame]:
    volumes, missing, dates = [], [], []
    for name, df in data.tables.items():
        volumes.append({"table": name, "rows": len(df)})
        mapped = set(data.mapped_fields[name])
        for f in TABLES[name].fields:
            if f.name not in df.columns:
                continue
            n_null = int(df[f.name].isna().sum())
            missing.append({
                "table": name,
                "field": f.name,
                "required": f.required,
                "mapped": f.name in mapped,
                "rows": len(df),
                "null": n_null,
                "null_pct": round(100 * n_null / len(df), 2) if len(df) else 0.0,
            })
            if f.type in (FieldType.DATE, FieldType.TIMESTAMP) and f.name in mapped:
                col = df[f.name]
                dates.append({"table": name, "field": f.name, "min": col.min(), "max": col.max()})

    events = (
        journal.groupby("event_type", sort=False)
        .agg(count=("seq", "size"), first_ts=("ts", "min"), last_ts=("ts", "max"))
        .reset_index()
    )
    issue_df = pd.DataFrame([asdict(i) for i in issues],
                            columns=["table", "field", "kind", "count", "detail"])
    return {
        "volumes": pd.DataFrame(volumes),
        "missing_fields": pd.DataFrame(missing),
        "date_ranges": pd.DataFrame(dates),
        "issues": issue_df,
        "events": events,
    }


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(vide)_\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = ["" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_profile_markdown(profile: dict[str, pd.DataFrame], meta: dict[str, Any]) -> str:
    missing = profile["missing_fields"]
    notable = missing[(~missing["mapped"]) | (missing["null"] > 0)]
    parts = [
        "# Profil de chargement — étape 1\n",
        "\n".join(f"- **{k}** : `{v}`" for k, v in meta.items()) + "\n",
        "## Volumes\n", _md_table(profile["volumes"]),
        "## Plages de dates\n", _md_table(profile["date_ranges"]),
        "## Champs non mappés ou incomplets\n", _md_table(notable),
        "## Anomalies\n", _md_table(profile["issues"]),
        "## Journal d'événements\n", _md_table(profile["events"]),
    ]
    return "\n".join(parts)


def write_reports(profile: dict[str, pd.DataFrame], meta: dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name, df in profile.items():
        df.to_csv(reports_dir / f"load_{name}.csv", index=False)
    md_path = reports_dir / "load_profile.md"
    md_path.write_text(render_profile_markdown(profile, meta), encoding="utf-8")
    return md_path
