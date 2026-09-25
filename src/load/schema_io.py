"""Lecture / écriture commentée de `config/schema.yaml` (mapping vers les noms réels).

Le fichier est régénéré à partir du modèle canonique : les commentaires
(champs requis, rôle de chaque clé) survivent à une sauvegarde depuis l'UI.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.config import DEFAULT_SCHEMA_PATH, read_yaml
from src.load.canonical import TABLES
from src.yaml_io import scalar

GLOBAL_KEYS: dict[str, str] = {
    "amount_unit": "Unité des montants dans les sources : \"cents\" (entiers de centimes) ou \"units\" (ex. 1234.56).",
    "decimal_separator": "Séparateur décimal des montants lus en texte (\".\" ou \",\").",
    "source_timezone": "Fuseau des horodatages sans fuseau explicite (ex. \"Europe/Paris\"). null → UTC.",
    "date_format": "Format strptime des dates (null → inférence pandas).",
    "timestamp_format": "Format strptime des horodatages (null → inférence pandas).",
    "base_dir": "Répertoire des fichiers sources (relatif à la racine du dépôt ou absolu).",
}

TABLE_NOTES: dict[str, str] = {
    "technical_account": "Optionnel — référentiel des IBAN de comptes techniques.",
    "client_file": "Optionnel — client files. FORMAT RÉEL À CONFIRMER (brief §3.2). Voie tabulaire uniquement.",
    "client_file_line": "Optionnel — lignes des client files.",
}

FIELD_NOTES: dict[tuple[str, str], str] = {
    ("payment", "booking_date"): "date de comptabilisation si elle existe (§3.5) — ordonne le journal",
    ("payment", "amount"): "signé",
    ("payment", "bankroll_code"): "souvent absent (§3.5)",
    ("invoice", "current_amount"): "audit uniquement, jamais utilisé par la pipeline",
    ("imputation", "status"): "FULL / PARTIAL après value_maps",
    ("imputation", "residual_amount"): "solde de la facture APRÈS la ligne",
    ("debtor", "closed_at"): "peut ne pas exister (§3.5)",
    ("agreement", "client_id"): "→ assignor.party_id",
    ("client_file", "received_at"): "pivot temporel",
}

VALUE_MAP_FIELDS: dict[str, list[str]] = {"imputation": ["status"]}

_HEADER = """\
# Mapping modèle canonique → noms réels des tables et colonnes.
# À REMPLIR PAR L'ÉQUIPE. Aucune valeur ne doit être devinée.
# Éditable à la main ou depuis l'UI (page Configuration).
#
# - Chaque `null` sous `columns` est une colonne réelle à renseigner.
#   Les champs marqués (requis) bloquent le chargement tant qu'ils sont vides.
#   Les autres peuvent rester à null : ils seront signalés dans le rapport.
# - `source` : fichier (csv ou parquet) relatif à `base_dir`.
# - `read_options` : options passées telles quelles à pandas.read_csv / read_parquet
#   (ex. sep: ";", encoding: "latin-1").
# - `value_maps` : traduction des valeurs réelles vers les valeurs canoniques,
#   ex. status: {FULL: [TOTAL, SOLDE], PARTIAL: PARTIEL}.
"""


def empty_schema() -> dict[str, Any]:
    cfg: dict[str, Any] = {k: None for k in GLOBAL_KEYS}
    cfg["tables"] = {}
    for name, table in TABLES.items():
        tcfg: dict[str, Any] = {"source": None, "read_options": {},
                                "columns": {f.name: None for f in table.fields}}
        if name in VALUE_MAP_FIELDS:
            tcfg["value_maps"] = {f: {} for f in VALUE_MAP_FIELDS[name]}
        cfg["tables"][name] = tcfg
    return cfg


def normalize_schema(cfg: dict[str, Any]) -> dict[str, Any]:
    """Complète un schéma partiel avec toutes les clés attendues (valeurs vides)."""
    out = empty_schema()
    for k in GLOBAL_KEYS:
        if k in cfg:
            out[k] = cfg[k]
    for name, tcfg in (cfg.get("tables") or {}).items():
        if name not in out["tables"] or not tcfg:
            continue
        target = out["tables"][name]
        target["source"] = tcfg.get("source")
        target["read_options"] = dict(tcfg.get("read_options") or {})
        for fld, col in (tcfg.get("columns") or {}).items():
            target["columns"][fld] = col
        for fld, mapping in (tcfg.get("value_maps") or {}).items():
            target.setdefault("value_maps", {})[fld] = copy.deepcopy(mapping) or {}
        for opt in ("date_format", "timestamp_format"):
            if tcfg.get(opt):
                target[opt] = tcfg[opt]
    return out


def dump_schema(cfg: dict[str, Any]) -> str:
    cfg = normalize_schema(cfg)
    lines = [_HEADER]
    for key, note in GLOBAL_KEYS.items():
        lines += [f"# {note}", f"{key}: {scalar(cfg[key])}"]
    lines += ["", "tables:"]
    for name, table in TABLES.items():
        tcfg = cfg["tables"][name]
        if name in TABLE_NOTES:
            lines.append(f"  # {TABLE_NOTES[name]}")
        lines += [f"  {name}:", f"    source: {scalar(tcfg['source'])}",
                  f"    read_options: {scalar(tcfg['read_options'])}"]
        for opt in ("date_format", "timestamp_format"):
            if tcfg.get(opt):
                lines.append(f"    {opt}: {scalar(tcfg[opt])}")
        lines.append("    columns:")
        for f in table.fields:
            notes = [n for n in ("(requis)" if f.required else None, FIELD_NOTES.get((name, f.name))) if n]
            comment = f"  # {' '.join(notes)}" if notes else ""
            lines.append(f"      {f.name}: {scalar(tcfg['columns'].get(f.name))}{comment}")
        if "value_maps" in tcfg:
            lines.append("    value_maps:")
            for fld, mapping in tcfg["value_maps"].items():
                lines.append(f"      {fld}: {scalar(mapping or {})}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    p = Path(path)
    return normalize_schema(read_yaml(p) if p.exists() else {})


def save_schema(cfg: dict[str, Any], path: str | Path = DEFAULT_SCHEMA_PATH) -> None:
    Path(path).write_text(dump_schema(cfg), encoding="utf-8")
