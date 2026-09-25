"""Relecture des sorties de l'étape 1 (tables normalisées + journal) pour les étapes suivantes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.load.canonical import TABLES
from src.load.events import journal_hash
from src.load.loader import LoadedData


class InterimError(RuntimeError):
    pass


def load_interim(directory: str | Path, verify: bool = True) -> tuple[LoadedData, pd.DataFrame, dict]:
    """Tables, journal et métadonnées de l'étape 1. Vérifie l'empreinte du journal si `verify`."""
    d = Path(directory)
    meta_path = d / "journal_meta.json"
    if not meta_path.exists():
        raise InterimError(f"aucune sortie de l'étape 1 dans {d} : lancer d'abord le chargement")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    names = [*TABLES, "party_iban"]
    tables = {name: pd.read_parquet(d / f"{name}.parquet") for name in names if (d / f"{name}.parquet").exists()}
    journal = pd.read_parquet(d / "journal.parquet")
    if verify and journal_hash(journal) != meta["journal_sha256"]:
        raise InterimError("le journal ne correspond pas à son empreinte : relancer l'étape 1")
    mapped = meta.get("mapped_fields") or {
        name: [f.name for f in TABLES[name].fields if f.name in df.columns and df[f.name].notna().any()]
        for name, df in tables.items() if name in TABLES
    }
    return LoadedData(tables=tables, mapped_fields=mapped), journal, meta
