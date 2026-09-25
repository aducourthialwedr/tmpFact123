from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from src.config import REPO_ROOT, read_yaml
from src.load.canonical import TABLES, FieldType
from src.load.loader import DATETIME_DTYPE, LoadedData
from src.synthetic.generate import SyntheticConfig, write_synthetic

SMALL = SyntheticConfig(seed=7, n_payments=5_000)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    write_synthetic(SMALL, out)
    return out


@pytest.fixture
def synthetic_schema() -> dict:
    return copy.deepcopy(read_yaml(REPO_ROOT / "config" / "schema.synthetic.yaml"))


def _typed(table: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = pd.DataFrame(index=range(len(df)))
    for f in TABLES[table].fields:
        col = df[f.name] if f.name in df.columns else pd.Series([None] * len(df))
        if f.type in (FieldType.DATE, FieldType.TIMESTAMP):
            out[f.name] = pd.to_datetime(col).astype(DATETIME_DTYPE)
        elif f.type is FieldType.AMOUNT:
            out[f.name] = pd.array(col.tolist(), dtype="Int64")
        else:
            out[f.name] = col.astype("string")
    return out


def make_data(**rows: list[dict]) -> LoadedData:
    """LoadedData construit à la main : toutes les tables requises, typées, sans normalisation."""
    tables = {}
    mapped = {}
    for name, table in TABLES.items():
        if name in rows or table.required:
            tables[name] = _typed(name, rows.get(name, []))
            given = set().union(*(r.keys() for r in rows.get(name, []))) if rows.get(name) else set()
            mapped[name] = sorted(given)
    tables["invoice"] = tables["invoice"].drop(columns=["current_amount"])
    return LoadedData(tables=tables, mapped_fields=mapped)
