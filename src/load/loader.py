"""Chargement des tables sources vers le modèle canonique (brief §3).

Tout le mapping vers les noms réels vient de `config/schema.yaml`. Le loader
ne devine rien : un mapping requis manquant lève `SchemaConfigError` avec la
liste complète de ce qui reste à renseigner.

Conventions de sortie :
- identifiants en chaînes (`string`) ;
- montants en entiers de centimes (`Int64`) ;
- dates et horodatages en `datetime64[us]` naïfs exprimés en UTC.
"""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from src.arrow_ops import to_arrow
from src.load import normalize
from src.load.canonical import TABLES, FieldType, Table

DATETIME_DTYPE = "datetime64[us]"


class SchemaConfigError(ValueError):
    """La configuration de schéma est incomplète ou incohérente."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("config/schema.yaml incomplet :\n  - " + "\n  - ".join(problems))


@dataclass
class Issue:
    table: str
    field: str | None
    kind: str
    count: int
    detail: str = ""


@dataclass
class LoadedData:
    tables: dict[str, pd.DataFrame]
    # Champs canoniques effectivement mappés, par table.
    mapped_fields: dict[str, list[str]]
    issues: list[Issue] = field(default_factory=list)
    # Données chargées pour contrôle uniquement, jamais exposées à la pipeline.
    audit: dict[str, pd.DataFrame] = field(default_factory=dict)

    def has_table(self, name: str) -> bool:
        return name in self.tables


# --- Conversion de types ----------------------------------------------------


def _to_id(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.where(out != "", pd.NA)


def _to_text(s: pd.Series) -> pd.Series:
    return _to_id(s)


def _parse_datetime(s: pd.Series, fmt: str | None) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, format=fmt, errors="coerce")


def _to_date(s: pd.Series, fmt: str | None) -> pd.Series:
    parsed = _parse_datetime(s, fmt)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize().astype(DATETIME_DTYPE)


def _to_timestamp(s: pd.Series, fmt: str | None, source_tz: str | None) -> pd.Series:
    parsed = _parse_datetime(s, fmt)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    elif source_tz:
        parsed = (
            parsed.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
    return parsed.astype(DATETIME_DTYPE)


_AMOUNT_PATTERN = r"^(?P<sign>[+-]?)(?P<integer>[0-9]*)(?:\.(?P<frac>[0-9]*))?$"


def _to_cents(s: pd.Series, unit: str, decimal_sep: str) -> tuple[pd.Series, int]:
    """Convertit en entiers de centimes, sans jamais passer par un flottant pour le texte.

    `unit` : "cents" (valeurs déjà en centimes) ou "units" (ex. 1234.56).
    Retourne (série Int64, nombre de valeurs non vides invalides). Une valeur
    avec plus de décimales que l'unité n'en permet (hors zéros) est invalide.
    """
    scale_digits = 2 if unit == "units" else 0
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if pd.api.types.is_integer_dtype(s):
            return s.astype("Int64") * 10**scale_digits, 0
        scaled = s.astype("Float64") * 10**scale_digits
        rounded = scaled.round()
        ok = (scaled - rounded).abs() < 1e-6
        invalid = int((scaled.notna() & ~ok.fillna(False)).sum())
        return rounded.where(ok).astype("Int64"), invalid

    text = s.astype("str").str.replace(r"[\s ]+", "", regex=True)
    text = text.where(text != "")
    if decimal_sep != ".":
        text = text.str.replace(".", "", regex=False).str.replace(decimal_sep, ".", regex=False)
    arr = to_arrow(text).cast(pa.string())
    parts = pc.extract_regex(arr, _AMOUNT_PATTERN)
    integer, frac = parts.field("integer"), parts.field("frac")
    extra = pc.utf8_slice_codeunits(frac, scale_digits, 1 << 30)
    ok = pc.and_kleene(
        pc.is_valid(parts),
        pc.and_kleene(
            pc.greater(pc.add(pc.utf8_length(integer), pc.utf8_length(frac)), 0),
            pc.match_substring_regex(extra, "^0*$"),
        ),
    ).fill_null(False)
    head = pc.utf8_rpad(pc.utf8_slice_codeunits(frac, 0, scale_digits), scale_digits, "0")
    digits = pc.binary_join_element_wise(integer, head, "")
    digits = pc.if_else(pc.and_(ok, pc.greater(pc.utf8_length(digits), 0)), digits, "0")
    cents = pc.cast(digits, pa.int64())
    cents = pc.if_else(pc.equal(parts.field("sign"), "-"), pc.negate(cents), cents)
    ok_np = ok.to_numpy(zero_copy_only=False)
    out = pd.Series(cents.to_numpy(zero_copy_only=False), index=s.index).astype("Int64").where(ok_np)
    invalid = int((text.notna().to_numpy() & ~ok_np).sum())
    return out.astype("Int64"), invalid


# --- Lecture ----------------------------------------------------------------


def _read_source(path: Path, read_options: dict[str, Any]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **read_options)
    if suffix in (".csv", ".txt", ".tsv"):
        opts = {"dtype": str, "keep_default_na": False, "na_values": [""], **read_options}
        return pd.read_csv(path, **opts)
    raise SchemaConfigError([f"{path}: format de fichier non supporté ({suffix})"])


def _check_config(schema_cfg: dict[str, Any], base_dir: Path) -> list[str]:
    problems: list[str] = []
    if schema_cfg.get("amount_unit") not in ("cents", "units"):
        problems.append("amount_unit : renseigner 'cents' ou 'units'")
    tables_cfg = schema_cfg.get("tables") or {}
    for name, table in TABLES.items():
        tcfg = tables_cfg.get(name) or {}
        source = tcfg.get("source")
        if not source:
            if not table.required:
                continue
            problems.append(f"tables.{name}.source : fichier source non renseigné")
        elif not (base_dir / source).exists():
            problems.append(f"tables.{name}.source : fichier introuvable ({base_dir / source})")
        columns = tcfg.get("columns") or {}
        unknown = sorted(set(columns) - {f.name for f in table.fields})
        if unknown:
            problems.append(f"tables.{name}.columns : champs canoniques inconnus {unknown}")
        for f in table.fields:
            if f.required and not columns.get(f.name):
                problems.append(f"tables.{name}.columns.{f.name} : colonne réelle non renseignée")
    return problems


def _load_table(
    table: Table,
    tcfg: dict[str, Any],
    schema_cfg: dict[str, Any],
    base_dir: Path,
    issues: list[Issue],
) -> tuple[pd.DataFrame, list[str]]:
    raw = _read_source(base_dir / tcfg["source"], tcfg.get("read_options") or {})
    columns: dict[str, str] = {k: v for k, v in (tcfg.get("columns") or {}).items() if v}

    missing_cols = sorted(v for v in columns.values() if v not in raw.columns)
    if missing_cols:
        raise SchemaConfigError(
            [f"tables.{table.name} : colonnes absentes du fichier source {missing_cols}"]
        )

    unit = schema_cfg["amount_unit"]
    decimal_sep = schema_cfg.get("decimal_separator") or "."
    date_fmt = tcfg.get("date_format") or schema_cfg.get("date_format")
    ts_fmt = tcfg.get("timestamp_format") or schema_cfg.get("timestamp_format")
    source_tz = schema_cfg.get("source_timezone")
    value_maps: dict[str, dict[str, str]] = tcfg.get("value_maps") or {}

    out = pd.DataFrame(index=raw.index)
    for f in table.fields:
        if f.name not in columns:
            out[f.name] = _empty(f.type, len(raw), raw.index)
            continue
        col = raw[columns[f.name]]
        non_null_in = int(col.notna().sum())
        if f.type is FieldType.ID:
            conv = _to_id(col)
        elif f.type is FieldType.TEXT:
            conv = _to_text(col)
        elif f.type is FieldType.DATE:
            conv = _to_date(col, date_fmt)
        elif f.type is FieldType.TIMESTAMP:
            conv = _to_timestamp(col, ts_fmt, source_tz)
        elif f.type is FieldType.AMOUNT:
            conv, _ = _to_cents(col, unit, decimal_sep)
        else:  # pragma: no cover
            raise AssertionError(f.type)

        if f.name in value_maps:
            # value_maps : {valeur_canonique: [valeurs réelles]} ou {canonique: réelle}
            reverse: dict[str, str] = {}
            for canonical_value, real in value_maps[f.name].items():
                for r in real if isinstance(real, list) else [real]:
                    reverse[str(r)] = canonical_value
            conv = conv.map(lambda v: reverse.get(v, v) if not pd.isna(v) else v).astype("string")

        lost = non_null_in - int(conv.notna().sum())
        if lost > 0:
            issues.append(Issue(table.name, f.name, "valeur_invalide", lost,
                                "valeurs non vides non convertibles, mises à NA"))
        out[f.name] = conv

    return out.reset_index(drop=True), sorted(columns)


def _empty(ftype: FieldType, n: int, index: pd.Index) -> pd.Series:
    if ftype in (FieldType.DATE, FieldType.TIMESTAMP):
        return pd.Series(pd.NaT, index=index, dtype=DATETIME_DTYPE)
    if ftype is FieldType.AMOUNT:
        return pd.Series(pd.NA, index=index, dtype="Int64")
    return pd.Series(pd.NA, index=index, dtype="string")


def _enrich(tables: dict[str, pd.DataFrame], executor: Executor | None = None) -> None:
    """Applique la normalisation (brief §3.3) aux libellés, références et noms."""
    pay = tables["payment"]
    for col in ("iban_debtor", "iban_creditor"):
        pay[col] = normalize.normalize_iban(pay[col])
    tables["payment"] = pd.concat([pay, normalize.enrich_label(pay["label"], executor=executor)], axis=1)

    inv = tables["invoice"]
    tables["invoice"] = pd.concat(
        [
            inv,
            normalize.enrich_reference(inv["client_reference"], "client_reference", executor),
            normalize.enrich_reference(inv["internal_reference"], "internal_reference", executor),
        ],
        axis=1,
    )

    for name in ("assignor", "debtor"):
        party = tables[name]
        party["iban"] = normalize.normalize_iban(party["iban"])
        tables[name] = pd.concat([party, normalize.enrich_name(party["name"])], axis=1)

    if "technical_account" in tables:
        tables["technical_account"]["iban"] = normalize.normalize_iban(
            tables["technical_account"]["iban"]
        )

    if "client_file" in tables:
        cf = tables["client_file"]
        cf["iban"] = normalize.normalize_iban(cf["iban"])
        tables["client_file"] = pd.concat(
            [cf, normalize.enrich_label(cf["payment_reference"], "payment_reference", executor)], axis=1
        )

    if "client_file_line" in tables:
        lines = tables["client_file_line"]
        tables["client_file_line"] = pd.concat(
            [lines, normalize.enrich_reference(lines["invoice_reference"], "invoice_reference", executor)],
            axis=1,
        )


def load_all(
    schema_cfg: dict[str, Any], base_dir: str | Path | None = None, workers: int = 1
) -> LoadedData:
    """Charge toutes les tables configurées et applique la normalisation.

    `base_dir` : répertoire des fichiers sources. Par défaut `schema_cfg["base_dir"]`.
    `workers`  : processus pour la normalisation des gros volumes (1 = séquentiel).
    Le résultat ne dépend pas de `workers`.
    """
    base = Path(base_dir or schema_cfg.get("base_dir") or ".")
    problems = _check_config(schema_cfg, base)
    if problems:
        raise SchemaConfigError(problems)

    tables_cfg = schema_cfg.get("tables") or {}
    tables: dict[str, pd.DataFrame] = {}
    mapped: dict[str, list[str]] = {}
    issues: list[Issue] = []
    for name, table in TABLES.items():
        tcfg = tables_cfg.get(name) or {}
        if not tcfg.get("source"):
            issues.append(Issue(name, None, "table_non_configuree", 0, "table optionnelle absente"))
            continue
        tables[name], mapped[name] = _load_table(table, tcfg, schema_cfg, base, issues)

    audit: dict[str, pd.DataFrame] = {}
    inv = tables["invoice"]
    if "current_amount" in mapped["invoice"]:
        audit["invoice_current_amount"] = inv[["invoice_id", "current_amount"]].copy()
    # Jamais de current_amount brut dans la pipeline (brief §4.1).
    tables["invoice"] = inv.drop(columns=["current_amount"])
    mapped["invoice"] = [f for f in mapped["invoice"] if f != "current_amount"]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            _enrich(tables, executor)
    else:
        _enrich(tables)
    return LoadedData(tables=tables, mapped_fields=mapped, issues=issues, audit=audit)
