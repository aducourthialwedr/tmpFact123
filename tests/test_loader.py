import shutil

import pandas as pd
import pytest

from src.config import DEFAULT_SCHEMA_PATH, REPO_ROOT, read_yaml
from src.load.loader import SchemaConfigError, _to_cents, load_all


# --- Montants -----------------------------------------------------------------

def cents(value, unit, sep="."):
    out, invalid = _to_cents(pd.Series([value], dtype=object), unit, sep)
    if invalid:
        raise ValueError(value)
    return None if pd.isna(out.iloc[0]) else int(out.iloc[0])


@pytest.mark.parametrize(
    "value, unit, sep, expected",
    [
        ("0.29", "units", ".", 29),          # piège flottant classique
        ("1234.56", "units", ".", 123456),
        ("-12.50", "units", ".", -1250),
        ("1 234,56", "units", ",", 123456),
        ("1.234,56", "units", ",", 123456),
        ("12.340", "units", ".", 1234),
        (".5", "units", ".", 50),
        ("150", "cents", ".", 150),
        ("", "units", ".", None),
        (None, "units", ".", None),
    ],
)
def test_cents_conversion_from_text(value, unit, sep, expected):
    assert cents(value, unit, sep) == expected


@pytest.mark.parametrize("value, unit", [("12.345", "units"), ("12.5", "cents"), ("abc", "units"), ("1-2", "units")])
def test_cents_conversion_rejects_invalid(value, unit):
    with pytest.raises(ValueError):
        cents(value, unit)


def test_cents_conversion_from_numeric_columns():
    out, invalid = _to_cents(pd.Series([0.29, 1.1, None]), "units", ".")
    assert out.tolist() == [29, 110, pd.NA] and invalid == 0
    out, invalid = _to_cents(pd.Series([1999, 5]), "cents", ".")
    assert out.tolist() == [1999, 5] and invalid == 0
    out, invalid = _to_cents(pd.Series([0.125]), "units", ".")
    assert invalid == 1


# --- Configuration --------------------------------------------------------------

def test_empty_schema_template_lists_every_missing_mapping():
    with pytest.raises(SchemaConfigError) as exc:
        load_all(read_yaml(DEFAULT_SCHEMA_PATH), ".")
    problems = exc.value.problems
    assert any("amount_unit" in p for p in problems)
    for table in ("payment", "invoice", "imputation", "assignor", "debtor", "agreement"):
        assert any(f"tables.{table}.source" in p for p in problems)
    assert "tables.imputation.columns.residual_amount : colonne réelle non renseignée" in problems
    # Les tables optionnelles non configurées ne bloquent pas.
    assert not any("client_file" in p for p in problems)


def test_missing_required_column_mapping(synthetic_dir, synthetic_schema):
    synthetic_schema["tables"]["payment"]["columns"]["label"] = None
    with pytest.raises(SchemaConfigError) as exc:
        load_all(synthetic_schema, synthetic_dir)
    assert exc.value.problems == ["tables.payment.columns.label : colonne réelle non renseignée"]


def test_mapped_column_absent_from_file(synthetic_dir, synthetic_schema):
    synthetic_schema["tables"]["payment"]["columns"]["label"] = "libelle"
    with pytest.raises(SchemaConfigError, match="libelle"):
        load_all(synthetic_schema, synthetic_dir)


def test_real_column_names_are_mapped(synthetic_dir, synthetic_schema, tmp_path):
    for f in synthetic_dir.iterdir():
        shutil.copy(f, tmp_path / f.name)
    pay = pd.read_csv(tmp_path / "payment.csv", dtype=str, keep_default_na=False)
    pay = pay.rename(columns={"payment_id": "ID_OPE", "label": "LIB_OPE"})
    pay.to_csv(tmp_path / "payment_real.csv", index=False, sep=";")
    tcfg = synthetic_schema["tables"]["payment"]
    tcfg["source"] = "payment_real.csv"
    tcfg["read_options"] = {"sep": ";"}
    tcfg["columns"]["payment_id"] = "ID_OPE"
    tcfg["columns"]["label"] = "LIB_OPE"

    renamed = load_all(synthetic_schema, tmp_path).tables["payment"]
    reference = load_all(read_yaml(REPO_ROOT / "config" / "schema.synthetic.yaml"),
                         synthetic_dir).tables["payment"]
    pd.testing.assert_frame_equal(renamed, reference)


# --- Chargement ----------------------------------------------------------------

def test_load_synthetic_types_and_conventions(synthetic_dir, synthetic_schema):
    data = load_all(synthetic_schema, synthetic_dir)
    pay, inv, imp = (data.tables[n] for n in ("payment", "invoice", "imputation"))

    assert str(pay["amount"].dtype) == "Int64"
    assert str(inv["initial_amount"].dtype) == "Int64"
    assert str(pay["value_date"].dtype) == "datetime64[us]"
    assert (pay["value_date"] == pay["value_date"].dt.normalize()).all()
    assert set(imp["status"].unique()) <= {"FULL", "PARTIAL"}   # value_maps appliqué

    # Fuite : current_amount n'est jamais dans la table invoice, seulement en audit.
    assert "current_amount" not in inv.columns
    assert "current_amount" not in data.mapped_fields["invoice"]
    assert list(data.audit["invoice_current_amount"].columns) == ["invoice_id", "current_amount"]

    # Normalisation appliquée.
    for col in ("label_norm", "label_tokens", "label_numbers"):
        assert col in pay.columns
    assert "client_reference_keys" in inv.columns
    assert "name_tokens" in data.tables["debtor"].columns
    assert not pay["iban_debtor"].dropna().str.contains(" ").any()


def test_optional_unmapped_field_is_empty(synthetic_dir, synthetic_schema):
    data = load_all(synthetic_schema, synthetic_dir)
    assert "closed_at" not in data.mapped_fields["debtor"]
    assert data.tables["debtor"]["closed_at"].isna().all()
    assert data.tables["payment"]["bankroll_code"].isna().all()


def test_optional_table_absent(synthetic_dir, synthetic_schema):
    synthetic_schema["tables"]["client_file"]["source"] = None
    synthetic_schema["tables"]["client_file_line"]["source"] = None
    data = load_all(synthetic_schema, synthetic_dir)
    assert "client_file" not in data.tables
    assert any(i.table == "client_file" and i.kind == "table_non_configuree" for i in data.issues)


def test_timestamps_converted_to_utc(synthetic_dir, synthetic_schema):
    raw = pd.read_csv(synthetic_dir / "imputation.csv", dtype=str)
    data = load_all(synthetic_schema, synthetic_dir)
    local = pd.to_datetime(raw["updated_at"]).dt.tz_localize("Europe/Paris")
    expected = local.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[us]")
    pd.testing.assert_series_equal(data.tables["imputation"]["updated_at"], expected, check_names=False)


def test_invalid_values_are_reported(synthetic_dir, synthetic_schema, tmp_path):
    for f in synthetic_dir.iterdir():
        shutil.copy(f, tmp_path / f.name)
    pay = pd.read_csv(tmp_path / "payment.csv", dtype=str, keep_default_na=False)
    pay.loc[0, "amount"] = "12,3,4"
    pay.loc[1, "value_date"] = "31/02/2024"
    pay.to_csv(tmp_path / "payment.csv", index=False)

    data = load_all(synthetic_schema, tmp_path)
    kinds = {(i.field, i.kind): i.count for i in data.issues if i.table == "payment"}
    assert kinds[("amount", "valeur_invalide")] == 1
    assert kinds[("value_date", "valeur_invalide")] == 1
    assert pd.isna(data.tables["payment"].loc[0, "amount"])
