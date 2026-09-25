from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pytest

from src.load import normalize
from src.load.events import build_journal, journal_hash
from src.load.loader import load_all
from src.synthetic.generate import SyntheticConfig, generate


def test_generation_is_deterministic():
    cfg = SyntheticConfig(seed=11, n_payments=3_000)
    a, b = generate(cfg), generate(cfg)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_generation_hits_target_volume():
    for target in (5_000, 40_000):
        n = len(generate(SyntheticConfig(seed=3, n_payments=target))["payment"])
        assert abs(n - target) / target < 0.10


def test_generated_ground_truth_is_consistent():
    d = generate(SyntheticConfig(seed=5, n_payments=5_000))
    inv, imp = d["invoice"], d["imputation"]
    # Solde final = montant initial − somme imputée, jamais négatif.
    paid = imp.merge(inv[["invoice_id", "initial_amount"]], on="invoice_id")
    last = paid.sort_values("updated_at").groupby("invoice_id").last()
    assert (last["residual_amount"] >= 0).all()
    final = inv.set_index("invoice_id")["current_amount"]
    assert (final.loc[last.index] == last["residual_amount"]).all()
    # Chaque paiement imputé cite des factures de son débiteur uniquement.
    pay_debtor = imp.merge(inv[["invoice_id", "debtor_id"]], on="invoice_id").groupby("payment_id")["debtor_id"].nunique()
    assert (pay_debtor == 1).all()


def test_parallel_normalization_matches_sequential(synthetic_dir, synthetic_schema, monkeypatch):
    sequential = load_all(synthetic_schema, synthetic_dir, workers=1)
    monkeypatch.setattr(normalize, "PARALLEL_MIN_VALUES", 0)
    monkeypatch.setattr(normalize, "_CHUNK_SIZE", 700)
    parallel = load_all(synthetic_schema, synthetic_dir, workers=2)
    for name, df in sequential.tables.items():
        pd.testing.assert_frame_equal(df, parallel.tables[name])
    assert journal_hash(build_journal(sequential)[0]) == journal_hash(build_journal(parallel)[0])


@pytest.mark.parametrize("fn", [normalize._label_numbers_of_norm, normalize._keys_of_compact])
def test_map_unique_with_executor(fn, monkeypatch):
    values = pd.Series(["FA 0012", "123 FACT", "FA 0012", "", "X"] * 3)
    monkeypatch.setattr(normalize, "PARALLEL_MIN_VALUES", 0)
    monkeypatch.setattr(normalize, "_CHUNK_SIZE", 2)
    with ProcessPoolExecutor(max_workers=2) as ex:
        par = normalize._map_unique(values, fn, ex)
    assert par.tolist() == normalize._map_unique(values, fn).tolist()
