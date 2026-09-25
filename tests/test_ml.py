"""Étape 5 — réconciliation ML."""

import hashlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.reconcile_ml.decision import calibrate_online, decide, propose
from src.reconcile_ml.model import GROUP, PairModel, competition_features
from src.reconcile_ml.pipeline import PipelineMatcher, fit_ml
from src.settings import DecisionSettings, RulesConfig, SetSettings, Settings
from src.timeline.loop import run_replay
from tests.test_timeline import ledger, truncate


def scored(rows):
    return pd.DataFrame(rows, columns=["row", "inv", "balance", "p"])


def test_single_exact_and_partial():
    cfg = SetSettings()
    props = propose(scored([(0, 1, 100_000, 0.9), (0, 2, 80_000, 0.2), (1, 3, 500_000, 0.8)]),
                    np.array([100_000, 200_000]), cfg)
    p = props.set_index("row")
    assert p.loc[0, "invoices"] == (1,) and p.loc[0, "kind"] == "single" and p.loc[0, "amounts"] == (100_000,)
    assert p.loc[1, "kind"] == "partial" and p.loc[1, "amounts"] == (200_000,)
    assert p.loc[0, "margin"] == pytest.approx(0.9)          # I2 n'absorbe pas le paiement : pas de concurrent


def test_set_beats_incomplete_single():
    cfg = SetSettings()
    # Paiement de 1500 = 1000 + 500 ; une facture seule n'absorbe pas le paiement.
    props = propose(scored([(0, 1, 100_000, 0.95), (0, 2, 50_000, 0.9), (0, 3, 30_000, 0.1)]),
                    np.array([150_000]), cfg)
    assert props.iloc[0]["invoices"] == (1, 2) and props.iloc[0]["kind"] == "set"


def test_discount_single_competes_with_exact_set():
    cfg = SetSettings()
    # 1000 soldée à 2 % d'escompte près, ou 980 + 20 exactement : les deux sont proposés, marge faible.
    props = propose(scored([(0, 1, 100_000, 0.9), (0, 2, 90_000, 0.88), (0, 3, 8_000, 0.87)]),
                    np.array([98_000]), cfg)
    assert props.iloc[0]["margin"] < 0.05


def test_decide_by_kind():
    props = pd.DataFrame({"score": [0.99, 0.99, 0.5], "margin": [0.5, 0.01, 0.5], "kind": ["single", "single", "set"]})
    th = {"tau_high": 1.01, "tau_low": 0.3, "min_margin": 0.05, "kinds": {"single": 0.9, "set": 0.95}}
    assert decide(props, th).tolist() == ["auto", "review", "review"]


def test_online_calibration_counts_first_crossing():
    # Paiement A : score 0,6 (faux) puis 0,95 (juste) ; B : 0,9 (faux). Avec τ = 0,95 seul A-jour-2 passe.
    n = 40
    records = pd.DataFrame({"day": [1, 2, 1] * n, "payment_id": [f"A{i // 3}" if i % 3 < 2 else f"B{i // 3}"
                                                            for i in range(3 * n)],
                            "kind": "single", "score": [0.6, 0.95, 0.9] * n, "margin": 0.5})
    correct = np.array([False, True, False] * n)
    th = calibrate_online(records, correct, 0.99, DecisionSettings())
    assert th["kinds"]["single"] == pytest.approx(0.95)


def test_competition_features():
    comp = competition_features(np.array([0.2, 0.9, 0.5, 0.7]), np.array(["a", "a", "a", "b"]))
    assert comp["rank_in_payment"].tolist() == [3, 1, 2, 1]
    assert comp["score_margin"].tolist() == pytest.approx([-0.7, 0.4, -0.4, 0.7], abs=1e-6)


# --- Entraînement et boucle sur jeu synthétique ---------------------------------------------------------

@pytest.fixture(scope="module")
def trained(tmp_path_factory, synthetic_dir):
    from src.config import REPO_ROOT, read_yaml
    from src.load.events import build_journal, journal_hash
    from src.load.loader import load_all
    data = load_all(read_yaml(REPO_ROOT / "config" / "schema.synthetic.yaml"), synthetic_dir)
    journal, _ = build_journal(data)
    root = tmp_path_factory.mktemp("ml")
    interim = root / "interim"
    interim.mkdir()
    for name, df in data.tables.items():
        df.to_parquet(interim / f"{name}.parquet", index=False)
    journal.to_parquet(interim / "journal.parquet", index=False)
    import json
    (interim / "journal_meta.json").write_text(json.dumps({"journal_sha256": journal_hash(journal)}))
    settings = Settings.model_validate({"reconcile_ml": {"training": {"payment_sample": 1.0, "num_boost_round": 60}}})
    meta = fit_ml(interim, root / "model", settings, RulesConfig(), log=lambda m: None)
    return data, settings, PairModel.load(root / "model"), meta


def test_fit_produces_versioned_model(trained):
    _, _, model, meta = trained
    assert meta["journal_sha256"] and meta["featurization_version"]
    assert set(meta["thresholds"]["kinds"]) <= {"single", "partial", "set"}
    assert meta["metrics"]["validation"]["auc"] > 0.9


def test_model_roundtrip(trained, tmp_path):
    data, settings, model, _ = trained
    ds = pd.DataFrame(np.random.default_rng(0).random((20, len(model.features))), columns=model.features)
    ds[GROUP] = np.repeat(["a", "b"], 10)
    model.save(tmp_path / "m")
    again = PairModel.load(tmp_path / "m")
    assert np.allclose(model.predict(ds, ds[GROUP].to_numpy()), again.predict(ds, ds[GROUP].to_numpy()))


def _pipeline_hashes(data, settings, model, start, end):
    state = ledger(data)
    matcher = PipelineMatcher(state, settings, RulesConfig(), model)
    hashes = {}

    def hook(ctx, row):
        pass

    class Probe:
        name = "probe"

        def process(self, ctx):
            dec = matcher.process(ctx)
            hashes[ctx.day] = hashlib.sha256(dec.sort_values(["payment_id", "invoice_id"]).round(9)
                                             .to_csv(index=False).encode()).hexdigest()
            return dec

    res = run_replay(state, Probe(), start, end, retention_days=30)
    return hashes, res


def test_pipeline_runs_without_leak(trained):
    data, settings, model, _ = trained
    start, end, cutoff_day = date(2024, 11, 5), date(2024, 11, 30), date(2024, 11, 18)
    full, res = _pipeline_hashes(data, settings, model, start, end)
    assert (res.decisions["step"] == "ml").any()
    cut, _ = _pipeline_hashes(truncate(data, pd.Timestamp(cutoff_day) + pd.Timedelta(days=1)), settings, model,
                              start, end)
    days = [d for d in full if d.date() <= cutoff_day]
    assert [full[d] for d in days] == [cut[d] for d in days]


def test_candidates_in_blocks_match_single_pass(trained, monkeypatch):
    """Découpage du lot par charge de paires : décisions identiques à un passage unique."""
    import src.reconcile_ml.features as features
    data, settings, model, _ = trained
    start, end = date(2024, 11, 5), date(2024, 11, 8)
    whole, _ = _pipeline_hashes(data, settings, model, start, end)
    monkeypatch.setattr(features, "CANDIDATE_PAIR_BUDGET", 500)
    blocks, _ = _pipeline_hashes(data, settings, model, start, end)
    assert whole == blocks


def test_features_in_blocks_match_single_pass(trained, monkeypatch):
    import src.reconcile_ml.pipeline as pipeline
    data, settings, model, _ = trained
    start, end = date(2024, 11, 5), date(2024, 11, 7)
    whole, _ = _pipeline_hashes(data, settings, model, start, end)
    monkeypatch.setattr(pipeline, "FEATURE_BLOCK_PAIRS", 300)
    blocks, _ = _pipeline_hashes(data, settings, model, start, end)
    assert whole == blocks
