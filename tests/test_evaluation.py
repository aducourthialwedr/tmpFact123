"""Étape 6 — harnais d'évaluation."""

import math
from datetime import date

import pandas as pd
import pytest

from src.evaluation.metrics import automation_at_precision, evaluate, ground_truth, precision_curve
from src.load.events import build_journal
from src.timeline.loop import AUTO, DECISION_COLUMNS, REVIEW, NullMatcher, run_replay
from src.timeline.state import LedgerState


def imp(pairs):
    return pd.DataFrame(pairs, columns=["payment_id", "invoice_id"])


def test_ground_truth_group_types():
    truth = ground_truth(imp([
        ("P1", "I1"),                       # 1↔1
        ("P2", "I2"), ("P2", "I3"),         # 1↔n
        ("P3", "I4"), ("P4", "I4"),         # n↔1
        ("P5", "I5"), ("P5", "I6"), ("P6", "I6"),   # n↔n
    ])).set_index("payment_id")
    assert truth["group_type"].to_dict() == {"P1": "1↔1", "P2": "1↔n", "P3": "n↔1", "P4": "n↔1",
                                             "P5": "n↔n", "P6": "n↔n"}
    assert truth.loc["P2", "truth_invoices"] == ("I2", "I3")


def decisions(rows):
    df = pd.DataFrame([{**dict(zip(DECISION_COLUMNS, r)), "day": pd.Timestamp("2024-03-05")} for r in rows])
    return df


def test_evaluate_counts_precision_and_rate():
    truth = ground_truth(imp([("P1", "I1"), ("P2", "I2"), ("P2", "I3"), ("P3", "I4"), ("P4", "I5")]))
    scope = pd.DataFrame({"payment_id": ["P1", "P2", "P3", "P4", "P9"],
                          "arrival_day": pd.to_datetime(["2024-03-05"] * 5)})
    dec = decisions([
        ("P1", "I1", 100, AUTO, "rules", "R2", 1, 1.0),       # correct
        ("P2", "I2", 100, AUTO, "ml", None, None, 0.9),       # incomplet (manque I3) → faux
        ("P3", "I4", 100, REVIEW, "ml", None, None, 0.8),     # revue, proposition correcte
        ("P9", "I9", 100, AUTO, "ml", None, None, 0.7),       # pas d'imputation réelle → faux
    ])
    ev = evaluate(dec, truth, scope, target_precision=0.995, current_automation_rate=0.2)
    s = ev.summary
    assert (s["paiements_périmètre"], s["auto"], s["auto_corrects"], s["revue"]) == (5, 3, 1, 1)
    assert s["taux_automatisation"] == pytest.approx(3 / 5)
    assert s["précision"] == pytest.approx(1 / 3)
    assert not s["cible_atteinte"]
    # Paires : décidées {P1-I1, P2-I2, P9-I9}, réelles {P1-I1, P2-I2, P2-I3, P3-I4, P4-I5}.
    assert s["précision_paires"] == pytest.approx(2 / 3)
    assert s["rappel_paires"] == pytest.approx(2 / 5)
    # Courbe : propositions triées par score 1.0 (ok), 0.9 (ko), 0.8 (ok), 0.7 (ko).
    assert s["taux_automatisation_à_précision_cible"] == pytest.approx(1 / 5)
    cascade = ev.cascade.set_index("niveau")["taux_automatisation"]
    assert cascade["Étape 4 — règles seules"] == pytest.approx(1 / 5)
    assert ev.cascade["gain_points"].iloc[1] == pytest.approx(0.0)
    assert ev.by_group.set_index("group_type").loc["1↔n", "auto"] == 1


def test_first_auto_decision_wins_over_earlier_review():
    truth = ground_truth(imp([("P1", "I1")]))
    scope = pd.DataFrame({"payment_id": ["P1"], "arrival_day": pd.to_datetime(["2024-03-05"])})
    dec = pd.concat([
        decisions([("P1", "I2", 1, REVIEW, "ml", None, None, 0.5)]),
        decisions([("P1", "I1", 1, AUTO, "ml", None, None, 0.99)]).assign(day=pd.Timestamp("2024-03-07")),
    ])
    assert evaluate(dec, truth, scope, 0.995).summary["auto_corrects"] == 1


def test_empty_decisions():
    truth = ground_truth(imp([("P1", "I1")]))
    scope = pd.DataFrame({"payment_id": ["P1"], "arrival_day": pd.to_datetime(["2024-03-05"])})
    s = evaluate(NullMatcher().process(None).assign(day=pd.Series(dtype="datetime64[us]")), truth, scope, 0.995).summary
    assert (s["auto"], s["taux_automatisation"], s["taux_automatisation_à_précision_cible"]) == (0, 0.0, 0.0)
    assert math.isnan(s["précision"]) and s["cible_atteinte"]


def test_precision_curve_cuts_only_between_distinct_scores():
    curve = precision_curve(pd.Series([0.9, 0.9, 0.5]).to_numpy(), pd.Series([True, False, True]).to_numpy(), 10)
    assert curve["taux_automatisation"].tolist() == [0.2, 0.3]
    assert automation_at_precision(curve, 0.6) == pytest.approx(0.3)
    assert automation_at_precision(curve, 0.7) == 0.0


# --- Intégration : rejeu + évaluation -------------------------------------------------------------

class Oracle:
    """Auto-valide la vraie réponse dès que toutes ses factures existent à D (test du harnais uniquement)."""

    name = "oracle"

    def __init__(self, truth: pd.DataFrame, wrong_every: int = 0):
        self.truth = truth.set_index("payment_id")["truth_invoices"].to_dict()
        self.wrong_every = wrong_every
        self.count = 0

    def process(self, ctx):
        rows = []
        for pid in ctx.batch["payment_id"]:
            invoices = self.truth.get(pid)
            if not invoices:
                continue
            known = ctx.state.invoices(list(invoices), ctx.as_of)
            if len(known) != len(invoices):
                continue
            self.count += 1
            chosen = invoices[:-1] if self.wrong_every and self.count % self.wrong_every == 0 and len(invoices) > 1 \
                else invoices
            rows += [(pid, i, 0, AUTO, "ml", None, None, 1.0) for i in chosen]
        return pd.DataFrame(rows, columns=DECISION_COLUMNS)


@pytest.fixture(scope="module")
def replay_inputs(synthetic_dir):
    from src.config import REPO_ROOT, read_yaml
    from src.load.loader import load_all
    data = load_all(read_yaml(REPO_ROOT / "config" / "schema.synthetic.yaml"), synthetic_dir)
    journal, _ = build_journal(data)
    truth = ground_truth(data.tables["imputation"])
    start, end = date(2024, 10, 1), date(2024, 11, 30)
    arrivals = journal[(journal["event_type"] == "PAYMENT_RECEIVED") & (journal["ts"] >= pd.Timestamp(start))
                       & (journal["ts"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    scope = pd.DataFrame({"payment_id": arrivals["entity_id"].to_numpy(), "arrival_day": arrivals["ts"].to_numpy()})
    return data, journal, truth, scope, start, end


def test_oracle_replay_is_fully_precise(replay_inputs):
    data, journal, truth, scope, start, end = replay_inputs
    res = run_replay(LedgerState(data, journal), Oracle(truth), start, end)
    s = evaluate(res.decisions, truth, scope, 0.995).summary
    assert s["auto"] > 0.8 * s["avec_imputation_réelle"]
    assert s["précision"] == 1.0 and s["cible_atteinte"]


def test_wrong_decisions_are_detected(replay_inputs):
    data, journal, truth, scope, start, end = replay_inputs
    res = run_replay(LedgerState(data, journal), Oracle(truth, wrong_every=2), start, end)
    s = evaluate(res.decisions, truth, scope, 0.995).summary
    assert s["précision"] < 1.0 and not s["cible_atteinte"]
    assert s["précision_paires"] == 1.0          # les paires décidées restent justes, l'ensemble est incomplet


def test_null_replay_scores_zero(replay_inputs):
    data, journal, truth, scope, start, end = replay_inputs
    res = run_replay(LedgerState(data, journal), NullMatcher(), start, end)
    s = evaluate(res.decisions, truth, scope, 0.995).summary
    assert s["paiements_périmètre"] == len(scope) and s["auto"] == 0
