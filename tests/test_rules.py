"""Étape 4 — règles déterministes : jeu figé de cas de non-régression."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.reconcile_rules.matcher import R1, R2, R3, R4, R5, RulesMatcher
from src.reconcile_rules.subset import AMBIGUOUS, BUDGET, NONE, UNIQUE, exact_subset
from src.settings import RulesConfig, Settings
from src.timeline.loop import run_replay
from tests.test_allocation import inv, pay
from tests.test_timeline import enriched, ledger

DEBTORS = [
    {"party_id": "D1", "name": "Dupont Bâtiment SARL", "iban": "FR76 1111", "opened_at": "2024-01-01"},
    {"party_id": "D2", "name": "Martin Transports SAS", "iban": "FR76 2222", "opened_at": "2024-01-01"},
    {"party_id": "D3", "name": "Durand Élec", "iban": "FR76 3333", "opened_at": "2024-01-01"},
    {"party_id": "D4", "name": "Petit Logistique", "opened_at": "2024-01-01"},
    {"party_id": "D5", "name": "Leroy Métal", "iban": "FR76 5555", "opened_at": "2024-01-01"},
]
INVOICES = [
    inv("I1", "D1", "FA0000001", 1000),
    inv("I2", "D2", "FA0000002", 2500), inv("I4", "D2", "FA0000004", 701), inv("I5", "D2", "FA0000005", 701),
    inv("I3", "D3", "FA0000003", 777), inv("I6", "D3", "FA0000006", 300), inv("I7", "D3", "FA0000007", 5000),
    inv("I8", "D4", "FA0000008", 900),
    inv("I9", "D5", "FA0000009", 1200),
]

# Jeu figé : (paiement, attendu) — attendu = (règle, factures) ou None (part à l'étape 5).
CASES = [
    (pay("P1", "VIR FA0000001", 1000), (R2, ("I1",))),
    (pay("P2", "PAIEMENT", 2500, "FR76 2222"), (R3, ("I2",))),
    (pay("P3", "PAIEMENT", 701, "FR76 2222"), None),                  # deux factures de 701 : ambigu
    (pay("P4", "VIREMENT", 1077, "FR76 3333"), (R4, ("I3", "I6"))),
    (pay("P5", "FACT FA0000008", 400), (R5, ("I8",))),
    (pay("P6", "VIR", 1200), (R1, ("I9",))),
    (pay("P7", "PAIEMENT DIVERS", 999), None),                        # rien d'exploitable
]


def world(payments, imputations=(), rules: RulesConfig | None = None):
    data = enriched(
        assignor=[{"party_id": "A1", "name": "CEDANT", "opened_at": "2023-01-01"}],
        debtor=DEBTORS,
        agreement=[{"agreement_id": f"AG{d['party_id']}", "debtor_id": d["party_id"], "client_id": "A1",
                    "created_at": "2024-01-01"} for d in DEBTORS],
        invoice=INVOICES, payment=list(payments), imputation=list(imputations),
        client_file=[{"file_id": "F1", "received_at": "2024-03-04 09:00", "total_amount": 1200,
                      "payment_date": "2024-03-05"}],
        client_file_line=[{"file_id": "F1", "invoice_reference": "FA0000009", "amount": 1200}],
    )
    state = ledger(data)
    return state, RulesMatcher(state, Settings(), rules or RulesConfig())


def decisions_by_payment(decisions: pd.DataFrame) -> dict:
    out = {}
    for pid, g in decisions.groupby("payment_id"):
        out[pid] = (g["rule_id"].iloc[0], tuple(sorted(g["invoice_id"])))
    return out


@pytest.fixture(scope="module")
def frozen():
    state, matcher = world([p for p, _ in CASES])
    res = run_replay(state, matcher, date(2024, 3, 5), date(2024, 3, 5), retention_days=5)
    return res.decisions


@pytest.mark.parametrize("payment, expected", CASES, ids=[p["payment_id"] for p, _ in CASES])
def test_frozen_case(frozen, payment, expected):
    assert decisions_by_payment(frozen).get(payment["payment_id"]) == expected


def test_decisions_carry_rule_and_version(frozen):
    assert (frozen["rule_version"] == RulesConfig().version).all()
    assert (frozen["step"] == "rules").all() and (frozen["action"] == "auto").all()
    p5 = frozen[frozen["payment_id"] == "P5"]
    assert p5["amount"].tolist() == [400]                              # partiel : montant payé


def test_same_rule_tie_on_one_invoice_rejects_both():
    state, matcher = world([pay("P1", "VIR FA0000001", 1000), pay("PX", "VIR FA0000001", 1000)])
    res = run_replay(state, matcher, date(2024, 3, 5), date(2024, 3, 5), retention_days=5)
    assert res.decisions.empty


def test_engine_claims_prevent_double_settlement():
    payments = [pay("P1", "VIR FA0000001", 1000),
                {**pay("PY", "VIR FA0000001", 1000), "value_date": "2024-03-06"}]
    state, matcher = world(payments)
    res = run_replay(state, matcher, date(2024, 3, 5), date(2024, 3, 6), retention_days=5)
    assert decisions_by_payment(res.decisions) == {"P1": (R2, ("I1",))}


def test_claim_released_when_real_imputation_arrives():
    payments = [pay("P1", "VIR FA0000001", 400), {**pay("PZ", "VIR FA0000001", 600), "value_date": "2024-03-07"}]
    imputations = [{"payment_id": "P1", "invoice_id": "I1", "updated_at": "2024-03-05 10:00",
                    "residual_amount": 600, "status": "PARTIAL"}]
    state, matcher = world(payments, imputations)
    res = run_replay(state, matcher, date(2024, 3, 5), date(2024, 3, 7), retention_days=5)
    # P1 partiel (R5) ; l'imputation réelle remplace la réservation ; PZ solde le reste (R2), sans double comptage.
    assert decisions_by_payment(res.decisions) == {"P1": (R5, ("I1",)), "PZ": (R2, ("I1",))}


def test_disabled_rule_falls_through_to_next():
    rules = RulesConfig()
    for r in rules.rules:
        if r.id == R2:
            r.enabled = False
    state, matcher = world([pay("P1", "VIR FA0000001", 1000)], rules=rules)
    res = run_replay(state, matcher, date(2024, 3, 5), date(2024, 3, 5), retention_days=5)
    assert decisions_by_payment(res.decisions) == {"P1": (R3, ("I1",))}


def test_proposals_record_each_rule_alone(frozen):
    state, matcher = world([p for p, _ in CASES])
    run_replay(state, matcher, date(2024, 3, 5), date(2024, 3, 5), retention_days=5)
    props = matcher.proposals()
    # P1 : R2 et R3 proposent chacune I1 (mesure « seule »), la cascade retient R2.
    assert set(props.loc[props["payment_id"] == "P1", "rule_id"]) == {R2, R3}


# --- Sous-ensembles ----------------------------------------------------------------------------------

@pytest.mark.parametrize("amounts, target, kwargs, status, indices", [
    ([100, 250, 400, 50], 650, {}, UNIQUE, (1, 2)),
    ([100, 100, 300], 400, {}, AMBIGUOUS, ()),
    ([100, 200, 300], 1000, {}, NONE, ()),
    ([100, 200, 300, 400], 600, {"max_size": 2}, UNIQUE, (1, 3)),
    (list(range(1, 60)), 150, {"node_budget": 50}, BUDGET, ()),
])
def test_exact_subset(amounts, target, kwargs, status, indices):
    res = exact_subset(np.array(amounts), target, **kwargs)
    assert res.status == status and res.indices == indices
