"""Étape 3 — allocation des paiements aux débiteurs."""

import hashlib
from datetime import date

import pandas as pd
import pytest

from src.allocation.allocator import (
    AMOUNT, ASSIGNOR, CLIENT_FILE, DEBTOR_DIRECT, FIRM, IBAN, MULTIPLE, NAME, NONE, REFERENCE, TECHNICAL_ACCOUNT,
    UNKNOWN, Allocator,
)
from src.allocation.evaluate import AllocationProbe, allocation_metrics, truth_debtors
from src.config import REPO_ROOT, read_yaml
from src.load.loader import load_all
from src.settings import AllocationSettings
from src.timeline.loop import DailyIterator, empty_decisions, run_replay
from tests.test_timeline import enriched, ledger, truncate

D = pd.Timestamp("2024-03-05")


def settings(**signals) -> AllocationSettings:
    """Seuils de nom relâchés : sur trois débiteurs, tout mot a une part de 33 %."""
    base = {"name": {"max_token_share": 1.0, "min_similarity": 0.5}}
    for k, v in signals.items():
        base.setdefault(k, {}).update(v)
    return AllocationSettings.model_validate({"signals": base})


def inv(iid, debtor, ref, amount, created="2024-02-01"):
    return {"invoice_id": iid, "client_reference": ref, "creation_date": created, "due_date": "2024-03-01",
            "initial_amount": amount, "currency": "EUR", "debtor_id": debtor, "agreement_id": f"AG{debtor}"}


def pay(pid, label, amount, iban=None):
    return {"payment_id": pid, "value_date": "2024-03-05", "amount": amount, "currency": "EUR", "label": label,
            "iban_debtor": iban}


def dataset():
    return enriched(
        assignor=[{"party_id": "A1", "name": "CEDANT", "iban": "FR76 5555", "opened_at": "2023-01-01"}],
        debtor=[{"party_id": "D1", "name": "Dupont Bâtiment SARL", "iban": "FR76 1111", "opened_at": "2024-01-01"},
                {"party_id": "D2", "name": "Martin Transports SAS", "iban": "FR76 2222", "opened_at": "2024-01-01"},
                {"party_id": "D3", "name": "Durand Élec", "opened_at": "2024-01-01"}],
        agreement=[{"agreement_id": f"AG{d}", "debtor_id": d, "client_id": "A1", "created_at": "2024-01-01"}
                   for d in ("D1", "D2", "D3")],
        technical_account=[{"iban": "FR76 9999"}],
        invoice=[inv("I1", "D1", "FA0012345", 1000), inv("I2", "D2", "FA0099999", 2500),
                 inv("I3", "D3", "FA0077777", 777), inv("I9", "D2", "FA0055555", 5, created="2024-03-20")],
        payment=[
            pay("P1", "VIR FA0012345", 1000, "FR76 0000"),           # référence + montant → D1 ferme
            pay("P2", "VIREMENT", 12, "FR76 2222"),                  # IBAN direct → D2 ferme
            pay("P3", "VIR MARTIN TRANSPORTS", 13, "FR76 9999"),     # compte technique, nom → D2
            pay("P4", "PAIEMENT", 777),                              # montant seul → D3
            pay("P5", "FACT FA0055555", 5),                          # facture future → aucun
            pay("P6", "VIR", 2500),                                  # client file → D2 ferme
            pay("P7", "FA0012345 FA0099999", 3500),                  # deux références fortes → multiple
            pay("P8", "RETOUR", 40, "FR76 5555"),                    # IBAN du cédant
        ],
        imputation=[],
        client_file=[{"file_id": "F1", "received_at": "2024-03-04 09:00", "total_amount": 2500,
                      "payment_date": "2024-03-05"}],
        client_file_line=[{"file_id": "F1", "invoice_reference": "FA0099999", "amount": 2500}],
    )


def allocate(data, cfg=None):
    state = ledger(data)
    alloc = Allocator(state, cfg or settings())
    ctx = next(iter(DailyIterator(state, D.date(), D.date(), retention_days=5)))
    return alloc.allocate(ctx)


@pytest.fixture(scope="module")
def result():
    a = allocate(dataset())
    return a.payments.set_index("payment_id"), a.candidates


def cands(candidates, pid):
    return candidates[candidates["payment_id"] == pid].sort_values("rank")


def test_reference_and_amount_give_firm_allocation(result):
    payments, candidates = result
    assert payments.loc["P1", "status"] == FIRM and payments.loc["P1", "firm_debtor_id"] == "D1"
    top = cands(candidates, "P1").iloc[0]
    assert top["debtor_id"] == "D1" and top["signals"] == f"{REFERENCE}+{AMOUNT}"


def test_iban_routes(result):
    payments, candidates = result
    assert payments.loc["P2", "iban_route"] == DEBTOR_DIRECT and payments.loc["P2", "firm_debtor_id"] == "D2"
    assert payments.loc["P3", "iban_route"] == TECHNICAL_ACCOUNT
    assert payments.loc["P8", "iban_route"] == ASSIGNOR
    assert payments.loc["P4", "iban_route"] == UNKNOWN
    assert IBAN not in cands(candidates, "P3")["signals"].str.cat()


def test_name_signal_is_not_firm(result):
    payments, candidates = result
    c = cands(candidates, "P3")
    assert c["debtor_id"].tolist() == ["D2"] and c["signal"].iloc[0] == NAME
    assert payments.loc["P3", "status"] == MULTIPLE


def test_amount_alone(result):
    payments, candidates = result
    assert cands(candidates, "P4")["debtor_id"].tolist() == ["D3"]
    assert payments.loc["P4", "status"] == MULTIPLE


def test_future_invoice_is_never_used(result):
    payments, _ = result
    assert payments.loc["P5", "status"] == NONE


def test_client_file_attached_and_firm(result):
    payments, candidates = result
    assert payments.loc["P6", "client_file_id"] == "F1"
    assert payments.loc["P6", "firm_debtor_id"] == "D2"
    assert CLIENT_FILE in cands(candidates, "P6")["signals"].iloc[0]


def test_two_strong_debtors_are_not_firm(result):
    payments, candidates = result
    assert payments.loc["P7", "status"] == MULTIPLE
    assert set(cands(candidates, "P7")["debtor_id"]) == {"D1", "D2"}


def test_disabled_signals_produce_nothing():
    off = {k: {"enabled": False} for k in ("client_file", "reference", "iban", "name", "amount")}
    a = allocate(dataset(), settings(**off))
    assert (a.payments["status"] == NONE).all() and a.candidates.empty


def test_client_file_not_visible_before_reception():
    data = dataset()
    data.tables["client_file"].loc[0, "received_at"] = pd.Timestamp("2024-03-05 09:00")
    a = allocate(data)
    assert a.payments.set_index("payment_id").loc["P6", "client_file_id"] is None


# --- Métriques ------------------------------------------------------------------------------------

def test_allocation_metrics():
    first = pd.DataFrame({
        "payment_id": ["P1", "P2", "P3", "P4"], "status": [FIRM, FIRM, MULTIPLE, NONE],
        "firm_debtor_id": ["D1", "D9", None, None], "n_candidates": [1, 1, 2, 0],
        "iban_route": [DEBTOR_DIRECT, DEBTOR_DIRECT, UNKNOWN, UNKNOWN], "client_file_id": [None] * 4,
        "candidates": [("D1",), ("D9",), ("D5", "D3"), ()], "candidate_signals": [("iban",), ("iban",),
                                                                                  ("name", "amount"), ()],
    })
    truth = pd.DataFrame({"payment_id": ["P1", "P2", "P3", "P4"], "truth_debtors": [("D1",), ("D2",), ("D3",), ("D4",)]})
    m = allocation_metrics(first, first, truth, 0.99)["summary"]
    assert m["rappel_premier_passage"] == 0.5          # P1, P3
    assert m["top1"] == 0.25                           # P1
    assert m["taux_ferme"] == 0.5 and m["précision_ferme"] == 0.5
    assert m["sans_candidat"] == 0.25 and not m["cible_atteinte"]


def test_truth_debtors():
    imp = pd.DataFrame({"payment_id": ["P1", "P1", "P2"], "invoice_id": ["I1", "I2", "I3"]})
    invoices = pd.DataFrame({"invoice_id": ["I1", "I2", "I3"], "debtor_id": ["D2", "D1", "D1"]})
    t = truth_debtors(imp, invoices).set_index("payment_id")["truth_debtors"]
    assert t.to_dict() == {"P1": ("D1", "D2"), "P2": ("D1",)}


# --- Sans fuite, déterministe, rappel sur jeu synthétique -------------------------------------------

@pytest.fixture(scope="module")
def synthetic(synthetic_dir):
    return load_all(read_yaml(REPO_ROOT / "config" / "schema.synthetic.yaml"), synthetic_dir)


class AllocationSnapshot:
    name = "allocation-snapshot"

    def __init__(self, allocator):
        self.allocator = allocator
        self.hashes = {}

    def process(self, ctx):
        a = self.allocator.allocate(ctx)
        h = hashlib.sha256(a.candidates.round(9).to_csv(index=False).encode())
        h.update(a.payments.to_csv(index=False).encode())
        self.hashes[ctx.day] = h.hexdigest()
        return empty_decisions()


def _alloc_hashes(data, start, end):
    from src.settings import AllocationSettings as S
    state = ledger(data)
    snap = AllocationSnapshot(Allocator(state, S()))
    run_replay(state, snap, start, end, retention_days=30)
    return snap.hashes


def test_allocation_does_not_leak(synthetic):
    start, end = date(2024, 6, 1), date(2024, 7, 15)
    cutoff_day = date(2024, 6, 20)
    full = _alloc_hashes(synthetic, start, end)
    cut = _alloc_hashes(truncate(synthetic, pd.Timestamp(cutoff_day) + pd.Timedelta(days=1)), start, end)
    days = [d for d in full if d.date() <= cutoff_day]
    assert [full[d] for d in days] == [cut[d] for d in days]
    after = [d for d in full if d.date() > cutoff_day]
    assert full[after[0]] != cut[after[0]]


def test_allocation_is_deterministic(synthetic):
    start, end = date(2024, 6, 1), date(2024, 6, 10)
    assert _alloc_hashes(synthetic, start, end) == _alloc_hashes(synthetic, start, end)


def test_synthetic_recall(synthetic):
    state = ledger(synthetic)
    probe = AllocationProbe(Allocator(state, AllocationSettings()))
    run_replay(state, probe, date(2024, 9, 1), date(2024, 10, 31), retention_days=60)
    m = allocation_metrics(probe.first_pass(), probe.last_pass(),
                           truth_debtors(synthetic.tables["imputation"], synthetic.tables["invoice"]), 0.99)["summary"]
    assert m["rappel_premier_passage"] >= 0.99
    assert m["précision_ferme"] >= 0.995
