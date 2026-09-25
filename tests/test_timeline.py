"""Étape 2 — état à date, boucle quotidienne, absence de fuite (brief §4.3)."""

import hashlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.load.events import build_journal
from src.load.loader import LoadedData, _enrich, load_all
from src.load.events import payment_event_time
from src.timeline.loop import AUTO, DECISION_COLUMNS, LeakError, NullMatcher, run_replay
from src.timeline.state import LedgerState, TemporalError
from tests.conftest import make_data
from tests.test_events import AGREEMENT, ASSIGNOR, DEBTOR, imputation, invoice, payment

D = pd.Timestamp


def enriched(**rows) -> LoadedData:
    data = make_data(**rows)
    _enrich(data.tables)
    return data


def ledger(data: LoadedData, window_days: int = 180) -> LedgerState:
    journal, _ = build_journal(data)
    return LedgerState(data, journal, window_days)


def base(**extra):
    return dict(debtor=DEBTOR, assignor=ASSIGNOR, agreement=AGREEMENT, **extra)


# --- LedgerState ----------------------------------------------------------------------------

def test_invoice_visible_only_after_creation_day():
    state = ledger(enriched(**base(invoice=[invoice("I1", created="2024-03-05")], payment=[], imputation=[])))
    state.advance_to("2024-03-05")
    assert state.open_invoices(["D1"], D("2024-03-05")).empty
    assert state.open_amount(["I1"], D("2024-03-05")).isna().all()
    state.advance_to("2024-03-06")
    assert state.open_invoices(["D1"], D("2024-03-06"))["invoice_id"].tolist() == ["I1"]
    assert state.open_amount(["I1"], D("2024-03-06")).tolist() == [1000]


def test_imputation_applies_from_next_day_and_closes_invoice():
    state = ledger(enriched(**base(
        invoice=[invoice("I1", amount=1000)], payment=[payment("P1", amount=1000), payment("P2", amount=600)],
        imputation=[imputation("P2", "I1", "2024-03-06 10:00", 400), imputation("P1", "I1", "2024-03-10 10:00", 0)])))
    state.advance_to("2024-03-06")
    assert state.open_amount(["I1"], D("2024-03-06")).tolist() == [1000]
    state.advance_to("2024-03-07")
    assert state.open_amount(["I1"], D("2024-03-07")).tolist() == [400]
    assert state.debtor_stats(["D1"], D("2024-03-07"))["open_invoice_count"].tolist() == [1]
    state.advance_to("2024-03-11")
    assert state.open_invoices(["D1"], D("2024-03-11")).empty
    stats = state.debtor_stats(["D1"], D("2024-03-11")).iloc[0]
    assert (stats["open_invoice_count"], stats["open_invoice_amount"]) == (0, 0)
    assert (stats["imputation_count"], stats["partial_payment_rate"]) == (2, 0.5)


def test_reads_require_matching_as_of_and_state_never_goes_back():
    state = ledger(enriched(**base(invoice=[invoice("I1")], payment=[], imputation=[])))
    with pytest.raises(TemporalError):
        state.open_amount(["I1"], D("2024-03-01"))          # pas encore avancé
    state.advance_to("2024-03-10")
    with pytest.raises(TemporalError):
        state.open_amount(["I1"], D("2024-03-09"))
    with pytest.raises(TemporalError):
        state.advance_to("2024-03-01")


def test_behavioral_window_is_strictly_prior_and_expires():
    state = ledger(enriched(**base(
        invoice=[invoice("I1", amount=1000)], payment=[payment("P1", amount=1000, value_date="2024-03-05")],
        imputation=[imputation("P1", "I1", "2024-03-06 10:00", 0)])), window_days=10)
    for day, expected in [("2024-03-06", 0), ("2024-03-07", 1), ("2024-03-16", 1), ("2024-03-17", 0)]:
        state.advance_to(day)
        assert state.debtor_stats(["D1"], D(day))["imputation_count"].iloc[0] == expected, day
    # Délai = date de valeur − échéance (2024-03-05 − 2024-03-01).
    state2 = ledger(enriched(**base(
        invoice=[invoice("I1", amount=1000)], payment=[payment("P1", amount=1000, value_date="2024-03-05")],
        imputation=[imputation("P1", "I1", "2024-03-06 10:00", 0)])))
    state2.advance_to("2024-03-07")
    assert state2.debtor_stats(["D1"], D("2024-03-07"))["mean_payment_delay"].iloc[0] == 4


def test_client_file_visible_after_reception():
    cf = [{"file_id": "F1", "received_at": "2024-03-05 14:00"}]
    lines = [{"file_id": "F1", "invoice_reference": "REFI1", "amount": 1000}]
    state = ledger(enriched(**base(invoice=[invoice("I1")], payment=[], imputation=[],
                                   client_file=cf, client_file_line=lines)))
    state.advance_to("2024-03-05")
    assert state.client_files(D("2024-03-05")).empty
    assert state.client_file_lines(["F1"], D("2024-03-05")).empty
    state.advance_to("2024-03-06")
    assert state.client_files(D("2024-03-06"))["file_id"].tolist() == ["F1"]
    assert len(state.client_file_lines(["F1"], D("2024-03-06"))) == 1


# --- Boucle quotidienne -------------------------------------------------------------------------

class Recorder:
    """Rapprocheur qui n'agit pas mais enregistre ce qu'il voit, et auto-valide sur demande."""

    name = "recorder"

    def __init__(self, auto_on: dict[str, list[tuple[str, str]]] | None = None):
        self.seen: dict[pd.Timestamp, list[tuple[str, bool, int]]] = {}
        self.auto_on = auto_on or {}

    def process(self, ctx):
        self.seen[ctx.day] = list(zip(ctx.batch["payment_id"], ctx.batch["is_new"], ctx.batch["days_pending"]))
        rows = [{"payment_id": p, "invoice_id": i, "amount": 1, "action": AUTO, "step": "test",
                 "rule_id": None, "rule_version": None, "score": 1.0}
                for p, i in self.auto_on.get(str(ctx.day.date()), [])]
        return pd.DataFrame(rows, columns=DECISION_COLUMNS)


def loop_data():
    return enriched(**base(
        invoice=[invoice("I1", created="2024-02-01"), invoice("I2", created="2024-02-01"),
                 invoice("I9", created="2024-03-20")],
        payment=[payment("P1", "2024-03-05"), payment("P2", "2024-03-05"), payment("P3", "2024-03-06")],
        imputation=[imputation("P1", "I1", "2024-03-07 10:00", 0)]))


def test_batch_contains_new_payments_and_carries_the_remainder():
    rec = Recorder()
    run_replay(ledger(loop_data()), rec, date(2024, 3, 4), date(2024, 3, 9), retention_days=2)
    seen = {d.strftime("%m-%d"): v for d, v in rec.seen.items()}
    assert seen["03-04"] == []
    assert seen["03-05"] == [("P1", True, 0), ("P2", True, 0)]
    assert seen["03-06"] == [("P1", False, 1), ("P2", False, 1), ("P3", True, 0)]
    # P1 imputé le 07 à 10 h : encore dans le lot du 07, sorti le 08.
    assert [p for p, *_ in seen["03-07"]] == ["P1", "P2", "P3"]
    # Rétention de 2 jours : P2 (arrivé le 05) est traité les 05, 06 et 07, puis sort.
    assert [p for p, *_ in seen["03-08"]] == ["P3"]
    assert seen["03-09"] == []


def test_retention_expiry_counts():
    res = run_replay(ledger(loop_data()), NullMatcher(), date(2024, 3, 5), date(2024, 3, 12), retention_days=2)
    assert res.daily["expired"].sum() == 2          # P2 et P3
    assert res.daily["left_imputed"].sum() == 1     # P1


def test_auto_decision_removes_payment_from_next_batch():
    rec = Recorder(auto_on={"2024-03-05": [("P2", "I2")]})
    res = run_replay(ledger(loop_data()), rec, date(2024, 3, 5), date(2024, 3, 6), retention_days=10)
    assert [p for p, *_ in rec.seen[D("2024-03-06")]] == ["P1", "P3"]
    assert res.decisions[["payment_id", "invoice_id", "action"]].values.tolist() == [["P2", "I2", "auto"]]


def test_decision_on_future_invoice_is_a_leak():
    rec = Recorder(auto_on={"2024-03-05": [("P2", "I9")]})     # I9 créée le 20
    with pytest.raises(LeakError, match="inconnue"):
        run_replay(ledger(loop_data()), rec, date(2024, 3, 5), date(2024, 3, 5))


def test_decision_on_payment_outside_batch_is_a_leak():
    rec = Recorder(auto_on={"2024-03-05": [("P3", "I1")]})     # P3 arrive le 06
    with pytest.raises(LeakError, match="hors du lot"):
        run_replay(ledger(loop_data()), rec, date(2024, 3, 5), date(2024, 3, 5))


# --- Tests obligatoires du brief (§4.3) sur jeu synthétique ----------------------------------------------

class Snapshot:
    """Enregistre, chaque jour, une empreinte de tout ce qu'un rapprocheur peut lire à D."""

    name = "snapshot"

    def __init__(self, debtors: list[str], invoices: list[str], agreements: list[str], assignors: list[str]):
        self.debtors, self.invoices, self.agreements, self.assignors = debtors, invoices, agreements, assignors
        self.hashes: dict[pd.Timestamp, str] = {}

    def process(self, ctx):
        s, t = ctx.state, ctx.as_of
        parts = [
            ctx.batch[["payment_id", "amount", "label", "is_new", "days_pending"]],
            s.open_invoices(self.debtors, t).drop(columns=["client_reference_keys", "internal_reference_keys"]),
            s.debtor_stats(self.debtors, t).round(9),
            s.open_amount(self.invoices, t).to_frame(),
            pd.DataFrame({"a": s.agreement_active(self.agreements, t)}),
            pd.DataFrame({"d": s.party_active("debtor", self.debtors, t)}),
            pd.DataFrame({"c": s.party_active("assignor", self.assignors, t)}),
            s.client_files(t)[["file_id"]] if len(s.client_files(t)) else pd.DataFrame(),
        ]
        h = hashlib.sha256()
        for p in parts:
            h.update(p.to_csv(index=False).encode())
        self.hashes[ctx.day] = h.hexdigest()
        return NullMatcher().process(ctx)


def truncate(data: LoadedData, cutoff: pd.Timestamp) -> LoadedData:
    """Données telles qu'elles existaient avant `cutoff` : tout événement ≥ cutoff supprimé."""
    t = {k: v.copy() for k, v in data.tables.items()}
    t["payment"] = t["payment"][payment_event_time(t["payment"]) < cutoff]
    t["invoice"] = t["invoice"][t["invoice"]["creation_date"] < cutoff]
    t["imputation"] = t["imputation"][t["imputation"]["updated_at"] < cutoff]
    t["client_file"] = t["client_file"][t["client_file"]["received_at"] < cutoff]
    t["client_file_line"] = t["client_file_line"][t["client_file_line"]["file_id"].isin(t["client_file"]["file_id"])]
    t["agreement"] = t["agreement"][t["agreement"]["created_at"] < cutoff]
    t["agreement"]["disabled_at"] = t["agreement"]["disabled_at"].where(t["agreement"]["disabled_at"] < cutoff)
    for role in ("assignor", "debtor"):
        p = t[role][~(t[role]["opened_at"] >= cutoff)].copy()
        p["closed_at"] = p["closed_at"].where(p["closed_at"] < cutoff)
        t[role] = p
    return LoadedData(tables={k: v.reset_index(drop=True) for k, v in t.items()},
                      mapped_fields=data.mapped_fields)


@pytest.fixture(scope="module")
def synthetic(synthetic_dir):
    from src.config import REPO_ROOT, read_yaml
    return load_all(read_yaml(REPO_ROOT / "config" / "schema.synthetic.yaml"), synthetic_dir)


def _snapshot(data: LoadedData, full: LoadedData, start, end) -> tuple[dict, pd.DataFrame]:
    t = full.tables
    snap = Snapshot(t["debtor"]["party_id"].tolist(), t["invoice"]["invoice_id"].tolist(),
                    t["agreement"]["agreement_id"].tolist(), t["assignor"]["party_id"].tolist())
    res = run_replay(ledger(data), snap, start, end, retention_days=30)
    return snap.hashes, res.daily.drop(columns="seconds")


def test_removing_events_after_d_changes_nothing_computed_up_to_d(synthetic):
    start, end = date(2024, 5, 1), date(2024, 7, 31)
    full_hashes, full_daily = _snapshot(synthetic, synthetic, start, end)
    for cutoff_day in (date(2024, 5, 20), date(2024, 6, 30)):
        cutoff = D(cutoff_day) + pd.Timedelta(days=1)            # événements postérieurs au jour D supprimés
        cut_hashes, cut_daily = _snapshot(truncate(synthetic, cutoff), synthetic, start, end)
        days = [d for d in full_hashes if d.date() <= cutoff_day]
        assert [cut_hashes[d] for d in days] == [full_hashes[d] for d in days]
        pd.testing.assert_frame_equal(cut_daily.iloc[:len(days)], full_daily.iloc[:len(days)])
        # Contrôle du test : sa sonde voit bien la différence dès le lendemain de D.
        after = [d for d in full_hashes if d.date() > cutoff_day]
        assert full_hashes[after[0]] != cut_hashes[after[0]]


def test_two_runs_on_same_journal_are_identical(synthetic):
    start, end = date(2024, 6, 1), date(2024, 6, 30)
    h1, d1 = _snapshot(synthetic, synthetic, start, end)
    h2, d2 = _snapshot(synthetic, synthetic, start, end)
    assert h1 == h2
    pd.testing.assert_frame_equal(d1, d2)


def test_open_amount_matches_reference_formula(synthetic):
    """current_amount_as_of = initial − Σ imputations antérieures (spec §5.2), à plusieurs dates."""
    from src.load.events import derive_imputed_amounts
    inv = synthetic.tables["invoice"]
    imp = derive_imputed_amounts(synthetic.tables["imputation"], inv)
    state = ledger(synthetic)
    for day in ("2024-02-15", "2024-06-01", "2024-11-30"):
        t = D(day)
        state.advance_to(t)
        paid = imp[imp["updated_at"] < t].groupby("invoice_id")["imputed_amount"].sum()
        expected = inv["initial_amount"] - paid.reindex(inv["invoice_id"]).fillna(0).to_numpy()
        created = (inv["creation_date"] < t).to_numpy()
        got = state.open_amount(inv["invoice_id"], t)
        assert np.array_equal(got.to_numpy()[created].astype(np.int64), expected.to_numpy()[created].astype(np.int64))
        assert got[~created].isna().all()
