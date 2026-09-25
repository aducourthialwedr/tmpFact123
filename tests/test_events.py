import pandas as pd

from src.load.events import EventType, build_journal, derive_imputed_amounts, journal_hash
from src.load.loader import load_all
from src.load.quality import check_quality
from tests.conftest import make_data

DEBTOR = [{"party_id": "D1", "name": "DUPONT", "opened_at": "2024-01-01"}]
ASSIGNOR = [{"party_id": "A1", "name": "CEDANT", "opened_at": "2023-01-01"}]
AGREEMENT = [{"agreement_id": "AG1", "debtor_id": "D1", "client_id": "A1", "created_at": "2024-01-01"}]


def invoice(iid, created="2024-02-01", amount=1000):
    return {"invoice_id": iid, "client_reference": f"REF{iid}", "creation_date": created,
            "due_date": "2024-03-01", "initial_amount": amount, "currency": "EUR",
            "debtor_id": "D1", "agreement_id": "AG1"}


def payment(pid, value_date="2024-03-05", amount=1000, booking_date=None):
    return {"payment_id": pid, "value_date": value_date, "booking_date": booking_date,
            "amount": amount, "currency": "EUR", "label": "VIR"}


def imputation(pid, iid, updated_at, residual, status=None):
    return {"payment_id": pid, "invoice_id": iid, "updated_at": updated_at,
            "residual_amount": residual, "status": status or ("FULL" if residual == 0 else "PARTIAL")}


def data_with(invoices, payments, imputations, **extra):
    return make_data(debtor=DEBTOR, assignor=ASSIGNOR, agreement=AGREEMENT,
                     invoice=invoices, payment=payments, imputation=imputations, **extra)


# --- Montant imputé dérivé de residual_amount -----------------------------------

def test_imputed_amount_from_residual_chain():
    d = data_with(
        [invoice("I1", amount=1000)],
        [payment("P1", amount=400), payment("P2", amount=600)],
        [imputation("P2", "I1", "2024-03-10 10:00", 0), imputation("P1", "I1", "2024-03-06 10:00", 600)],
    )
    out = derive_imputed_amounts(d.tables["imputation"], d.tables["invoice"])
    assert out.set_index("payment_id")["imputed_amount"].to_dict() == {"P2": 600, "P1": 400}
    assert out.set_index("payment_id")["balance_before"].to_dict() == {"P2": 600, "P1": 1000}


def test_imputed_amount_same_timestamp_ordered_by_residual():
    # Même horodatage : l'ordre de chaînage suit le résidu décroissant, pas l'identifiant.
    d = data_with(
        [invoice("I1", amount=1000)],
        [payment("PA"), payment("PZ")],
        [imputation("PA", "I1", "2024-03-06 10:00", 0), imputation("PZ", "I1", "2024-03-06 10:00", 700)],
    )
    out = derive_imputed_amounts(d.tables["imputation"], d.tables["invoice"])
    assert out.set_index("payment_id")["imputed_amount"].to_dict() == {"PA": 700, "PZ": 300}


# --- Ordonnancement -------------------------------------------------------------

def test_tie_break_by_event_type_then_id():
    d = data_with(
        [invoice("I2", created="2024-03-05"), invoice("I1", created="2024-03-05")],
        [payment("P2"), payment("P1")],
        [],
    )
    journal, _ = build_journal(d)
    day = journal[journal["ts"] == pd.Timestamp("2024-03-05")]
    assert list(zip(day["event_type"], day["entity_id"])) == [
        ("INVOICE_CREATED", "I1"), ("INVOICE_CREATED", "I2"),
        ("PAYMENT_RECEIVED", "P1"), ("PAYMENT_RECEIVED", "P2"),
    ]
    assert list(journal["seq"]) == list(range(len(journal)))


def test_closing_events_come_last_on_same_day():
    debtor = [{"party_id": "D1", "name": "DUPONT", "opened_at": "2024-01-01", "closed_at": "2024-03-05"}]
    d = make_data(debtor=debtor, assignor=ASSIGNOR, agreement=AGREEMENT,
                  invoice=[], payment=[payment("P1")], imputation=[])
    journal, _ = build_journal(d)
    day = journal[journal["ts"] == pd.Timestamp("2024-03-05")]
    assert list(day["event_type"]) == ["PAYMENT_RECEIVED", "PARTY_CLOSED"]


def test_booking_date_orders_payment_events():
    d = data_with([invoice("I1")], [payment("P1", value_date="2024-03-05", booking_date="2024-03-07")], [])
    journal, _ = build_journal(d)
    ev = journal[journal["event_type"] == EventType.PAYMENT_RECEIVED.value]
    assert ev["ts"].iloc[0] == pd.Timestamp("2024-03-07")


def test_imputation_event_carries_amount():
    d = data_with([invoice("I1", amount=1000)], [payment("P1", amount=1000)],
                  [imputation("P1", "I1", "2024-03-06 09:30", 0)])
    journal, _ = build_journal(d)
    ev = journal[journal["event_type"] == "IMPUTATION_APPLIED"].iloc[0]
    assert (ev["entity_id"], ev["related_id"], ev["amount"]) == ("P1", "I1", 1000)


# --- Reproductibilité -----------------------------------------------------------

def test_journal_is_deterministic_and_independent_of_row_order(synthetic_dir, synthetic_schema):
    data = load_all(synthetic_schema, synthetic_dir)
    j1, _ = build_journal(data)
    j2, _ = build_journal(load_all(synthetic_schema, synthetic_dir))
    assert journal_hash(j1) == journal_hash(j2)

    shuffled = {name: df.sample(frac=1, random_state=3).reset_index(drop=True)
                for name, df in data.tables.items()}
    data.tables = shuffled
    j3, _ = build_journal(data)
    assert journal_hash(j3) == journal_hash(j1)


def test_synthetic_journal_is_consistent(synthetic_dir, synthetic_schema):
    data = load_all(synthetic_schema, synthetic_dir)
    journal, _ = build_journal(data)
    counts = journal["event_type"].value_counts()
    assert counts["PAYMENT_RECEIVED"] == len(data.tables["payment"])
    assert counts["INVOICE_CREATED"] == len(data.tables["invoice"])
    assert counts["IMPUTATION_APPLIED"] == len(data.tables["imputation"])
    assert journal["ts"].is_monotonic_increasing
    # Le jeu synthétique est propre : aucune anomalie, audit des soldes compris.
    assert check_quality(data) == []


# --- Contrôles qualité ------------------------------------------------------------

def test_quality_detects_anomalies():
    d = data_with(
        [invoice("I1", amount=1000), invoice("I2", amount=500)],
        [payment("P1", value_date="2024-03-05", amount=1000), payment("P2", amount=500)],
        [
            imputation("P1", "I1", "2024-03-04 10:00", 0),              # avant la date de valeur
            imputation("P2", "I2", "2024-03-06 10:00", 100, "FULL"),    # FULL avec résidu
            imputation("P2", "I9", "2024-03-06 10:00", 0),              # facture inconnue
        ],
    )
    kinds = {(i.table, i.kind) for i in check_quality(d)}
    assert ("imputation", "avant_date_de_valeur_du_paiement") in kinds
    assert ("imputation", "full_avec_residu_non_nul") in kinds
    assert ("imputation", "orphelin") in kinds


def test_quality_audit_detects_balance_mismatch():
    d = data_with([invoice("I1", amount=1000)], [payment("P1", amount=400)],
                  [imputation("P1", "I1", "2024-03-06 10:00", 600)])
    d.audit["invoice_current_amount"] = pd.DataFrame(
        {"invoice_id": pd.array(["I1"], dtype="string"), "current_amount": pd.array([500], dtype="Int64")})
    kinds = {i.kind for i in check_quality(d)}
    assert "solde_reconstruit_different" in kinds

    d.audit["invoice_current_amount"]["current_amount"] = pd.array([600], dtype="Int64")
    assert "solde_reconstruit_different" not in {i.kind for i in check_quality(d)}
