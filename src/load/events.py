"""Journal d'événements ordonné (brief §3.4, spec §8.1).

Le journal est un DataFrame léger : chaque ligne référence une entité par son
identifiant, le détail reste dans les tables chargées. Colonnes :

- `seq`         : position dans le journal (0..n-1)
- `ts`          : horodatage UTC naïf
- `event_type`  : voir `EventType`
- `entity_id`   : payment_id, invoice_id, party_id, agreement_id ou file_id
- `related_id`  : invoice_id pour une imputation, rôle pour une partie, sinon ""
- `amount`      : montant imputé (centimes) pour IMPUTATION_APPLIED, sinon NA

Ordre : `ts`, puis rang du type d'événement, puis `entity_id`, puis `related_id`.
Le rang place, à horodatage égal, ce qui ouvre avant ce qui l'utilise, et ce qui
ferme en dernier.
"""

from __future__ import annotations

import hashlib
from enum import Enum

import pandas as pd

from src.arrow_ops import lookup
from src.load.loader import DATETIME_DTYPE, Issue, LoadedData


class EventType(str, Enum):
    PARTY_OPENED = "PARTY_OPENED"
    AGREEMENT_CREATED = "AGREEMENT_CREATED"
    INVOICE_CREATED = "INVOICE_CREATED"
    CLIENT_FILE_RECEIVED = "CLIENT_FILE_RECEIVED"
    PAYMENT_RECEIVED = "PAYMENT_RECEIVED"
    IMPUTATION_APPLIED = "IMPUTATION_APPLIED"
    AGREEMENT_DISABLED = "AGREEMENT_DISABLED"
    PARTY_CLOSED = "PARTY_CLOSED"


EVENT_RANK = {e.value: i for i, e in enumerate(EventType)}

JOURNAL_COLUMNS = ["seq", "ts", "event_type", "entity_id", "related_id", "amount"]


def derive_imputed_amounts(imputation: pd.DataFrame, invoice: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `balance_before` et `imputed_amount` à chaque ligne d'imputation.

    `residual_amount` est le solde de la facture après la ligne (brief §3.5) :
    montant imputé = solde avant − solde après, le solde avant étant le résidu
    de la ligne précédente sur la même facture, ou `initial_amount` pour la
    première. Chaque ligne n'utilise que les lignes qui la précèdent.

    Ordre de chaînage par facture : `updated_at`, puis résidu décroissant (le
    solde ne fait que baisser dans le cas nominal), puis `payment_id`.
    """
    imp = imputation.copy()
    imp["_neg_residual"] = -imp["residual_amount"].astype("Float64")
    imp = imp.sort_values(
        ["invoice_id", "updated_at", "_neg_residual", "payment_id"], kind="mergesort"
    )
    prev = imp.groupby("invoice_id", sort=False)["residual_amount"].shift(1)
    first = lookup(imp["invoice_id"], invoice["invoice_id"], invoice["initial_amount"]).astype("Int64")
    imp["balance_before"] = prev.astype("Int64").fillna(first)
    imp["imputed_amount"] = imp["balance_before"] - imp["residual_amount"]
    return imp.drop(columns="_neg_residual").sort_index()


def _events(ts: pd.Series, event_type: EventType, entity_id: pd.Series,
            related_id: pd.Series | str = "", amount: pd.Series | None = None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "ts": ts.astype(DATETIME_DTYPE),
            "event_type": event_type.value,
            "entity_id": entity_id.astype("string"),
            "related_id": related_id if isinstance(related_id, str)
            else related_id.astype("string").fillna(""),
            "amount": amount.astype("Int64") if amount is not None
            else pd.Series(pd.NA, index=ts.index, dtype="Int64"),
        }
    )
    return df[df["ts"].notna()]


def payment_event_time(payment: pd.DataFrame) -> pd.Series:
    """Date de connaissance du paiement : `booking_date` si connue, sinon `value_date`."""
    return payment["booking_date"].fillna(payment["value_date"])


def build_journal(data: LoadedData) -> tuple[pd.DataFrame, list[Issue]]:
    t = data.tables
    issues: list[Issue] = []
    parts: list[pd.DataFrame] = []

    for role in ("assignor", "debtor"):
        party = t[role]
        role_tag = role.upper()
        parts.append(_events(party["opened_at"], EventType.PARTY_OPENED, party["party_id"], role_tag))
        parts.append(_events(party["closed_at"], EventType.PARTY_CLOSED, party["party_id"], role_tag))
        n_no_open = int(party["opened_at"].isna().sum())
        if n_no_open:
            issues.append(Issue(role, "opened_at", "sans_evenement_ouverture", n_no_open,
                                "partie considérée active depuis toujours"))

    agr = t["agreement"]
    parts.append(_events(agr["created_at"], EventType.AGREEMENT_CREATED, agr["agreement_id"]))
    parts.append(_events(agr["disabled_at"], EventType.AGREEMENT_DISABLED, agr["agreement_id"]))

    inv = t["invoice"]
    parts.append(_events(inv["creation_date"], EventType.INVOICE_CREATED, inv["invoice_id"]))

    if "client_file" in t:
        cf = t["client_file"]
        parts.append(_events(cf["received_at"], EventType.CLIENT_FILE_RECEIVED, cf["file_id"]))

    pay = t["payment"]
    parts.append(_events(payment_event_time(pay), EventType.PAYMENT_RECEIVED, pay["payment_id"]))
    if "booking_date" in data.mapped_fields["payment"]:
        n_fallback = int(pay["booking_date"].isna().sum())
        if n_fallback:
            issues.append(Issue("payment", "booking_date", "repli_sur_value_date", n_fallback))

    imp = derive_imputed_amounts(t["imputation"], inv)
    parts.append(_events(imp["updated_at"], EventType.IMPUTATION_APPLIED, imp["payment_id"],
                         imp["invoice_id"], imp["imputed_amount"]))

    journal = pd.concat(parts, ignore_index=True)
    journal["_rank"] = journal["event_type"].map(EVENT_RANK)
    journal = journal.sort_values(
        ["ts", "_rank", "entity_id", "related_id"], kind="mergesort"
    ).drop(columns="_rank").reset_index(drop=True)
    journal.insert(0, "seq", range(len(journal)))
    return journal[JOURNAL_COLUMNS], issues


def journal_hash(journal: pd.DataFrame) -> str:
    """Empreinte SHA-256 du journal, stable entre exécutions et versions de pandas.

    Calculée colonne par colonne sur une représentation binaire explicite
    (horodatages en µs int64, chaînes UTF-8 séparées par le caractère US (0x1F), montants int64
    avec sentinelle pour NA), plutôt que sur une sérialisation CSV complète.
    """
    h = hashlib.sha256()
    h.update(f"{len(journal)}|{','.join(JOURNAL_COLUMNS)}".encode())
    h.update(journal["seq"].to_numpy(dtype="<i8").tobytes())
    h.update(journal["ts"].astype(DATETIME_DTYPE).to_numpy().view("<i8").tobytes())
    for col in ("event_type", "entity_id", "related_id"):
        h.update("\x1f".join(journal[col].astype("str").fillna("").tolist()).encode("utf-8"))
    h.update(journal["amount"].astype("Int64").fillna(-(2**63)).to_numpy(dtype="<i8").tobytes())
    return h.hexdigest()
