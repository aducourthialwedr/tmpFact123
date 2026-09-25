"""Modèle de données canonique du POC.

Ce sont les noms *internes* (ceux du brief et de la spec §2). Les noms réels des
tables et colonnes sont fournis par `config/schema.yaml`, jamais déduits ici.

`required=True` sur un champ : la pipeline ne peut pas fonctionner sans, le
mapping doit être renseigné. Un champ optionnel non mappé est ajouté vide et
signalé dans le rapport de chargement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FieldType(str, Enum):
    ID = "id"                # identifiant, stocké en chaîne
    TEXT = "text"
    DATE = "date"            # date sans heure (minuit)
    TIMESTAMP = "timestamp"  # horodatage, converti en UTC naïf
    AMOUNT = "amount"        # entier de centimes


@dataclass(frozen=True)
class Field:
    name: str
    type: FieldType
    required: bool = True


@dataclass(frozen=True)
class Table:
    name: str
    fields: tuple[Field, ...]
    required: bool = True
    primary_key: tuple[str, ...] = ()

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"{self.name}.{name}")


T = FieldType

PAYMENT = Table(
    "payment",
    (
        Field("payment_id", T.ID),
        Field("value_date", T.DATE),
        # Date de comptabilisation / de connaissance, si elle existe (§3.5).
        # Si mappée, c'est elle qui ordonne le journal.
        Field("booking_date", T.DATE, required=False),
        Field("amount", T.AMOUNT),
        Field("currency", T.TEXT),
        Field("iban_debtor", T.TEXT, required=False),
        Field("iban_creditor", T.TEXT, required=False),
        Field("label", T.TEXT),
        Field("channel", T.TEXT, required=False),
        Field("payment_type", T.TEXT, required=False),
        Field("bankroll_code", T.TEXT, required=False),
    ),
    primary_key=("payment_id",),
)

INVOICE = Table(
    "invoice",
    (
        Field("invoice_id", T.ID),
        Field("client_reference", T.TEXT),
        Field("internal_reference", T.TEXT, required=False),
        Field("creation_date", T.DATE),
        Field("due_date", T.DATE),
        Field("initial_amount", T.AMOUNT),
        # Restant dû *final* : fuite directe. Chargé uniquement pour l'audit de
        # cohérence des imputations, retiré de la table `invoice` au chargement.
        Field("current_amount", T.AMOUNT, required=False),
        Field("currency", T.TEXT),
        Field("debtor_id", T.ID),
        Field("agreement_id", T.ID),
    ),
    primary_key=("invoice_id",),
)

IMPUTATION = Table(
    "imputation",
    (
        Field("payment_id", T.ID),
        Field("invoice_id", T.ID),
        Field("status", T.TEXT),              # FULL / PARTIAL après value_maps
        Field("updated_at", T.TIMESTAMP),
        Field("residual_amount", T.AMOUNT),   # solde de la facture APRÈS la ligne
    ),
)


def _party(name: str) -> Table:
    return Table(
        name,
        (
            Field("party_id", T.ID),
            Field("bankroll_code", T.TEXT, required=False),
            Field("iban", T.TEXT, required=False),
            Field("name", T.TEXT),
            Field("opened_at", T.DATE, required=False),
            Field("closed_at", T.DATE, required=False),
        ),
        primary_key=("party_id",),
    )


ASSIGNOR = _party("assignor")
DEBTOR = _party("debtor")

AGREEMENT = Table(
    "agreement",
    (
        Field("agreement_id", T.ID),
        Field("debtor_id", T.ID),
        Field("client_id", T.ID),
        Field("contract_number", T.TEXT, required=False),
        Field("created_at", T.DATE),
        Field("disabled_at", T.DATE, required=False),
        Field("market", T.TEXT, required=False),
        Field("product", T.TEXT, required=False),
        Field("recourse", T.TEXT, required=False),
    ),
    primary_key=("agreement_id",),
)

# Référentiel des IBAN de comptes techniques (§3.5) — optionnel.
TECHNICAL_ACCOUNT = Table(
    "technical_account",
    (
        Field("iban", T.TEXT),
        Field("bankroll_code", T.TEXT, required=False),
        Field("description", T.TEXT, required=False),
    ),
    required=False,
    primary_key=("iban",),
)

# Client files (§3.2) — format réel à confirmer. Seule la voie tabulaire
# (fichiers déjà structurés en lignes) est branchée ici.
CLIENT_FILE = Table(
    "client_file",
    (
        Field("file_id", T.ID),
        Field("received_at", T.TIMESTAMP),
        Field("source_format", T.TEXT, required=False),
        Field("payment_reference", T.TEXT, required=False),
        Field("total_amount", T.AMOUNT, required=False),
        Field("payment_date", T.DATE, required=False),
        Field("iban", T.TEXT, required=False),
        Field("issuer_name", T.TEXT, required=False),
    ),
    required=False,
    primary_key=("file_id",),
)

CLIENT_FILE_LINE = Table(
    "client_file_line",
    (
        Field("file_id", T.ID),
        Field("line_no", T.ID, required=False),
        Field("invoice_reference", T.TEXT),
        Field("amount", T.AMOUNT, required=False),
        Field("gap_reason", T.TEXT, required=False),
    ),
    required=False,
)

TABLES: dict[str, Table] = {
    t.name: t
    for t in (
        PAYMENT, INVOICE, IMPUTATION, ASSIGNOR, DEBTOR, AGREEMENT,
        TECHNICAL_ACCOUNT, CLIENT_FILE, CLIENT_FILE_LINE,
    )
}

IMPUTATION_STATUSES = ("FULL", "PARTIAL")
