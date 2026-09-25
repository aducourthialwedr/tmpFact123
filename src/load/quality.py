"""Contrôles de qualité des données chargées (brief §3.5).

Rien n'est corrigé ni supprimé ici : chaque anomalie est comptée et remontée
dans le rapport de chargement pour décision.
"""

from __future__ import annotations

import pandas as pd

from src.arrow_ops import isin, lookup
from src.load.canonical import IMPUTATION_STATUSES, TABLES
from src.load.events import derive_imputed_amounts, payment_event_time
from src.load.loader import Issue, LoadedData


def _count(mask: pd.Series) -> int:
    return int(mask.fillna(False).sum())


def _orphans(child: pd.Series, parent: pd.Series) -> int:
    return int((~isin(child.dropna(), parent)).sum())


def check_quality(data: LoadedData, imputation_derived: pd.DataFrame | None = None) -> list[Issue]:
    """`imputation_derived` : sortie de `derive_imputed_amounts`, recalculée si absente."""
    t = data.tables
    issues: list[Issue] = []

    def add(table: str, fld: str | None, kind: str, n: int, detail: str = "") -> None:
        if n:
            issues.append(Issue(table, fld, kind, n, detail))

    # Unicité des clés primaires.
    for name, df in t.items():
        pk = list(TABLES[name].primary_key) if name in TABLES else []
        if pk:
            add(name, ",".join(pk), "cle_dupliquee", _count(df.duplicated(pk, keep=False)))

    # Intégrité référentielle.
    inv, pay, imp, agr = t["invoice"], t["payment"], t["imputation"], t["agreement"]
    add("invoice", "debtor_id", "orphelin", _orphans(inv["debtor_id"], t["debtor"]["party_id"]))
    add("invoice", "agreement_id", "orphelin", _orphans(inv["agreement_id"], agr["agreement_id"]))
    add("agreement", "debtor_id", "orphelin", _orphans(agr["debtor_id"], t["debtor"]["party_id"]))
    add("agreement", "client_id", "orphelin", _orphans(agr["client_id"], t["assignor"]["party_id"]))
    add("imputation", "payment_id", "orphelin", _orphans(imp["payment_id"], pay["payment_id"]))
    add("imputation", "invoice_id", "orphelin", _orphans(imp["invoice_id"], inv["invoice_id"]))
    if "client_file_line" in t and "client_file" in t:
        add("client_file_line", "file_id", "orphelin",
            _orphans(t["client_file_line"]["file_id"], t["client_file"]["file_id"]))

    # Cohérence de la facture.
    agr_debtor = lookup(inv["agreement_id"], agr["agreement_id"], agr["debtor_id"])
    add("invoice", "debtor_id", "debiteur_different_du_contrat",
        _count(agr_debtor.notna() & (inv["debtor_id"] != agr_debtor)))
    add("invoice", "due_date", "echeance_avant_creation", _count(inv["due_date"] < inv["creation_date"]))
    add("invoice", "initial_amount", "montant_non_positif", _count(inv["initial_amount"] <= 0))

    # Paiements hors périmètre v1.
    add("payment", "amount", "montant_negatif_hors_perimetre", _count(pay["amount"] < 0))
    add("payment", "amount", "montant_nul", _count(pay["amount"] == 0))
    main_ccy = pay["currency"].mode()
    if len(main_ccy):
        add("payment", "currency", "devise_minoritaire", _count(pay["currency"] != main_ccy.iloc[0]),
            f"devise principale {main_ccy.iloc[0]}")

    # Imputations.
    add("imputation", "status", "statut_inconnu",
        _count(~imp["status"].isin(IMPUTATION_STATUSES) & imp["status"].notna()),
        "renseigner value_maps.status dans config/schema.yaml")
    add("imputation", "residual_amount", "full_avec_residu_non_nul",
        _count((imp["status"] == "FULL") & (imp["residual_amount"] != 0)))
    add("imputation", "residual_amount", "partial_avec_residu_nul",
        _count((imp["status"] == "PARTIAL") & (imp["residual_amount"] == 0)))
    add("imputation", "payment_id,invoice_id", "lignes_multiples_meme_paire",
        _count(imp.duplicated(["payment_id", "invoice_id"], keep=False)))

    derived = imputation_derived if imputation_derived is not None else derive_imputed_amounts(imp, inv)
    add("imputation", "imputed_amount", "montant_impute_negatif", _count(derived["imputed_amount"] < 0),
        "le solde de la facture remonte : vérifier l'interprétation de residual_amount")
    add("imputation", "imputed_amount", "montant_impute_nul", _count(derived["imputed_amount"] == 0))

    imp_value = lookup(imp["payment_id"], pay["payment_id"], pay["value_date"])
    imp_known = lookup(imp["payment_id"], pay["payment_id"], payment_event_time(pay))
    add("imputation", "updated_at", "avant_date_de_valeur_du_paiement",
        _count(imp["updated_at"] < imp_value))
    add("imputation", "updated_at", "avant_connaissance_du_paiement",
        _count(imp["updated_at"] < imp_known),
        "imputation antérieure à l'événement PAYMENT_RECEIVED")
    inv_created = lookup(imp["invoice_id"], inv["invoice_id"], inv["creation_date"])
    add("imputation", "updated_at", "avant_creation_facture", _count(imp["updated_at"] < inv_created))

    # Audit : le solde final reconstruit doit égaler invoice.current_amount.
    if "invoice_current_amount" in data.audit:
        imputed = derived.groupby("invoice_id", sort=False)["imputed_amount"].sum()
        imputed_total = lookup(inv["invoice_id"], pd.Series(imputed.index), imputed.reset_index(drop=True))
        rebuilt = inv["initial_amount"] - imputed_total.fillna(0).astype("Int64")
        audit_df = data.audit["invoice_current_amount"]
        final = lookup(inv["invoice_id"], audit_df["invoice_id"], audit_df["current_amount"])
        add("invoice", "current_amount", "solde_reconstruit_different",
            _count(final.notna() & (rebuilt != final)),
            "initial_amount − Σ imputations ≠ current_amount final")

    # Parties.
    for role in ("assignor", "debtor"):
        party = t[role]
        add(role, "closed_at", "fermeture_avant_ouverture", _count(party["closed_at"] < party["opened_at"]))
    add("agreement", "disabled_at", "desactivation_avant_creation",
        _count(agr["disabled_at"] < agr["created_at"]))

    return issues
