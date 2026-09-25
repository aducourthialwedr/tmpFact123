"""Étape 5 — candidats et features des paires (paiement, facture) (brief §7.1-7.2, spec §4, §5.3).

Candidats (union des clés), pour chaque paiement du résiduel :
- K1 débiteur : factures ouvertes des débiteurs candidats de l'allocation, échéance dans la fenêtre ;
- K2 référence : factures ouvertes dont une clé figure dans le libellé, sans fenêtre ;
- K3 montant : factures ouvertes de restant dû égal au paiement, créées à ± 90 jours ;
- K4 client file : factures citées par le client file rattaché.
Filtres durs : même devise, facture ouverte à D (réservations du moteur déduites), contrat actif à D.

Toutes les features se calculent sur l'état à D ; les agrégats comportementaux sont ceux de
`LedgerState.debtor_stats` (fenêtre strictement antérieure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.allocation.allocator import SIGNALS, Allocation, Allocator
from src.allocation.indexes import Postings, _flatten
from src.settings import MLSettings
from src.timeline.state import LedgerState, _days

FEATURIZATION_VERSION = "1.1.0"
SRC_DEBTOR, SRC_REFERENCE, SRC_AMOUNT, SRC_CLIENT_FILE = 1, 2, 4, 8
ROUTES = ("DEBTOR_DIRECT", "ASSIGNOR", "TECHNICAL_ACCOUNT", "UNKNOWN")

FAMILIES: dict[str, list[str]] = {
    "amount": ["amount_diff", "amount_diff_rel", "amount_exact", "payment_covers", "amount_ratio",
               "is_typical_discount", "is_bank_fee_gap", "is_retention_gap", "ratio_to_initial", "balance_is_partial"],
    "temporal": ["days_to_due", "days_since_creation", "is_before_creation", "days_to_due_zscore"],
    "textual": ["ref_full_in_label", "ref_key_in_label", "internal_ref_in_label", "label_length",
                "label_has_no_alpha", "n_label_numbers"],
    "identity": ["iban_route", "iban_matches_invoice_debtor", "channel", "bankroll_code"],
    "behavioral": ["debtor_mean_payment_delay", "debtor_std_payment_delay", "debtor_partial_payment_rate",
                   "debtor_grouping_rate", "debtor_ref_citation_rate", "debtor_payment_count",
                   "debtor_open_invoice_count", "debtor_open_invoice_amount"],
    "contract": ["market", "product", "recourse"],
    "allocation": ["alloc_rank", "alloc_score", "alloc_is_firm_debtor", "debtor_in_allocation",
                   *[f"alloc_sig_{s}" for s in SIGNALS]],
    "client_file": ["has_client_file", "invoice_cited_in_client_file", "client_file_line_amount_diff",
                    "client_file_total_matches_payment"],
}
BASE = ["src_debtor", "src_reference", "src_amount", "src_client_file", "n_candidates", "amount_rank_in_payment"]
COMPETITION = ["rank_in_payment", "score_margin", "score_best_other"]
CATEGORICAL = ["iban_route", "channel", "bankroll_code", "market", "product", "recourse"]


def active_features(ml: MLSettings) -> list[str]:
    """Features du modèle selon les familles activées dans les paramètres."""
    fam = ml.features
    return BASE + [f for name, cols in FAMILIES.items() if getattr(fam, name) for f in cols]


def _codes(values: pd.Series, categories: list) -> np.ndarray:
    return pd.Categorical(pd.Series(values).astype(object), categories=categories).codes.astype(np.float32)


def _group_starts(keys: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.array([], dtype=np.int64)


# Paires (paiement, facture) examinées au plus par bloc avant plafonnement : les gros débiteurs, les
# références et montants fréquents produisent des milliers de paires par paiement.
CANDIDATE_PAIR_BUDGET = 2_000_000


def _row_blocks(load: np.ndarray, budget: int) -> list[tuple[int, int]]:
    """Découpe consécutive de lignes dont la charge cumulée reste de l'ordre de `budget`."""
    if len(load) == 0:
        return [(0, 0)]
    block = (np.cumsum(load) - load) // max(budget, 1)
    starts = np.flatnonzero(np.r_[True, block[1:] != block[:-1]])
    return list(zip(starts.tolist(), np.r_[starts[1:], len(load)].tolist()))


class Featurizer:
    """Candidats et features ; attributs statiques pré-calculés une fois."""

    def __init__(self, state: LedgerState, allocator: Allocator, ml: MLSettings, min_key_length: int,
                 categories: dict[str, list] | None = None):
        """`categories` : vocabulaires figés du modèle (inférence) ; à l'entraînement, construits sur les
        données disponibles puis sauvegardés avec le modèle."""
        self.state, self.allocator, self.ml = state, allocator, ml
        self.ref = allocator.ref
        self.min_key_length = min_key_length
        inv, pay = state.table("invoice"), state.table("payment")
        agr = state.table("agreement")
        self.inv_debtor = state.invoice_debtor_positions
        self.inv_due = _days(inv["due_date"])
        self.inv_created = _days(inv["creation_date"])
        self.inv_initial = inv["initial_amount"].fillna(0).to_numpy(dtype=np.int64)
        self.inv_agr = state.agr_pos(inv["agreement_id"])
        self.pay_amount = pay["amount"].fillna(-1).to_numpy(dtype=np.int64)
        self.pay_value = _days(pay["value_date"])
        currencies = sorted(set(inv["currency"].dropna()) | set(pay["currency"].dropna()))
        self.inv_ccy = _codes(inv["currency"], currencies)
        self.pay_ccy = _codes(pay["currency"], currencies)
        self.categories = categories or {
            "iban_route": list(ROUTES),
            "channel": sorted(pay["channel"].dropna().astype(str).unique()),
            "bankroll_code": sorted(pay["bankroll_code"].dropna().astype(str).unique()),
            "market": sorted(agr["market"].dropna().astype(str).unique()),
            "product": sorted(agr["product"].dropna().astype(str).unique()),
            "recourse": sorted(agr["recourse"].dropna().astype(str).unique()),
        }
        self.pay_channel = _codes(pay["channel"], self.categories["channel"])
        self.pay_bankroll = _codes(pay["bankroll_code"], self.categories["bankroll_code"])
        agr_idx = np.maximum(self.inv_agr, 0)
        has_agr = self.inv_agr >= 0
        self.inv_market = np.where(has_agr, _codes(agr["market"], self.categories["market"])[agr_idx], -1)
        self.inv_product = np.where(has_agr, _codes(agr["product"], self.categories["product"])[agr_idx], -1)
        self.inv_recourse = np.where(has_agr, _codes(agr["recourse"], self.categories["recourse"])[agr_idx], -1)
        labels = pay["label_norm"].fillna("").astype(str)
        self.pay_label_len = labels.str.len().to_numpy(dtype=np.float32)
        self.pay_no_alpha = (~labels.str.contains(r"[A-Z]", regex=True)).to_numpy(dtype=np.float32)
        # Nombre de clés du libellé lui-même (et non de celles présentes dans le vocabulaire des références,
        # qui dépend des factures futures).
        self.pay_n_numbers = pay["label_numbers"].map(len).to_numpy(dtype=np.float32)
        # Facture → clés de sa référence client / de sa référence interne ; clé complète.
        self.inv_client_keys = self._inv_keys(inv["client_reference_keys"])
        self.inv_internal_keys = self._inv_keys(inv["internal_reference_keys"])
        compact = inv["client_reference_norm"].fillna("").astype(str).str.replace(" ", "", regex=False)
        self.inv_full_key = self.ref.vocab.lookup(compact.astype(object).to_numpy())

    def _inv_keys(self, column: pd.Series) -> Postings:
        lengths, flat = _flatten(column)
        return Postings.from_lists(lengths, self.ref.vocab.lookup(flat))

    # --- Candidats ------------------------------------------------------------------------------------

    def candidates(self, rows: np.ndarray, pos: np.ndarray, alloc: Allocation, scope: pd.DataFrame, as_of,
                   claimed: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Candidats (row, inv, balance, src) des lignes `rows` du lot, et factures citées par client file.

        `pos` : positions des paiements de tout le lot ; `claimed` : réservations du moteur par facture.
        Les lignes sont traitées par blocs de charge bornée (`CANDIDATE_PAIR_BUDGET` paires avant
        plafonnement) : même résultat qu'en un seul passage, mémoire indépendante de la taille du lot.
        """
        rows = np.asarray(rows, dtype=np.int64)
        blocks = _row_blocks(self._pair_load(rows, pos, scope, as_of), CANDIDATE_PAIR_BUDGET)
        if len(blocks) == 1:
            return self._candidates_rows(rows, pos, alloc, scope, as_of, claimed)
        parts = [self._candidates_rows(rows[a:b], pos, alloc, scope, as_of, claimed) for a, b in blocks]
        c = pd.concat([p[0] for p in parts], ignore_index=True)
        c = c.iloc[np.argsort(c["row"].to_numpy(), kind="stable")].reset_index(drop=True)
        return c, pd.concat([p[1] for p in parts], ignore_index=True)

    def _pair_load(self, rows: np.ndarray, pos: np.ndarray, scope: pd.DataFrame, as_of) -> np.ndarray:
        """Majorant du nombre de paires examinées par ligne : factures ouvertes des débiteurs alloués,
        factures portant une clé du libellé, factures ouvertes de même montant."""
        cfg = self.ml.candidates
        load = np.zeros(len(rows), dtype=np.int64)
        index_of = pd.Series(np.arange(len(rows)), index=rows)
        if cfg.allocated_debtors and len(scope):
            sc = scope[scope["row"].isin(rows)]
            if "rank" in sc.columns:
                sc = sc[sc["rank"] <= cfg.max_debtors]
            if len(sc):
                debtors, d_of = np.unique(sc["debtor"].to_numpy(), return_inverse=True)
                owner, _, _ = self.state.debtor_open_invoices_at(debtors, as_of)
                n_open = np.bincount(owner, minlength=len(debtors))
                np.add.at(load, index_of.loc[sc["row"].to_numpy()].to_numpy(), n_open[d_of])
        if cfg.reference_no_window:
            r, keys = self.ref.payment_keys.gather(pos[rows])
            np.add.at(load, r, self.ref.key_invoices.lengths(keys))
        if cfg.amount_exact:
            _, balance = self.state.open_invoice_positions(as_of)
            balance = np.sort(balance)
            amount = self.pay_amount[pos[rows]]
            load += np.searchsorted(balance, amount, side="right") - np.searchsorted(balance, amount, side="left")
        return load

    def _candidates_rows(self, rows: np.ndarray, pos: np.ndarray, alloc: Allocation, scope: pd.DataFrame, as_of,
                         claimed: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.ml.candidates
        state = self.state
        rows = np.asarray(rows, dtype=np.int64)
        parts = []
        if cfg.allocated_debtors:
            sc = scope[scope["row"].isin(rows)]
            if "rank" in sc.columns:
                sc = sc[sc["rank"] <= cfg.max_debtors][["row", "debtor"]]
            if len(sc):
                debtors = np.unique(sc["debtor"].to_numpy())
                owner, inv, _ = state.debtor_open_invoices_at(debtors, as_of)
                k1 = sc.merge(pd.DataFrame({"debtor": debtors[owner], "inv": inv}), on="debtor")
                due = self.inv_due[k1["inv"].to_numpy()]
                value = self.pay_value[pos[k1["row"].to_numpy()]]
                ok = (due >= value - cfg.debtor_window_before_days) & (due <= value + cfg.debtor_window_after_days)
                parts.append(pd.DataFrame({"row": k1["row"].to_numpy()[ok], "inv": k1["inv"].to_numpy()[ok],
                                           "src": SRC_DEBTOR}))
        if cfg.reference_no_window:
            r, keys = self.ref.payment_keys.gather(pos[rows])
            keep = self.ref.vocab.lengths[keys] >= self.min_key_length
            r, keys = rows[r[keep]], keys[keep]
            if len(keys):
                uk, k_of = np.unique(keys, return_inverse=True)
                k_owner, inv = self.ref.key_invoices.gather(uk)
                pairs = pd.DataFrame({"row": r, "k": k_of}).drop_duplicates().merge(
                    pd.DataFrame({"k": k_owner, "inv": inv}), on="k")
                parts.append(pd.DataFrame({"row": pairs["row"].to_numpy(), "inv": pairs["inv"].to_numpy(),
                                           "src": SRC_REFERENCE}))
        if cfg.amount_exact:
            inv, balance = state.open_invoice_positions(as_of)
            opens = pd.DataFrame({"amount": balance - claimed[inv], "inv": inv})
            pays = pd.DataFrame({"row": rows, "amount": self.pay_amount[pos[rows]]})
            k3 = pays[pays["amount"] > 0].merge(opens, on="amount")
            gap = np.abs(self.inv_created[k3["inv"].to_numpy()] - self.pay_value[pos[k3["row"].to_numpy()]])
            k3 = k3[gap <= cfg.amount_window_days]
            parts.append(pd.DataFrame({"row": k3["row"].to_numpy(), "inv": k3["inv"].to_numpy(), "src": SRC_AMOUNT}))
        cited = self._client_file_invoices(rows, alloc, as_of)
        if cfg.client_file_cited and len(cited):
            parts.append(pd.DataFrame({"row": cited["row"].to_numpy(dtype=np.int64),
                                       "inv": cited["inv"].to_numpy(dtype=np.int64), "src": SRC_CLIENT_FILE}))
        parts = [p for p in parts if len(p)]
        if not parts:
            empty = pd.DataFrame({c: pd.Series(dtype="int64") for c in ("row", "inv", "balance", "src")})
            return empty, cited

        c = self._or_sources(pd.concat(parts, ignore_index=True))
        inv = c["inv"].to_numpy()
        balance = state.open_balance_at(inv, as_of) - claimed[inv]
        agr = self.inv_agr[inv]
        ok = (balance > 0) & (self.inv_ccy[inv] == self.pay_ccy[pos[c["row"].to_numpy()]])
        ok &= (agr < 0) | state.agreement_active_at(np.maximum(agr, 0), as_of)
        c = c[ok].assign(balance=balance[ok])
        return self._cap(c, pos, cfg.max_per_payment), cited

    def _cap(self, c: pd.DataFrame, pos: np.ndarray, cap: int) -> pd.DataFrame:
        """Plafond par paiement : clés précises (référence, montant, client file) d'abord ; puis, parmi les
        factures du débiteur, l'union des plus proches en montant et des échéances les plus proches de la
        date de valeur parmi celles qui peuvent entrer dans le paiement (paiements groupés)."""
        row = c["row"].to_numpy()
        amount = self.pay_amount[pos[row]]
        balance = c["balance"].to_numpy()
        strong = (c["src"].to_numpy() & ~SRC_DEBTOR) > 0
        gap = np.abs(balance - amount) / np.maximum(amount, 1)
        fits = balance <= amount + np.maximum(500, 0.03 * amount)
        due_gap = np.abs(self.pay_value[pos[row]] - self.inv_due[c["inv"].to_numpy()]).astype(np.float64)
        due_gap = np.where(fits, due_gap, np.inf)

        def rank_by(key: np.ndarray) -> np.ndarray:
            order = np.lexsort((c["inv"].to_numpy(), key, row))
            r = np.empty(len(row), dtype=np.int64)
            starts = np.flatnonzero(np.r_[True, row[order][1:] != row[order][:-1]]) if len(row) else np.array([], int)
            r[order] = np.arange(len(row)) - np.repeat(starts, np.diff(np.r_[starts, len(row)]))
            return r

        half = cap // 3
        keep = strong | (rank_by(gap) < half) | ((rank_by(due_gap) < cap - half) & fits)
        c = c[keep].assign(_order=np.where(strong, 0, 1)[keep], _gap=gap[keep])
        c = c.sort_values(["row", "_order", "_gap", "inv"], kind="mergesort")
        c = c[c.groupby("row").cumcount().to_numpy() < cap]
        return c.drop(columns=["_order", "_gap"]).reset_index(drop=True)

    @staticmethod
    def _or_sources(c: pd.DataFrame) -> pd.DataFrame:
        c = c.sort_values(["row", "inv"], kind="mergesort")
        row, inv, src = c["row"].to_numpy(), c["inv"].to_numpy(), c["src"].to_numpy(dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, (row[1:] != row[:-1]) | (inv[1:] != inv[:-1])])
        return pd.DataFrame({"row": row[starts], "inv": inv[starts], "src": np.bitwise_or.reduceat(src, starts)})

    def _client_file_invoices(self, rows: np.ndarray, alloc: Allocation, as_of) -> pd.DataFrame:
        """(row, inv, line_amount, inv_cited) des factures citées par le client file rattaché."""
        empty = pd.DataFrame({"row": pd.Series(dtype="int64"), "inv": pd.Series(dtype="int64"),
                              "line_amount": pd.Series(dtype="float64"), "inv_cited": pd.Series(dtype="bool")})
        files = alloc.payments["client_file_id"].to_numpy(dtype=object)[rows]
        has = pd.notna(files)
        if not has.any():
            return empty
        row_of_file = pd.Series(rows[has], index=files[has])
        lines = self.state.client_file_lines(files[has], as_of).reset_index(drop=True)
        if lines.empty:
            return empty
        lengths, flat = _flatten(lines["invoice_reference_keys"])
        line_of_key = np.repeat(np.arange(len(lines)), lengths)
        keys = self.ref.vocab.lookup(flat)
        ok = keys >= 0
        ok[ok] = self.ref.vocab.lengths[keys[ok]] >= self.min_key_length
        k_owner, inv = self.ref.key_invoices.gather(keys[ok])
        line = line_of_key[ok][k_owner]
        amount = pd.to_numeric(lines["amount"], errors="coerce").to_numpy(dtype=np.float64)
        out = pd.DataFrame({"row": row_of_file.reindex(lines["file_id"].to_numpy()[line]).to_numpy(),
                            "inv": inv, "line_amount": amount[line], "inv_cited": True})
        return out.drop_duplicates(["row", "inv"])

    # --- Features ---------------------------------------------------------------------------------------

    def features(self, c: pd.DataFrame, pos: np.ndarray, alloc: Allocation, batch_ids: np.ndarray, as_of,
                 cited: pd.DataFrame | None = None) -> pd.DataFrame:
        """Une ligne de features par candidat (même ordre que `c`)."""
        n = len(c)
        row, inv = c["row"].to_numpy(), c["inv"].to_numpy()
        p = pos[row]
        amount = self.pay_amount[p].astype(np.float64)
        balance = c["balance"].to_numpy(dtype=np.float64)
        diff = amount - balance
        gap_rel = (balance - amount) / np.maximum(balance, 1)
        f: dict[str, np.ndarray] = {}
        src = c["src"].to_numpy(dtype=np.int64)
        f["src_debtor"] = (src & SRC_DEBTOR) > 0
        f["src_reference"] = (src & SRC_REFERENCE) > 0
        f["src_amount"] = (src & SRC_AMOUNT) > 0
        f["src_client_file"] = (src & SRC_CLIENT_FILE) > 0
        f["n_candidates"] = np.bincount(row, minlength=len(pos))[row]
        order = np.lexsort((np.abs(diff), row))
        starts = _group_starts(row[order])
        rank = np.empty(n, dtype=np.float32)
        rank[order] = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n])) + 1
        f["amount_rank_in_payment"] = rank

        f["amount_diff"] = diff / 100.0
        f["amount_diff_rel"] = diff / np.maximum(balance, 1)
        f["amount_exact"] = diff == 0
        f["payment_covers"] = amount >= balance
        f["amount_ratio"] = amount / np.maximum(balance, 1)
        f["is_typical_discount"] = (gap_rel >= 0.005) & (gap_rel <= 0.03)
        f["is_bank_fee_gap"] = (balance - amount >= 500) & (balance - amount <= 4000)
        f["is_retention_gap"] = (gap_rel >= 0.04) & (gap_rel <= 0.06)
        f["ratio_to_initial"] = amount / np.maximum(self.inv_initial[inv], 1)
        f["balance_is_partial"] = balance < self.inv_initial[inv]

        value = self.pay_value[p]
        f["days_to_due"] = (value - self.inv_due[inv]).astype(np.float64)
        f["days_since_creation"] = (value - self.inv_created[inv]).astype(np.float64)
        f["is_before_creation"] = value < self.inv_created[inv]

        f.update(self._text(inv, p))
        f["label_length"] = self.pay_label_len[p]
        f["label_has_no_alpha"] = self.pay_no_alpha[p]
        f["n_label_numbers"] = self.pay_n_numbers[p]

        # Allocation : rang, score et signaux du débiteur de la facture pour ce paiement.
        deb = self.inv_debtor[inv]
        a = alloc.candidates
        debtor_pos = self.state.party_pos["debtor"]
        row_of = pd.Index(batch_ids)
        alloc_tab = pd.DataFrame({"row": row_of.get_indexer(a["payment_id"].astype(object).to_numpy()),
                                  "deb": debtor_pos(a["debtor_id"]), "rank": a["rank"].to_numpy(dtype=np.float64),
                                  "score": a["score"].to_numpy(dtype=np.float64), "signals": a["signals"].to_numpy()})
        joined = pd.DataFrame({"row": row, "deb": deb}).merge(alloc_tab, on=["row", "deb"], how="left")
        f["alloc_rank"] = joined["rank"].to_numpy(dtype=np.float64)
        f["alloc_score"] = joined["score"].fillna(0).to_numpy(dtype=np.float64)
        f["debtor_in_allocation"] = joined["rank"].notna().to_numpy()
        sig = joined["signals"].fillna("").astype(str)
        for s in SIGNALS:
            f[f"alloc_sig_{s}"] = sig.str.contains(s, regex=False).to_numpy()
        firm = alloc.payments["firm_debtor_id"].to_numpy(dtype=object)
        firm_pos = np.full(len(firm), -1, dtype=np.int64)
        has_firm = pd.notna(firm)
        if has_firm.any():
            firm_pos[has_firm] = debtor_pos(pd.Series(firm[has_firm], dtype=object))
        f["alloc_is_firm_debtor"] = firm_pos[row] == deb
        route = alloc.payments["iban_route"].to_numpy(dtype=object)
        f["iban_route"] = _codes(pd.Series(route[row]), self.categories["iban_route"])
        f["iban_matches_invoice_debtor"] = (route[row] == "DEBTOR_DIRECT") & f["alloc_sig_iban"]
        f["channel"] = self.pay_channel[p]
        f["bankroll_code"] = self.pay_bankroll[p]
        f["market"], f["product"], f["recourse"] = self.inv_market[inv], self.inv_product[inv], self.inv_recourse[inv]

        udeb = np.unique(deb[deb >= 0])
        stats = self.state.debtor_stats(self.state.table("debtor")["party_id"].to_numpy()[udeb], as_of)
        stats.index = udeb
        st = stats.reindex(deb)
        for col in ("mean_payment_delay", "std_payment_delay", "partial_payment_rate", "grouping_rate",
                    "ref_citation_rate", "payment_count", "open_invoice_count", "open_invoice_amount"):
            f[f"debtor_{col}"] = st[col].to_numpy(dtype=np.float64)
        f["debtor_open_invoice_amount"] = f["debtor_open_invoice_amount"] / 100.0
        f["days_to_due_zscore"] = (f["days_to_due"] - f["debtor_mean_payment_delay"]) / \
            np.maximum(np.nan_to_num(f["debtor_std_payment_delay"], nan=1.0), 1.0)

        files = alloc.payments["client_file_id"].to_numpy(dtype=object)
        f["has_client_file"] = pd.notna(files)[row]
        # Le rattachement du client file exige l'égalité du montant total et du paiement.
        f["client_file_total_matches_payment"] = f["has_client_file"]
        if cited is not None and len(cited):
            cj = pd.DataFrame({"row": row, "inv": inv}).merge(cited, on=["row", "inv"], how="left")
            f["invoice_cited_in_client_file"] = cj["inv_cited"].fillna(False).to_numpy(dtype=bool)
            f["client_file_line_amount_diff"] = (cj["line_amount"].to_numpy(dtype=np.float64) - balance) / 100.0
        else:
            f["invoice_cited_in_client_file"] = np.zeros(n, dtype=bool)
            f["client_file_line_amount_diff"] = np.full(n, np.nan)
        out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
        for col in CATEGORICAL:
            out.loc[out[col] < 0, col] = np.nan          # catégorie inconnue
        return out

    def _text(self, inv: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
        """La référence (une clé quelconque, la clé complète, la référence interne) figure-t-elle au libellé ?"""
        n = len(inv)
        up, u_of = np.unique(p, return_inverse=True)
        lo, lk = self.ref.payment_keys.gather(up)
        label = pd.DataFrame({"u": lo, "k": lk}).drop_duplicates()

        def hits(i: np.ndarray, k: np.ndarray) -> np.ndarray:
            found = pd.DataFrame({"i": i, "u": u_of[i], "k": k}).merge(label, on=["u", "k"])["i"].unique()
            v = np.zeros(n, dtype=bool)
            v[found] = True
            return v

        out = {}
        for name, postings in (("ref_key_in_label", self.inv_client_keys),
                               ("internal_ref_in_label", self.inv_internal_keys)):
            out[name] = hits(*postings.gather(inv))
        full = self.inv_full_key[inv]
        idx = np.flatnonzero(full >= 0)
        out["ref_full_in_label"] = hits(idx, full[idx])
        return out
