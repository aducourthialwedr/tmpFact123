"""Allocation des paiements aux débiteurs (brief §5).

Pour chaque paiement du lot, une liste classée de débiteurs candidats avec leur
score et les signaux qui les ont produits, et le client file rattaché.

Signaux, du plus fort au plus faible :
1. client file rattaché sans ambiguïté → débiteurs des factures qu'il cite ;
2. référence de facture trouvée dans le libellé → débiteur de la facture ;
3. IBAN, routé DEBTOR_DIRECT / ASSIGNOR / TECHNICAL_ACCOUNT / UNKNOWN ;
4. nom du débiteur retrouvé dans le libellé (index inversé, mots rares pondérés) ;
5. montant : facture ouverte à D de restant dû exactement égal au paiement
   (extension du brief, clé K3 de la spec ; désactivable).

Les scores des signaux se combinent en « ou » probabiliste : 1 − Π(1 − sᵢ).
L'allocation est **ferme** si un seul débiteur porte un signal fort (client
file univoque, référence désignant un seul débiteur, IBAN direct unique).

L'allocateur garde un état propre au moteur : les client files déjà rattachés.
Il se crée au début d'un rejeu et ne se partage pas entre deux rejeux.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.allocation.indexes import IbanIndex, NameIndex, Postings, ReferenceIndex, _flatten
from src import memory
from src.settings import AllocationSettings
from src.timeline.loop import DayContext
from src.timeline.state import DAY_US, LedgerState, _days

CLIENT_FILE, REFERENCE, IBAN, NAME, AMOUNT = "client_file", "reference", "iban", "name", "amount"
SIGNALS = (CLIENT_FILE, REFERENCE, IBAN, NAME, AMOUNT)
DEBTOR_DIRECT, ASSIGNOR, TECHNICAL_ACCOUNT, UNKNOWN = "DEBTOR_DIRECT", "ASSIGNOR", "TECHNICAL_ACCOUNT", "UNKNOWN"
FIRM, MULTIPLE, NONE = "ferme", "multiple", "aucun"

# Poids des signaux (score maximal qu'un signal seul peut donner). Valeurs initiales, à calibrer.
WEIGHTS = {CLIENT_FILE: 1.0, REFERENCE: 0.95, IBAN: 0.9, NAME: 0.7, AMOUNT: 0.5}

# Taille des blocs de paiements traités ensemble : borne la mémoire des jointures d'une journée.
CHUNK_ROWS = 20_000

_SIGNAL_COLUMNS = ["row", "debtor", "signal", "score", "strong"]
_SIGNAL_CODE = {s: i for i, s in enumerate(SIGNALS)}
_SIGNAL_BIT = {s: 1 << i for i, s in enumerate(SIGNALS)}
_SIGNAL_LABEL = {m: "+".join(s for s in SIGNALS if m & _SIGNAL_BIT[s]) for m in range(1 << len(SIGNALS))}


@dataclass
class Allocation:
    candidates: pd.DataFrame   # payment_id, rank, debtor_id, score, signal, signals, strong
    payments: pd.DataFrame     # payment_id, status, firm_debtor_id, n_candidates, iban_route, client_file_id


def _empty_signal() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                         zip(_SIGNAL_COLUMNS, ["int64", "int64", "object", "float64", "bool"])})


class Allocator:
    def __init__(self, state: LedgerState, settings: AllocationSettings):
        self.state = state
        self.cfg = settings
        sig = settings.signals
        inv, pay = state.table("invoice"), state.table("payment")
        deb, asg = state.table("debtor"), state.table("assignor")
        self._debtor_ids = deb["party_id"].astype(object).to_numpy()
        self._inv_debtor = state.invoice_debtor_positions
        self._pay_value_day = _days(pay["value_date"])
        self._pay_amount = pay["amount"].fillna(-1).to_numpy(dtype=np.int64)
        self._pay_label = pay["label_norm"].fillna("").astype(object).to_numpy()

        self.ref = ReferenceIndex(inv, pay) if (sig.reference.enabled or sig.client_file.enabled) else None
        self.names = NameIndex(deb, pay, sig.name.min_token_length) if sig.name.enabled else None
        self.iban = None
        if sig.iban.enabled or sig.client_file.enabled:
            self.iban = IbanIndex(deb, asg, state.table("technical_account"), state.table("party_iban"))
            self._pay_iban = self.iban.vocab.lookup(pay["iban_debtor"].astype(object).to_numpy())
            self._pay_bankroll = pay["bankroll_code"].astype(object).to_numpy()

        self._cf = state.table("client_file")
        if sig.client_file.enabled and self._cf is not None and self.ref is not None:
            cf = self._cf
            self._cf_ids = cf["file_id"].astype(object).to_numpy()
            self._cf_amount = cf["total_amount"].fillna(-1).to_numpy(dtype=np.int64)
            self._cf_day = _days(cf["payment_date"])
            self._cf_iban = self.iban.vocab.lookup(cf["iban"].astype(object).to_numpy()) if self.iban else None
            self._cf_ref = cf["payment_reference_norm"].fillna("").astype(object).to_numpy()
            lines = state.table("client_file_line")
            file_of_line = pd.Index(cf["file_id"].astype(object)).get_indexer(
                lines["file_id"].astype(object)).astype(np.int64)
            lengths, flat = _flatten(lines["invoice_reference_keys"])
            owners = np.repeat(file_of_line, lengths)
            self._file_keys = Postings.build(owners, self.ref.vocab.lookup(flat), len(cf))
        else:
            self._cf = None
        self._attached: dict[int, int] = {}       # paiement → client file (positions)
        self._consumed: set[int] = set()
        self._amounts_as_of = None                 # (as_of, montants, débiteurs) du jour courant

    # --- Signaux -------------------------------------------------------------------------------

    def _reference(self, pos: np.ndarray, as_of) -> pd.DataFrame:
        cfg = self.cfg.signals.reference
        row, keys = self.ref.payment_keys.gather(pos)
        keep = self.ref.vocab.lengths[keys] >= cfg.min_key_length
        row, keys = row[keep], keys[keep]
        return self._debtors_of_keys(row, keys, as_of, REFERENCE, WEIGHTS[REFERENCE], cfg.max_debtors_per_key,
                                     cfg.strong_min_key_length, self._pay_amount[pos])

    def _debtors_of_keys(self, row: np.ndarray, keys: np.ndarray, as_of, signal: str, weight: float,
                         max_debtors: int, strong_min_length: int,
                         row_amount: np.ndarray | None = None) -> pd.DataFrame:
        """(ligne, clé) → débiteurs des factures **ouvertes** à D portant la clé.

        Si `row_amount` est fourni et qu'une de ces factures a un restant dû égal au montant du
        paiement, seules celles-là sont retenues. Spécificité = 1 / nombre de débiteurs restants,
        calculée par (paiement, clé) : une référence partagée par plusieurs cédants reste
        exploitable quand une seule facture ouverte correspond.
        """
        if len(keys) == 0:
            return _empty_signal()
        ukeys, k_of_row = np.unique(keys, return_inverse=True)
        k_owner, inv = self.ref.key_invoices.gather(ukeys)
        balance = self.state.open_balance_at(inv, as_of)
        deb = self._inv_debtor[inv]
        ok = (balance > 0) & (deb >= 0)
        k_owner, deb, balance = k_owner[ok], deb[ok], balance[ok]
        asked = pd.DataFrame({"row": row, "k": k_of_row}).drop_duplicates()
        # Jointures sur des tables dédoublonnées, jamais (ligne × toutes les factures de la clé) : une clé
        # fréquente (numéro de commande générique, année) chez un gros débiteur porte des milliers de
        # factures. Clé → débiteurs distincts ; clés partagées par trop de débiteurs écartées d'emblée.
        key_debtors = pd.DataFrame({"k": k_owner, "debtor": deb}).drop_duplicates()
        n_key = np.bincount(key_debtors["k"].to_numpy(), minlength=len(ukeys))
        exact = pd.DataFrame({c: pd.Series(dtype="int64") for c in ("row", "k", "debtor")})
        if row_amount is not None:
            # Factures de la clé dont le restant dû égale le montant du paiement : elles seules comptent.
            by_amount = pd.DataFrame({"k": k_owner, "amount": balance, "debtor": deb}).drop_duplicates()
            exact = (asked.assign(amount=row_amount[asked["row"].to_numpy()])
                     .merge(by_amount, on=["k", "amount"])[["row", "k", "debtor"]])
            if len(exact):
                code = asked["row"].to_numpy() * len(ukeys) + asked["k"].to_numpy()
                matched = np.unique(exact["row"].to_numpy() * len(ukeys) + exact["k"].to_numpy())
                asked = asked[~np.isin(code, matched)]
        light = key_debtors[n_key[key_debtors["k"].to_numpy()] <= max_debtors]
        pairs = pd.concat([exact, asked.merge(light, on="k")], ignore_index=True)
        if pairs.empty:
            return _empty_signal()
        n_deb = pairs.groupby(["row", "k"])["debtor"].transform("size").to_numpy()
        pairs = pairs[n_deb <= max_debtors]
        n_deb = n_deb[n_deb <= max_debtors]
        k = pairs["k"].to_numpy()
        pairs = pairs.assign(signal=signal, score=weight / n_deb,
                             strong=(n_deb == 1) & (self.ref.vocab.lengths[ukeys[k]] >= strong_min_length))
        return (pairs.groupby(["row", "debtor", "signal"], as_index=False)
                .agg(score=("score", "max"), strong=("strong", "any")))

    def _iban(self, pos: np.ndarray, as_of) -> tuple[pd.DataFrame, np.ndarray]:
        n = len(pos)
        ib = self._pay_iban[pos]
        route = np.full(n, UNKNOWN, dtype=object)
        rows = np.flatnonzero(ib >= 0)
        if len(rows) == 0:
            return _empty_signal(), route
        tech = self.iban.technical[ib[rows]]
        d_owner, d_pos = self.iban.debtors.gather(ib[rows])
        known = self.state.party_known_at("debtor", d_pos, as_of)
        d_owner, d_pos = d_owner[known], d_pos[known]
        a_owner, a_pos = self.iban.assignors.gather(ib[rows])
        known = self.state.party_known_at("assignor", a_pos, as_of)
        a_owner, a_pos = a_owner[known], a_pos[known]
        n_d = np.bincount(d_owner, minlength=len(rows))
        n_a = np.bincount(a_owner, minlength=len(rows))

        # IBAN à la fois débiteur et cédant : arbitrage par bankroll_code du paiement s'il est connu.
        conflict = (n_d > 0) & (n_a > 0)
        to_assignor = np.zeros(len(rows), dtype=bool)
        if conflict.any():
            pay_br = self._pay_bankroll[pos[rows]]
            ib_rows = ib[rows]
            a_match = pd.DataFrame({"o": a_owner, "ib": ib_rows[a_owner], "pos": a_pos}).merge(
                self.iban.bankroll["assignor"], on=["ib", "pos"])
            d_match = pd.DataFrame({"o": d_owner, "ib": ib_rows[d_owner], "pos": d_pos}).merge(
                self.iban.bankroll["debtor"], on=["ib", "pos"])
            a_hit = np.zeros(len(rows), dtype=bool)
            d_hit = np.zeros(len(rows), dtype=bool)
            for frame, hit in ((a_match, a_hit), (d_match, d_hit)):
                eq = frame["br"].to_numpy() == pay_br[frame["o"].to_numpy()]
                hit[frame["o"].to_numpy()[eq & pd.notna(frame["br"]).to_numpy()]] = True
            to_assignor = conflict & a_hit & ~d_hit
        sub = np.select([tech, to_assignor, n_d > 0, n_a > 0], [TECHNICAL_ACCOUNT, ASSIGNOR, DEBTOR_DIRECT, ASSIGNOR],
                        default=UNKNOWN)
        route[rows] = sub
        direct = (sub == DEBTOR_DIRECT)[d_owner]
        o, d = d_owner[direct], d_pos[direct]
        signal = pd.DataFrame({"row": rows[o], "debtor": d, "signal": IBAN,
                               "score": WEIGHTS[IBAN] / n_d[o], "strong": (n_d[o] == 1) & ~conflict[o]})
        return signal, route

    def _name(self, pos: np.ndarray, as_of) -> pd.DataFrame:
        cfg = self.cfg.signals.name
        idx = self.names
        row, terms = idx.payment_terms.gather(pos)
        if len(terms) == 0:
            return _empty_signal()
        n_known = max(self.state.known_party_count("debtor", as_of), 1)
        limit = min(cfg.max_token_share * n_known, cfg.max_debtors_per_term)

        def df_of(term_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Débiteurs connus à D par terme et fréquence documentaire à D."""
            owner, deb = idx.term_debtors.gather(term_ids)
            known = self.state.party_known_at("debtor", deb, as_of)
            owner, deb = owner[known], deb[known]
            return owner, deb, np.bincount(owner, minlength=len(term_ids))

        uterms, t_of_row = np.unique(terms, return_inverse=True)
        t_owner, t_deb, df = df_of(uterms)
        usable = (df > 0) & (df <= limit)
        idf = np.where(usable, np.log1p(n_known / np.maximum(df, 1)), 0.0)
        keep = usable[t_owner]
        postings = pd.DataFrame({"t": t_owner[keep], "debtor": t_deb[keep]})
        matched = (pd.DataFrame({"row": row, "t": t_of_row}).drop_duplicates()
                   .merge(postings, on="t"))
        if matched.empty:
            return _empty_signal()
        matched["idf"] = idf[matched["t"].to_numpy()]
        matched["spec"] = 1.0 / np.maximum(df[matched["t"].to_numpy()], 1)
        score = matched.groupby(["row", "debtor"], as_index=False).agg(idf=("idf", "sum"), spec=("spec", "max"))

        # Dénominateur : poids de tous les termes (utilisables à D) du nom de chaque débiteur candidat.
        cand = np.unique(score["debtor"].to_numpy())
        c_owner, c_terms = idx.debtor_terms.gather(cand)
        ut, t_inv = np.unique(c_terms, return_inverse=True)
        _, _, df_c = df_of(ut)
        w = np.where((df_c > 0) & (df_c <= limit), np.log1p(n_known / np.maximum(df_c, 1)), 0.0)
        total = np.bincount(c_owner, weights=w[t_inv], minlength=len(cand))
        score["total"] = total[np.searchsorted(cand, score["debtor"].to_numpy())]
        score = score[score["total"] > 0]
        # Couverture du nom, ou spécificité du meilleur mot retrouvé : un mot porté par un seul débiteur
        # connu à D (patronyme, marque) désigne ce débiteur même si le libellé tronque le nom.
        coverage = (score["idf"] / score["total"]).clip(upper=1.0).to_numpy()
        similarity = np.maximum(coverage, score["spec"].to_numpy())
        keep = similarity >= cfg.min_similarity
        strong_names = pd.DataFrame({"row": score["row"].to_numpy()[keep], "debtor": score["debtor"].to_numpy()[keep],
                                     "signal": NAME, "score": WEIGHTS[NAME] * similarity[keep], "strong": False})
        weak = score[~keep][["row", "debtor"]]
        if self.cfg.signals.amount.enabled and len(weak):
            return pd.concat([strong_names, self._weak_name_near_amount(pos, weak, as_of)], ignore_index=True)
        return strong_names

    def _weak_name_near_amount(self, pos: np.ndarray, weak: pd.DataFrame, as_of) -> pd.DataFrame:
        """Nom peu spécifique (homonymes) corroboré par une facture ouverte de restant dû proche du paiement.

        Tolérance d'escompte ou de frais : 5 € ou 3 %. Seuls les débiteurs retenus par le nom sont
        examinés (pas de recherche sur tous les débiteurs).
        """
        debtors, d_of = np.unique(weak["debtor"].to_numpy(), return_inverse=True)
        owner, _, balance = self.state.debtor_open_invoices_at(debtors, as_of)
        if len(owner) == 0:
            return _empty_signal()
        # Restants dus triés par (débiteur, montant) : pour chaque couple (paiement, débiteur), existe-t-il
        # un restant dû dans [montant − tolérance, montant + tolérance] ? Deux recherches dichotomiques.
        top = np.int64(1) << 40
        sorted_keys = np.sort(owner.astype(np.int64) * top + np.clip(balance, 0, top - 1))
        amount = self._pay_amount[pos[weak["row"].to_numpy()]]
        tol = np.floor(np.maximum(500, 0.03 * amount)).astype(np.int64)   # |restant − montant| ≤ tol, en entiers
        base = d_of.astype(np.int64) * top
        lo = np.searchsorted(sorted_keys, base + np.clip(amount - tol, 0, top - 1), side="left")
        hi = np.searchsorted(sorted_keys, base + np.clip(amount + tol, 0, top - 1), side="right")
        pairs = weak[hi > lo]
        if pairs.empty:
            return _empty_signal()
        n = pairs.groupby("row")["debtor"].transform("size").to_numpy()
        return pd.DataFrame({"row": pairs["row"].to_numpy(), "debtor": pairs["debtor"].to_numpy(), "signal": NAME,
                             "score": WEIGHTS[NAME] * 0.8 / n, "strong": False})

    def _amount(self, pos: np.ndarray, as_of) -> pd.DataFrame:
        """Débiteurs ayant une facture ouverte de restant dû égal au paiement.

        Montant partagé par peu de débiteurs : tous retenus. Par davantage (jusqu'à
        `max_debtors_with_name_hint`) : seuls ceux dont un mot du nom figure dans le libellé.
        """
        cfg = self.cfg.signals.amount
        amounts, debtors = self._open_amounts(as_of)
        if len(amounts) == 0:
            return _empty_signal()
        # Couples (montant, débiteur) distincts triés par montant : les débiteurs d'un montant forment une
        # tranche ; un montant rond partagé par trop de débiteurs est écarté sans être développé.
        pay_amount = self._pay_amount[pos]
        lo = np.searchsorted(amounts, pay_amount, side="left")
        n_per_row = np.searchsorted(amounts, pay_amount, side="right") - lo
        limit = max(cfg.max_debtors_per_amount, cfg.max_debtors_with_name_hint if self.names is not None else 0)
        n_per_row[(pay_amount <= 0) | (n_per_row > limit)] = 0
        total = int(n_per_row.sum())
        if total == 0:
            return _empty_signal()
        rows = np.repeat(np.arange(len(pos)), n_per_row)
        starts = np.repeat(lo - (np.cumsum(n_per_row) - n_per_row), n_per_row)
        pairs = pd.DataFrame({"row": rows, "debtor": debtors[starts + np.arange(total)]})
        n_all = n_per_row[rows]
        hint = self._name_hint(pos, pairs["row"].to_numpy(), pairs["debtor"].to_numpy())             if self.names is not None else np.zeros(len(pairs), dtype=bool)
        keep = (n_all <= cfg.max_debtors_per_amount) | hint
        pairs, hint, n_all = pairs[keep], hint[keep], n_all[keep]
        n_hint = pd.Series(hint).groupby(pairs["row"].to_numpy()).transform("sum").to_numpy()
        n = np.where(hint & (n_all > cfg.max_debtors_per_amount), n_hint, n_all)
        return pd.DataFrame({"row": pairs["row"].to_numpy(), "debtor": pairs["debtor"].to_numpy(), "signal": AMOUNT,
                             "score": WEIGHTS[AMOUNT] / np.maximum(n, 1), "strong": False})

    def _open_amounts(self, as_of) -> tuple[np.ndarray, np.ndarray]:
        """Couples (restant dû, débiteur) distincts des factures ouvertes à D, triés ; calculés une fois par jour."""
        if self._amounts_as_of is None or self._amounts_as_of[0] != as_of:
            inv, balance = self.state.open_invoice_positions(as_of)
            deb = self._inv_debtor[inv]
            ok = deb >= 0
            amount, deb = balance[ok].astype(np.int64), deb[ok].astype(np.int64)
            order = np.lexsort((deb, amount))
            amount, deb = amount[order], deb[order]
            keep = np.ones(len(amount), dtype=bool)
            keep[1:] = (amount[1:] != amount[:-1]) | (deb[1:] != deb[:-1])
            self._amounts_as_of = (as_of, amount[keep], deb[keep])
        return self._amounts_as_of[1], self._amounts_as_of[2]

    def _name_hint(self, pos: np.ndarray, rows: np.ndarray, debtors: np.ndarray) -> np.ndarray:
        """Pour chaque couple (ligne, débiteur) : un terme du nom du débiteur figure-t-il dans le libellé ?"""
        if len(rows) == 0:
            return np.zeros(0, dtype=bool)
        pairs = pd.DataFrame({"i": np.arange(len(rows)), "row": rows, "debtor": debtors})
        p_row, p_term = self.names.payment_terms.gather(pos)
        pay_terms = pd.DataFrame({"row": p_row, "term": p_term})
        udeb, d_of = np.unique(debtors, return_inverse=True)
        d_owner, d_term = self.names.debtor_terms.gather(udeb)
        deb_terms = pd.DataFrame({"debtor": udeb[d_owner], "term": d_term})
        hits = pairs.merge(pay_terms, on="row").merge(deb_terms, on=["debtor", "term"])
        out = np.zeros(len(rows), dtype=bool)
        out[hits["i"].to_numpy()] = True
        return out

    def _client_file(self, pos: np.ndarray, as_of) -> tuple[pd.DataFrame, np.ndarray]:
        """Rattache les client files reçus aux paiements, puis en déduit les débiteurs cités."""
        cfg = self.cfg.signals.client_file
        attached = np.array([self._attached.get(int(p), -1) for p in pos], dtype=np.int64)
        free = np.flatnonzero(attached < 0)
        received = self.state.client_files_received_at(as_of)
        available = received[~np.isin(received, np.fromiter(self._consumed, np.int64, len(self._consumed)))]
        if len(free) and len(available):
            files = pd.DataFrame({"f": available, "amount": self._cf_amount[available]})
            pays = pd.DataFrame({"r": free, "amount": self._pay_amount[pos[free]]})
            pairs = pays.merge(files[files["amount"] >= 0], on="amount")
            if len(pairs):
                r, f = pairs["r"].to_numpy(), pairs["f"].to_numpy()
                p = pos[r]
                date_ok = np.abs(self._cf_day[f] - self._pay_value_day[p]) <= cfg.date_tolerance_days
                iban_ok = (self._cf_iban[f] >= 0) & (self._cf_iban[f] == self._pay_iban[p]) \
                    if self._cf_iban is not None else np.zeros(len(f), bool)
                ref_ok = np.fromiter((bool(c) and c in lbl for c, lbl in zip(self._cf_ref[f], self._pay_label[p])),
                                     dtype=bool, count=len(f))
                pairs = pairs[date_ok | iban_ok | ref_ok]
                # Sans ambiguïté : un seul fichier pour le paiement, un seul paiement pour le fichier.
                pairs = pairs[~pairs.duplicated("r", keep=False) & ~pairs.duplicated("f", keep=False)]
                for r_i, f_i in zip(pairs["r"].to_numpy(), pairs["f"].to_numpy()):
                    self._attached[int(pos[r_i])] = int(f_i)
                    self._consumed.add(int(f_i))
                    attached[r_i] = f_i
        rows = np.flatnonzero(attached >= 0)
        if len(rows) == 0:
            return _empty_signal(), attached
        f_owner, keys = self._file_keys.gather(attached[rows])
        cfg_ref = self.cfg.signals.reference
        keep = self.ref.vocab.lengths[keys] >= cfg_ref.min_key_length
        by_key = self._debtors_of_keys(rows[f_owner[keep]], keys[keep], as_of, CLIENT_FILE, 1.0,
                                       cfg_ref.max_debtors_per_key, 0)
        if by_key.empty:
            return by_key, attached
        # Univoque si toutes les lignes du fichier désignent le même débiteur.
        n = by_key.groupby("row")["debtor"].transform("nunique").to_numpy()
        by_key = by_key.assign(score=WEIGHTS[CLIENT_FILE] / n, strong=n == 1)
        return by_key, attached

    # --- Allocation du lot -------------------------------------------------------------------------

    def allocate(self, ctx: DayContext) -> Allocation:
        """Allocation du lot. Le rattachement des client files se fait sur tout le lot (unicité paiement ↔
        fichier) ; les autres signaux, propres à chaque paiement, par blocs de `CHUNK_ROWS` lignes, ce qui
        borne la mémoire quelle que soit la taille du reliquat."""
        as_of = ctx.as_of
        batch_ids = ctx.batch["payment_id"].astype(object).to_numpy()
        pos = self.state.pay_pos(pd.Series(batch_ids, dtype=object))
        attached = np.full(len(pos), -1, dtype=np.int64)
        cf_signal = _empty_signal()
        if self.cfg.signals.client_file.enabled and self._cf is not None:
            memory.mark("allocation · client files")
            cf_signal, attached = self._client_file(pos, as_of)
        cf_row = cf_signal["row"].to_numpy()
        chunks = []
        n_blocks = max(-(-len(pos) // CHUNK_ROWS), 1)
        for start in range(0, max(len(pos), 1), CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, len(pos))
            self._block = f"bloc {start // CHUNK_ROWS + 1}/{n_blocks}"
            in_chunk = (cf_row >= start) & (cf_row < end)
            cf_part = cf_signal[in_chunk].assign(row=cf_row[in_chunk] - start)
            chunks.append(self._allocate_rows(pos[start:end], batch_ids[start:end], attached[start:end],
                                              cf_part, as_of))
        if len(chunks) == 1:
            return chunks[0]
        # infer_objects : mêmes types qu'en un seul bloc (un bloc sans aucun débiteur ferme reste en `object`).
        return Allocation(pd.concat([c.candidates for c in chunks], ignore_index=True).infer_objects(),
                          pd.concat([c.payments for c in chunks], ignore_index=True).infer_objects())

    def _allocate_rows(self, pos: np.ndarray, batch_ids: np.ndarray, attached: np.ndarray,
                       cf_signal: pd.DataFrame, as_of) -> Allocation:
        sig = self.cfg.signals
        parts = [cf_signal]
        route = np.full(len(pos), UNKNOWN, dtype=object)
        block = getattr(self, "_block", "")
        if len(pos):
            if sig.reference.enabled:
                memory.mark(f"allocation · référence · {block}")
                parts.append(self._reference(pos, as_of))
            if sig.iban.enabled:
                memory.mark(f"allocation · IBAN · {block}")
                iban_signal, route = self._iban(pos, as_of)
                parts.append(iban_signal)
            if sig.name.enabled:
                memory.mark(f"allocation · nom · {block}")
                parts.append(self._name(pos, as_of))
            if sig.amount.enabled:
                memory.mark(f"allocation · montant · {block}")
                parts.append(self._amount(pos, as_of))
        memory.mark(f"allocation · combinaison · {block}")
        signals = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) \
            else _empty_signal()
        return self._combine(signals, batch_ids, route, attached)

    def _combine(self, s: pd.DataFrame, batch_ids: np.ndarray, route: np.ndarray, attached: np.ndarray) -> Allocation:
        n = len(batch_ids)
        cf_ids = np.full(n, None, dtype=object)
        if self._cf is not None:
            has = attached >= 0
            cf_ids[has] = self._cf_ids[attached[has]]
        if s.empty:
            cand = pd.DataFrame(columns=["payment_id", "rank", "debtor_id", "score", "signal", "signals", "strong"])
            status = np.full(n, NONE, dtype=object)
            firm = np.full(n, None, dtype=object)
            n_cand = np.zeros(n, dtype=np.int64)
        else:
            # Réductions par segments (row, débiteur) en numpy : pas de groupby pandas par jour.
            row = s["row"].to_numpy(dtype=np.int64)
            deb = s["debtor"].to_numpy(dtype=np.int64)
            score = s["score"].to_numpy(dtype=np.float64)
            code = s["signal"].map(_SIGNAL_CODE).to_numpy(dtype=np.int64)
            strong = s["strong"].to_numpy(dtype=bool)
            order = np.lexsort((-score, deb, row))
            row, deb, score, code, strong = row[order], deb[order], score[order], code[order], strong[order]
            starts = np.flatnonzero(np.r_[True, (row[1:] != row[:-1]) | (deb[1:] != deb[:-1])])
            c_row, c_deb = row[starts], deb[starts]
            c_score = 1.0 - np.multiply.reduceat(1.0 - score, starts)
            c_strong = np.logical_or.reduceat(strong, starts)
            c_bits = np.bitwise_or.reduceat(1 << code, starts)
            c_signal = code[starts]                          # signal au score le plus élevé
            # Classement : score décroissant, puis position du débiteur (déterministe). Score arrondi pour le
            # tri : deux scores égaux au bruit de sommation flottante près sont départagés par le débiteur.
            order = np.lexsort((c_deb, -np.round(c_score, 9), c_row))
            c_row, c_deb, c_score, c_strong, c_bits, c_signal = (
                a[order] for a in (c_row, c_deb, c_score, c_strong, c_bits, c_signal))
            first = np.flatnonzero(np.r_[True, c_row[1:] != c_row[:-1]])
            rank = np.arange(len(c_row)) - np.repeat(first, np.diff(np.r_[first, len(c_row)])) + 1
            n_strong = np.bincount(c_row[c_strong], minlength=n)
            firm_mask = c_strong & (n_strong[c_row] == 1)
            firm = np.full(n, None, dtype=object)
            firm[c_row[firm_mask]] = self._debtor_ids[c_deb[firm_mask]]
            keep = rank <= self.cfg.max_candidates
            n_cand = np.bincount(c_row[keep], minlength=n)
            status = np.where(pd.notna(firm), FIRM, np.where(n_cand > 0, MULTIPLE, NONE))
            cand = pd.DataFrame({
                "payment_id": batch_ids[c_row[keep]], "rank": rank[keep], "debtor_id": self._debtor_ids[c_deb[keep]],
                "score": c_score[keep], "signal": np.asarray(SIGNALS, dtype=object)[c_signal[keep]],
                "signals": pd.Series(c_bits[keep]).map(_SIGNAL_LABEL).to_numpy(), "strong": c_strong[keep],
            })
        payments = pd.DataFrame({"payment_id": batch_ids, "status": status, "firm_debtor_id": firm,
                                 "n_candidates": n_cand, "iban_route": route, "client_file_id": cf_ids})
        return Allocation(cand, payments)
