"""État du grand livre à une date (brief §4.1, spec §8.2).

`LedgerState` rejoue le journal d'événements et répond aux questions du moteur
pour l'instant `as_of` auquel il a été avancé : il reflète exactement les
événements d'horodatage **strictement antérieur** à `as_of`.

Garanties anti-fuite, mécaniques :
- toute lecture exige un `as_of` égal à l'instant de l'état, sinon `TemporalError` ;
- l'état ne recule jamais (`advance_to` refuse un instant passé) ;
- une facture n'est visible qu'après son `INVOICE_CREATED`, un client file après
  son `CLIENT_FILE_RECEIVED` ;
- le restant dû est reconstruit par les imputations (`invoice.current_amount`
  n'est jamais chargé dans les tables).

Volumétrie : l'état est en tableaux numpy indexés par position ; le journal est
converti une fois en entiers et appliqué par blocs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.load.events import EVENT_RANK, EventType
from src.load.loader import LoadedData

DAY_US = 86_400_000_000
_CODE = {e: EVENT_RANK[e.value] for e in EventType}
_ROLES = ("assignor", "debtor")

# Agrégats comportementaux en fenêtre glissante (colonnes de `_win`).
_W_PAYMENTS, _W_LINES, _W_PARTIAL, _W_GROUPED, _W_CITED, _W_DELAY_N, _W_DELAY_SUM, _W_DELAY_SQ = range(8)
_N_WIN = 8

INVOICE_COLUMNS = [
    "invoice_id", "client_reference", "internal_reference", "creation_date", "due_date", "initial_amount",
    "currency", "debtor_id", "agreement_id", "client_reference_keys", "internal_reference_keys",
]


class TemporalError(RuntimeError):
    """Lecture ou avance de l'état incompatible avec l'instant qu'il représente."""


def _to_us(t: pd.Timestamp | np.datetime64 | str) -> int:
    return int(pd.Timestamp(t).as_unit("us").asm8.view("i8"))


def _days(values: pd.Series) -> np.ndarray:
    """Dates → numéro de jour (int64), NaT → valeur sentinelle très basse."""
    arr = values.to_numpy(dtype="datetime64[us]").astype("datetime64[D]")
    out = arr.astype(np.int64)
    out[np.isnat(arr)] = np.iinfo(np.int64).min // 2
    return out


class _Positions:
    """Identifiant → position dans une table (−1 si inconnu). Table de hachage construite une fois."""

    def __init__(self, ids: pd.Series):
        self.index = pd.Index(ids.astype(object).to_numpy())

    def __call__(self, ids) -> np.ndarray:
        values = ids.astype(object).to_numpy() if isinstance(ids, pd.Series) else np.asarray(ids, dtype=object)
        return self.index.get_indexer(values).astype(np.int64)

    def __len__(self) -> int:
        return len(self.index)


@dataclass(frozen=True)
class Journal:
    """Journal converti en tableaux : un événement par ligne, dans l'ordre du journal."""

    ts: np.ndarray       # int64, µs
    code: np.ndarray     # int8, rang du type d'événement
    pos: np.ndarray      # int64, position de l'entité dans sa table
    pos2: np.ndarray     # int64, facture (imputation) ou rôle (partie), sinon −1
    amount: np.ndarray   # int64, montant imputé (imputation), sinon 0

    def __len__(self) -> int:
        return len(self.ts)


class LedgerState:
    def __init__(self, data: LoadedData, journal: pd.DataFrame, window_days: int = 180):
        t = data.tables
        self.window_days = window_days
        self._inv = t["invoice"].reset_index(drop=True)
        self._pay = t["payment"].reset_index(drop=True)
        self._agr = t["agreement"].reset_index(drop=True)
        self._parties = {r: t[r].reset_index(drop=True) for r in _ROLES}
        self._cf = t["client_file"].reset_index(drop=True) if "client_file" in t else None
        self._cfl = t["client_file_line"].reset_index(drop=True) if "client_file_line" in t else None
        self._tech = t["technical_account"].reset_index(drop=True) if "technical_account" in t else None
        self._party_iban = t["party_iban"].reset_index(drop=True) if "party_iban" in t else None

        self.inv_pos = _Positions(self._inv["invoice_id"])
        self.pay_pos = _Positions(self._pay["payment_id"])
        self.agr_pos = _Positions(self._agr["agreement_id"])
        self.party_pos = {r: _Positions(p["party_id"]) for r, p in self._parties.items()}
        self.cf_pos = _Positions(self._cf["file_id"]) if self._cf is not None else None
        debtor_pos = self.party_pos["debtor"]

        n_inv, n_pay, n_deb = len(self._inv), len(self._pay), len(self._parties["debtor"])
        # Attributs statiques utilisés par l'état.
        self._inv_initial = self._inv["initial_amount"].fillna(0).to_numpy(dtype=np.int64)
        self._inv_debtor = debtor_pos(self._inv["debtor_id"])
        self._inv_due_day = _days(self._inv["due_date"])
        self._pay_value_day = _days(self._pay["value_date"])
        # Débiteur → factures (CSR) pour les lectures par débiteur.
        order = np.argsort(self._inv_debtor, kind="stable")
        self._by_debtor = order
        self._debtor_offsets = np.searchsorted(self._inv_debtor[order], np.arange(n_deb + 1))

        # État mutable.
        self._cursor = 0                       # prochain événement à appliquer
        self._as_of: int | None = None         # µs ; événements < as_of appliqués
        self._inv_created = np.zeros(n_inv, dtype=bool)
        self._balance = np.zeros(n_inv, dtype=np.int64)
        self._pay_received = np.zeros(n_pay, dtype=bool)
        self._pay_imputed = np.zeros(n_pay, dtype=np.int64)
        self._pay_lines = np.zeros(n_pay, dtype=np.int64)
        self._agr_active = np.zeros(len(self._agr), dtype=bool)
        # Partie sans date d'ouverture : active depuis toujours.
        self._party_active = {r: p["opened_at"].isna().to_numpy().copy() for r, p in self._parties.items()}
        self._party_known = {r: a.copy() for r, a in self._party_active.items()}
        self._cf_received = np.zeros(len(self._cf) if self._cf is not None else 0, dtype=bool)
        self._open_count = np.zeros(n_deb, dtype=np.int64)
        self._open_amount = np.zeros(n_deb, dtype=np.int64)
        self._win = np.zeros((n_deb, _N_WIN), dtype=np.float64)
        self._contrib: deque[tuple[int, np.ndarray, np.ndarray]] = deque()   # (jour, débiteurs, valeurs)

        self.journal = self._encode(journal)

    # --- Construction ----------------------------------------------------------------------

    def _encode(self, journal: pd.DataFrame) -> Journal:
        j = journal.reset_index(drop=True)
        n = len(j)
        code = j["event_type"].map(EVENT_RANK).to_numpy(dtype=np.int8)
        pos = np.full(n, -1, dtype=np.int64)
        pos2 = np.full(n, -1, dtype=np.int64)
        ent = j["entity_id"]

        def fill(types: tuple[EventType, ...], positions: _Positions | None) -> np.ndarray:
            mask = np.isin(code, [_CODE[t] for t in types])
            if positions is not None and mask.any():
                pos[mask] = positions(ent[mask])
            return mask

        fill((EventType.INVOICE_CREATED,), self.inv_pos)
        fill((EventType.PAYMENT_RECEIVED,), self.pay_pos)
        fill((EventType.AGREEMENT_CREATED, EventType.AGREEMENT_DISABLED), self.agr_pos)
        fill((EventType.CLIENT_FILE_RECEIVED,), self.cf_pos)
        imp = fill((EventType.IMPUTATION_APPLIED,), self.pay_pos)
        pos2[imp] = self.inv_pos(j["related_id"][imp])
        party = np.isin(code, [_CODE[EventType.PARTY_OPENED], _CODE[EventType.PARTY_CLOSED]])
        for r, role in enumerate(_ROLES):
            mask = party & (j["related_id"] == role.upper()).to_numpy()
            pos[mask] = self.party_pos[role](ent[mask])
            pos2[mask] = r
        ts = j["ts"].to_numpy(dtype="datetime64[us]").astype(np.int64)
        if np.any(np.diff(ts) < 0):
            raise ValueError("journal non trié par horodatage")
        return Journal(ts, code, pos, pos2, j["amount"].fillna(0).to_numpy(dtype=np.int64))

    # --- Avance -------------------------------------------------------------------------------

    @property
    def as_of(self) -> pd.Timestamp | None:
        return None if self._as_of is None else pd.Timestamp(self._as_of, unit="us")

    def advance_to(self, as_of) -> None:
        """Applique tous les événements d'horodatage strictement antérieur à `as_of`."""
        target = _to_us(as_of)
        if self._as_of is not None and target < self._as_of:
            raise TemporalError(f"l'état est au {self.as_of}, il ne peut pas revenir au {pd.Timestamp(as_of)}")
        ts = self.journal.ts
        hi = int(np.searchsorted(ts, target, side="left"))
        # Application jour par jour : le résultat ne dépend pas de la taille des sauts.
        while self._cursor < hi:
            day_end = (int(ts[self._cursor]) // DAY_US + 1) * DAY_US
            nxt = min(hi, int(np.searchsorted(ts, day_end, side="left")))
            self._apply(self._cursor, nxt)
            self._cursor = nxt
        self._as_of = target
        self._expire(target // DAY_US - self.window_days)

    def _apply(self, lo: int, hi: int) -> None:
        """Applique les événements [lo, hi) d'un même jour."""
        j = self.journal
        code, pos, pos2, amount = j.code[lo:hi], j.pos[lo:hi], j.pos2[lo:hi], j.amount[lo:hi]
        day = j.ts[lo:hi] // DAY_US

        def sel(event: EventType) -> np.ndarray:
            return (code == _CODE[event]) & (pos >= 0)

        for event, active in ((EventType.PARTY_OPENED, True), (EventType.PARTY_CLOSED, False)):
            m = sel(event)
            for r, role in enumerate(_ROLES):
                self._party_active[role][pos[m & (pos2 == r)]] = active
                if active:
                    self._party_known[role][pos[m & (pos2 == r)]] = True
        self._agr_active[pos[sel(EventType.AGREEMENT_CREATED)]] = True
        self._agr_active[pos[sel(EventType.AGREEMENT_DISABLED)]] = False
        self._cf_received[pos[sel(EventType.CLIENT_FILE_RECEIVED)]] = True
        self._pay_received[pos[sel(EventType.PAYMENT_RECEIVED)]] = True

        created = pos[sel(EventType.INVOICE_CREATED)]
        imp = sel(EventType.IMPUTATION_APPLIED) & (pos2 >= 0)
        imp_inv = pos2[imp]
        touched = np.unique(np.concatenate([created, imp_inv]))
        before = np.where(self._inv_created[touched], np.maximum(self._balance[touched], 0), 0)
        was_open = self._inv_created[touched] & (self._balance[touched] > 0)

        self._inv_created[created] = True
        self._balance[created] = self._inv_initial[created]
        np.subtract.at(self._balance, imp_inv, amount[imp])
        np.add.at(self._pay_imputed, pos[imp], amount[imp])
        np.add.at(self._pay_lines, pos[imp], 1)

        after = np.where(self._inv_created[touched], np.maximum(self._balance[touched], 0), 0)
        is_open = self._inv_created[touched] & (self._balance[touched] > 0)
        deb = self._inv_debtor[touched]
        ok = deb >= 0
        np.add.at(self._open_count, deb[ok], is_open[ok].astype(np.int64) - was_open[ok])
        np.add.at(self._open_amount, deb[ok], after[ok] - before[ok])

        if imp.any():
            self._add_window(day[imp], pos[imp], imp_inv)

    def _add_window(self, day: np.ndarray, pay: np.ndarray, inv: np.ndarray) -> None:
        """Contributions des imputations aux agrégats glissants, par jour d'imputation."""
        deb = self._inv_debtor[inv]
        keep = deb >= 0
        day, pay, inv, deb = day[keep], pay[keep], inv[keep], deb[keep]
        lines = pd.DataFrame({"day": day, "pay": pay, "inv": inv, "deb": deb})
        # Ligne groupée : le paiement impute plusieurs factures le même jour.
        per_pay = lines.groupby(["day", "pay"])["inv"].transform("nunique").to_numpy()
        first_of_pay = ~lines.duplicated(["day", "pay", "deb"]).to_numpy()
        delay = (self._pay_value_day[pay] - self._inv_due_day[inv]).astype(np.float64)
        has_delay = np.abs(delay) < 1e6
        delay = np.where(has_delay, delay, 0.0)
        pay_numbers = self._pay["label_numbers"].to_numpy()
        inv_keys = self._inv["client_reference_keys"].to_numpy()
        cited = np.fromiter((bool(set(pay_numbers[p]) & set(inv_keys[i])) for p, i in zip(pay, inv)),
                            dtype=bool, count=len(pay))
        values = np.zeros((len(lines), _N_WIN))
        values[:, _W_PAYMENTS] = first_of_pay
        values[:, _W_LINES] = 1
        values[:, _W_PARTIAL] = self._balance[inv] > 0
        values[:, _W_GROUPED] = per_pay > 1
        values[:, _W_CITED] = cited
        values[:, _W_DELAY_N] = has_delay
        values[:, _W_DELAY_SUM] = delay
        values[:, _W_DELAY_SQ] = delay * delay
        np.add.at(self._win, deb, values)
        for d in np.unique(day):
            m = day == d
            self._contrib.append((int(d), deb[m], values[m]))

    def _expire(self, first_kept_day: int) -> None:
        while self._contrib and self._contrib[0][0] < first_kept_day:
            _, deb, values = self._contrib.popleft()
            np.subtract.at(self._win, deb, values)

    # --- Lectures (toutes exigent as_of) ------------------------------------------------------

    def _check(self, as_of) -> None:
        if self._as_of is None:
            raise TemporalError("l'état n'a pas encore été avancé")
        if _to_us(as_of) != self._as_of:
            raise TemporalError(f"lecture au {pd.Timestamp(as_of)} sur un état au {self.as_of}")

    def _invoice_frame(self, positions: np.ndarray) -> pd.DataFrame:
        df = self._inv.iloc[positions][INVOICE_COLUMNS].reset_index(drop=True)
        df["open_amount"] = pd.array(self._balance[positions], dtype="Int64")
        return df

    def open_invoices(self, debtor_ids, as_of) -> pd.DataFrame:
        """Factures créées et non soldées à `as_of` des débiteurs donnés, avec leur restant dû."""
        self._check(as_of)
        deb = self.party_pos["debtor"](pd.Series(debtor_ids, dtype=object))
        deb = np.unique(deb[deb >= 0])
        if len(deb):
            starts, ends = self._debtor_offsets[deb], self._debtor_offsets[deb + 1]
            lengths = ends - starts
            idx = np.repeat(ends - lengths.cumsum(), lengths) + np.arange(lengths.sum())
            cand = self._by_debtor[idx]
        else:
            cand = np.array([], dtype=np.int64)
        cand = cand[self._inv_created[cand] & (self._balance[cand] > 0)]
        return self._invoice_frame(np.sort(cand))

    def invoices(self, invoice_ids, as_of) -> pd.DataFrame:
        """Factures demandées déjà créées à `as_of` (les autres sont ignorées), avec leur restant dû."""
        self._check(as_of)
        p = self.inv_pos(pd.Series(invoice_ids, dtype=object))
        p = p[p >= 0]
        return self._invoice_frame(p[self._inv_created[p]])

    def open_amount(self, invoice_ids, as_of) -> pd.Series:
        """Restant dû à `as_of` (`current_amount_as_of`), NA si la facture n'existe pas encore."""
        self._check(as_of)
        p = self.inv_pos(pd.Series(invoice_ids, dtype=object))
        known = (p >= 0) & self._inv_created[np.maximum(p, 0)]
        out = pd.array(np.where(known, self._balance[np.maximum(p, 0)], 0), dtype="Int64")
        out[~known] = pd.NA
        return pd.Series(out, index=invoice_ids.index if isinstance(invoice_ids, pd.Series) else None)

    def party_active(self, role: str, party_ids, as_of) -> np.ndarray:
        self._check(as_of)
        p = self.party_pos[role](pd.Series(party_ids, dtype=object))
        return np.where(p >= 0, self._party_active[role][np.maximum(p, 0)], False)

    def agreement_active(self, agreement_ids, as_of) -> np.ndarray:
        self._check(as_of)
        p = self.agr_pos(pd.Series(agreement_ids, dtype=object))
        return np.where(p >= 0, self._agr_active[np.maximum(p, 0)], False)

    def client_files(self, as_of) -> pd.DataFrame:
        """Client files reçus avant `as_of`."""
        self._check(as_of)
        if self._cf is None:
            return pd.DataFrame()
        return self._cf[self._cf_received].reset_index(drop=True)

    def client_file_lines(self, file_ids, as_of) -> pd.DataFrame:
        """Lignes des client files demandés, uniquement ceux déjà reçus."""
        self._check(as_of)
        if self._cf is None or self._cfl is None:
            return pd.DataFrame()
        p = self.cf_pos(pd.Series(file_ids, dtype=object))
        received = set(self._cf["file_id"].iloc[p[(p >= 0)][self._cf_received[p[p >= 0]]]])
        return self._cfl[self._cfl["file_id"].isin(received)].reset_index(drop=True)

    def payment_imputed(self, payment_ids, as_of) -> np.ndarray:
        """Le paiement a-t-il déjà au moins une imputation prononcée avant `as_of` ?"""
        self._check(as_of)
        p = self.pay_pos(pd.Series(payment_ids, dtype=object))
        return np.where(p >= 0, self._pay_lines[np.maximum(p, 0)] > 0, False)

    def debtor_stats(self, debtor_ids, as_of) -> pd.DataFrame:
        """Agrégats comportementaux à `as_of` : encours (instantané) et historique sur la fenêtre
        glissante `[as_of − window_days, as_of)`, strictement antérieure."""
        self._check(as_of)
        ids = pd.Series(debtor_ids, dtype=object).reset_index(drop=True)
        p = self.party_pos["debtor"](ids)
        ok = p >= 0
        q = np.maximum(p, 0)
        w = np.where(ok[:, None], self._win[q], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lines = w[:, _W_LINES]
            n_delay = w[:, _W_DELAY_N]
            mean = w[:, _W_DELAY_SUM] / n_delay
            var = np.maximum(w[:, _W_DELAY_SQ] / n_delay - mean * mean, 0.0)
            out = pd.DataFrame({
                "debtor_id": ids,
                "open_invoice_count": np.where(ok, self._open_count[q], 0),
                "open_invoice_amount": np.where(ok, self._open_amount[q], 0),
                "payment_count": np.rint(w[:, _W_PAYMENTS]).astype(np.int64),
                "imputation_count": np.rint(lines).astype(np.int64),
                "mean_payment_delay": np.where(n_delay > 0, mean, np.nan),
                "std_payment_delay": np.where(n_delay > 0, np.sqrt(var), np.nan),
                "partial_payment_rate": np.where(lines > 0, w[:, _W_PARTIAL] / lines, np.nan),
                "grouping_rate": np.where(lines > 0, w[:, _W_GROUPED] / lines, np.nan),
                "ref_citation_rate": np.where(lines > 0, w[:, _W_CITED] / lines, np.nan),
            })
        return out

    # --- Accès par positions (index de l'allocation, itérateur) -----------------------------------------
    # Les attributs statiques (débiteur d'une facture, nom d'un débiteur) ne dépendent pas du temps ;
    # ce qui en dépend (existence à D) passe par des lectures contrôlées.

    @property
    def invoice_debtor_positions(self) -> np.ndarray:
        return self._inv_debtor

    def invoice_created_at(self, positions: np.ndarray, as_of) -> np.ndarray:
        self._check(as_of)
        return self._inv_created[positions]

    def party_known_at(self, role: str, positions: np.ndarray, as_of) -> np.ndarray:
        """La partie existe-t-elle à `as_of` (ouverte, même fermée depuis) ?"""
        self._check(as_of)
        return self._party_known[role][positions]

    def known_party_count(self, role: str, as_of) -> int:
        self._check(as_of)
        return int(self._party_known[role].sum())

    def open_balance_at(self, positions: np.ndarray, as_of) -> np.ndarray:
        """Restant dû à `as_of` par positions ; 0 pour une facture pas encore créée."""
        self._check(as_of)
        return np.where(self._inv_created[positions], self._balance[positions], 0)

    def debtor_open_invoices_at(self, debtor_positions: np.ndarray, as_of) -> tuple[np.ndarray, np.ndarray,
                                                                                    np.ndarray]:
        """Factures ouvertes à `as_of` des débiteurs donnés : (index du débiteur demandé, facture, restant dû)."""
        self._check(as_of)
        deb = np.asarray(debtor_positions, dtype=np.int64)
        starts, ends = self._debtor_offsets[deb], self._debtor_offsets[deb + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            return (np.array([], dtype=np.int64),) * 3
        owner = np.repeat(np.arange(len(deb)), lengths)
        inv = self._by_debtor[np.repeat(ends - lengths.cumsum(), lengths) + np.arange(total)]
        ok = self._inv_created[inv] & (self._balance[inv] > 0)
        return owner[ok], inv[ok], self._balance[inv[ok]]

    def agreement_active_at(self, positions: np.ndarray, as_of) -> np.ndarray:
        self._check(as_of)
        return self._agr_active[positions]

    def open_invoice_positions(self, as_of) -> tuple[np.ndarray, np.ndarray]:
        """Positions et restant dû des factures ouvertes à `as_of`."""
        self._check(as_of)
        pos = np.flatnonzero(self._inv_created & (self._balance > 0))
        return pos, self._balance[pos]

    def client_files_received_at(self, as_of) -> np.ndarray:
        """Positions des client files reçus avant `as_of`."""
        self._check(as_of)
        return np.flatnonzero(self._cf_received)

    def table(self, name: str) -> pd.DataFrame:
        """Table statique (attributs non temporels). Toute lecture d'existence passe par les accesseurs datés."""
        return {"invoice": self._inv, "payment": self._pay, "agreement": self._agr, "client_file": self._cf,
                "client_file_line": self._cfl, "technical_account": self._tech,
                "party_iban": self._party_iban, **self._parties}[name]

    # --- Accès pour l'itérateur ---------------------------------------------------------------------

    def payments_frame(self, positions: np.ndarray) -> pd.DataFrame:
        return self._pay.iloc[positions].reset_index(drop=True)

    def payments_arriving(self, start, end) -> tuple[np.ndarray, np.ndarray]:
        """Paiements dont l'événement PAYMENT_RECEIVED tombe dans [start, end) : (positions, numéros de jour).

        Lu dans le journal, pas dans la table : c'est l'arrivée du lot, pas une lecture d'état.
        """
        j = self.journal
        lo, hi = np.searchsorted(j.ts, [_to_us(start), _to_us(end)], side="left")
        m = (j.code[lo:hi] == _CODE[EventType.PAYMENT_RECEIVED]) & (j.pos[lo:hi] >= 0)
        return j.pos[lo:hi][m], j.ts[lo:hi][m] // DAY_US

    def payment_day_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Premier et dernier jour d'arrivée de paiement dans le journal."""
        j = self.journal
        ts = j.ts[j.code == _CODE[EventType.PAYMENT_RECEIVED]]
        return pd.Timestamp(int(ts.min() // DAY_US), unit="D"), pd.Timestamp(int(ts.max() // DAY_US), unit="D")

    def payment_imputed_at(self, positions: np.ndarray, as_of) -> np.ndarray:
        """Comme `payment_imputed`, par positions (usage interne de l'itérateur)."""
        self._check(as_of)
        return self._pay_lines[positions] > 0
