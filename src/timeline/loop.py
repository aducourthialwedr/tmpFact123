"""Boucle quotidienne (brief §4.1) : un seul code pour le rejeu de l'historique et la production.

Pour chaque jour D, `DailyIterator` avance l'état à D (événements < D, soit
l'état figé à la veille) et fournit le lot à traiter : paiements arrivés le
jour D + reliquat non résolu des jours précédents, dans la limite de
`retention_days`. Un paiement quitte le reliquat quand :
- une imputation réelle a été prononcée avant D (traité par ailleurs) ;
- le rapprocheur l'a auto-validé ;
- sa rétention est dépassée (il sort du cycle automatique).

Tout paiement en reliquat est retraité chaque jour : il l'est donc notamment
le jour où son client file devient visible.

Un rapprocheur (`Matcher`) reçoit un `DayContext` et renvoie ses décisions ;
`run_replay` les contrôle (paiement du lot, facture déjà connue à D) — une
décision qui cite une facture future est une fuite et lève `LeakError`.

En rejeu, l'état suit le journal réel : les décisions du moteur ne modifient
pas les soldes, elles sont comparées aux imputations réelles à l'évaluation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

from src.timeline.state import DAY_US, LedgerState, _to_us

AUTO, REVIEW, REJECT = "auto", "review", "reject"
ACTIONS = (AUTO, REVIEW, REJECT)
DECISION_COLUMNS = ["payment_id", "invoice_id", "amount", "action", "step", "rule_id", "rule_version", "score"]


class LeakError(RuntimeError):
    """Une décision utilise une information non disponible au jour D."""


@dataclass
class DayContext:
    day: pd.Timestamp
    batch: pd.DataFrame          # paiements à traiter + is_new, first_day, days_pending
    state: LedgerState

    @property
    def as_of(self) -> pd.Timestamp:
        """Instant de l'état : minuit du jour D (événements strictement antérieurs)."""
        return self.day


class Matcher(Protocol):
    name: str

    def process(self, ctx: DayContext) -> pd.DataFrame:
        """Décisions du jour, colonnes `DECISION_COLUMNS` (une ligne par paiement × facture)."""
        ...


class NullMatcher:
    """Rapprocheur vide : ne décide rien. Sert de référence et de test du harnais."""

    name = "null"

    def process(self, ctx: DayContext) -> pd.DataFrame:
        return empty_decisions()


def empty_decisions() -> pd.DataFrame:
    return pd.DataFrame({
        "payment_id": pd.Series(dtype="string"), "invoice_id": pd.Series(dtype="string"),
        "amount": pd.Series(dtype="Int64"), "action": pd.Series(dtype="string"),
        "step": pd.Series(dtype="string"), "rule_id": pd.Series(dtype="string"),
        "rule_version": pd.Series(dtype="Int64"), "score": pd.Series(dtype="float64"),
    })


class DailyIterator:
    def __init__(self, state: LedgerState, start: date, end: date, retention_days: int = 60):
        if start > end:
            raise ValueError("start postérieur à end")
        self.state = state
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.retention_days = retention_days
        self._pending = np.array([], dtype=np.int64)      # positions des paiements en reliquat
        self._first_day = np.array([], dtype=np.int64)    # jour d'arrivée (numéro de jour)
        self._resolved: set[int] = set()
        self.last_counts: dict[str, int] = {}

    def resolve(self, payment_ids) -> None:
        """Retire du reliquat des paiements auto-validés par le rapprocheur."""
        p = self.state.pay_pos(pd.Series(list(payment_ids), dtype=object))
        self._resolved.update(int(x) for x in p[p >= 0])

    def __iter__(self) -> Iterator[DayContext]:
        state = self.state
        # Reliquat initial : paiements arrivés dans la fenêtre de rétention avant `start`.
        warm = self.start - pd.Timedelta(days=self.retention_days)
        state.advance_to(warm)
        self._pending, self._first_day = state.payments_arriving(warm, self.start)

        day = self.start
        while day <= self.end:
            state.advance_to(day)
            today = _to_us(day) // DAY_US
            new, _ = state.payments_arriving(day, day + pd.Timedelta(days=1))

            pend, first = self._pending, self._first_day
            imputed = state.payment_imputed_at(pend, day)
            engine = np.isin(pend, np.fromiter(self._resolved, dtype=np.int64, count=len(self._resolved)))
            expired = (today - first) > self.retention_days
            keep = ~(imputed | engine | expired)
            self.last_counts = {"new": len(new), "carried": int(keep.sum()), "left_imputed": int(imputed.sum()),
                                "left_engine": int((engine & ~imputed).sum()),
                                "expired": int((expired & ~imputed & ~engine).sum())}

            positions = np.concatenate([pend[keep], new])
            first_days = np.concatenate([first[keep], np.full(len(new), today, dtype=np.int64)])
            batch = state.payments_frame(positions)
            batch["is_new"] = np.concatenate([np.zeros(keep.sum(), bool), np.ones(len(new), bool)])
            batch["first_day"] = pd.to_datetime(first_days, unit="D")
            batch["days_pending"] = today - first_days
            self._pending, self._first_day = positions, first_days

            yield DayContext(day=day, batch=batch, state=state)
            day += pd.Timedelta(days=1)



@dataclass
class ReplayResult:
    decisions: pd.DataFrame
    daily: pd.DataFrame
    matcher: str
    start: date
    end: date
    seconds: float
    extra: dict = field(default_factory=dict)


def _validate(decisions: pd.DataFrame, ctx: DayContext) -> pd.DataFrame:
    missing = [c for c in DECISION_COLUMNS if c not in decisions.columns]
    if missing:
        raise ValueError(f"décisions sans les colonnes {missing}")
    dec = decisions[DECISION_COLUMNS].copy()
    if len(dec) == 0:
        return dec
    bad_action = ~dec["action"].isin(ACTIONS)
    if bad_action.any():
        raise ValueError(f"actions inconnues : {sorted(dec.loc[bad_action, 'action'].unique())}")
    outside = ~dec["payment_id"].isin(set(ctx.batch["payment_id"]))
    if outside.any():
        raise LeakError(f"{int(outside.sum())} décision(s) sur des paiements hors du lot du {ctx.day.date()}")
    cited = dec["invoice_id"].dropna().unique()
    known = set(ctx.state.invoices(cited, ctx.as_of)["invoice_id"])
    unknown = [i for i in cited if i not in known]
    if unknown:
        raise LeakError(f"{len(unknown)} facture(s) inconnue(s) au {ctx.day.date()} citées par le rapprocheur, "
                        f"ex. {unknown[:3]}")
    return dec


def run_replay(state: LedgerState, matcher: Matcher, start: date, end: date, retention_days: int = 60,
               on_day: Callable[[DayContext, dict], None] | None = None) -> ReplayResult:
    """Rejoue les jours [start, end] avec `matcher` ; décisions contrôlées et journalisées."""
    t0 = time.perf_counter()
    it = DailyIterator(state, start, end, retention_days)
    frames, daily = [], []
    for ctx in it:
        t = time.perf_counter()
        dec = _validate(matcher.process(ctx), ctx)
        it.resolve(dec.loc[dec["action"] == AUTO, "payment_id"].unique())
        if len(dec):
            frames.append(dec.assign(day=ctx.day))
        row = {"day": ctx.day, "batch": len(ctx.batch), **it.last_counts,
               "auto": int(dec.loc[dec["action"] == AUTO, "payment_id"].nunique()),
               "review": int(dec.loc[dec["action"] == REVIEW, "payment_id"].nunique()),
               "seconds": round(time.perf_counter() - t, 4)}
        daily.append(row)
        if on_day is not None:
            on_day(ctx, row)
    decisions = pd.concat(frames, ignore_index=True) if frames else empty_decisions().assign(
        day=pd.Series(dtype="datetime64[us]"))
    return ReplayResult(decisions, pd.DataFrame(daily), getattr(matcher, "name", type(matcher).__name__),
                        pd.Timestamp(start).date(), pd.Timestamp(end).date(), round(time.perf_counter() - t0, 1))
