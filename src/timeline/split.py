"""Découpage temporel en périodes train / validation / test (brief §4.2).

Découpage sur les jours, jamais aléatoire, ancré sur la fin de l'historique :
le test couvre les derniers mois, la validation les mois précédents, et
l'entraînement le reste. `purge_days` jours sont exclus entre deux blocs.
Toutes les bornes sont inclusives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.settings import SplitSettings

PERIODS = ("train", "validation", "test")
PURGE = "purge"
OUTSIDE = "hors période"


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class Period:
    name: str
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass(frozen=True)
class Split:
    periods: tuple[Period, ...]
    warnings: tuple[str, ...] = field(default=())

    def period(self, name: str) -> Period:
        return next(p for p in self.periods if p.name == name)


def _months_before(end: date, months: int) -> date:
    """Premier jour d'une période de `months` mois se terminant le jour `end`."""
    return (pd.Timestamp(end) + pd.Timedelta(days=1) - pd.DateOffset(months=months)).date()


def compute_split(cfg: SplitSettings, data_start: date, data_end: date) -> Split:
    if data_start > data_end:
        raise SplitError("plage de données vide")
    end = cfg.anchor_end or data_end
    if not data_start <= end <= data_end:
        raise SplitError(f"anchor_end {end} hors de la plage des données [{data_start}, {data_end}]")
    gap = timedelta(days=cfg.purge_days + 1)

    test = Period("test", _months_before(end, cfg.test_months), end)
    val_end = test.start - gap
    validation = Period("validation", _months_before(val_end, cfg.validation_months), val_end)
    train_end = validation.start - gap
    train_start = data_start
    if cfg.train_months is not None:
        train_start = max(data_start, _months_before(train_end, cfg.train_months))
    if train_start > train_end:
        raise SplitError("période d'entraînement vide : historique trop court pour ces durées")
    train = Period("train", train_start, train_end)

    warnings: list[str] = []
    if cfg.train_months is not None and train_start == data_start and \
            _months_before(train_end, cfg.train_months) < data_start:
        warnings.append(f"historique insuffisant : l'entraînement couvre {train.days} jours "
                        f"au lieu de {cfg.train_months} mois")
    if validation.start < data_start:
        raise SplitError("période de validation hors des données : historique trop court")
    return Split((train, validation, test), tuple(warnings))


def assign_period(days: pd.Series, split: Split) -> pd.Series:
    """Nom de la période de chaque date : train / validation / test / purge / hors période."""
    d = pd.to_datetime(days).dt.normalize()
    out = np.full(len(d), OUTSIDE, dtype=object)
    first, last = split.periods[0].start, split.periods[-1].end
    inside = (d >= pd.Timestamp(first)) & (d <= pd.Timestamp(last))
    out[inside.to_numpy()] = PURGE
    for p in split.periods:
        mask = (d >= pd.Timestamp(p.start)) & (d <= pd.Timestamp(p.end))
        out[mask.to_numpy()] = p.name
    return pd.Series(out, index=days.index)
