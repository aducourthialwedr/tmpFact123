from datetime import date

import pandas as pd
import pytest

from src.settings import SplitSettings
from src.timeline.split import SplitError, assign_period, compute_split


def test_default_split_is_anchored_on_last_day():
    split = compute_split(SplitSettings(), date(2024, 1, 1), date(2024, 12, 31))
    train, val, test = split.periods
    assert (test.start, test.end) == (date(2024, 11, 1), date(2024, 12, 31))
    # 5 jours de purge entre les blocs.
    assert (val.start, val.end) == (date(2024, 8, 27), date(2024, 10, 26))
    assert (train.start, train.end) == (date(2024, 1, 1), date(2024, 8, 21))
    assert split.warnings == ()


def test_periods_never_overlap_and_respect_purge():
    cfg = SplitSettings(purge_days=3, test_months=1, validation_months=1)
    train, val, test = compute_split(cfg, date(2024, 1, 1), date(2024, 6, 30)).periods
    assert (val.start - train.end).days == 4
    assert (test.start - val.end).days == 4


def test_train_months_and_anchor():
    cfg = SplitSettings(train_months=3, anchor_end=date(2024, 10, 31), purge_days=0)
    train, val, test = compute_split(cfg, date(2024, 1, 1), date(2024, 12, 31)).periods
    assert test.end == date(2024, 10, 31)
    assert (train.start, train.end) == (date(2024, 4, 1), date(2024, 6, 30))


def test_warns_when_history_shorter_than_train_months():
    split = compute_split(SplitSettings(train_months=12), date(2024, 1, 1), date(2024, 12, 31))
    assert split.period("train").start == date(2024, 1, 1)
    assert split.warnings


@pytest.mark.parametrize("cfg, start, end", [
    (SplitSettings(test_months=6, validation_months=6), date(2024, 1, 1), date(2024, 12, 31)),
    (SplitSettings(anchor_end=date(2025, 6, 1)), date(2024, 1, 1), date(2024, 12, 31)),
])
def test_invalid_splits(cfg, start, end):
    with pytest.raises(SplitError):
        compute_split(cfg, start, end)


def test_assign_period_labels_purge_and_outside():
    split = compute_split(SplitSettings(train_months=1, purge_days=2, test_months=1, validation_months=1),
                          date(2024, 1, 1), date(2024, 12, 31))
    days = pd.Series(pd.to_datetime(["2024-12-15", "2024-11-29", "2024-11-15", "2024-10-10", "2024-01-05"]))
    assert assign_period(days, split).tolist() == ["test", "purge", "validation", "train", "hors période"]
