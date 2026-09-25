import pandas as pd

from src.arrow_ops import isin, lookup


def s(values, dtype="string"):
    return pd.Series(values, dtype=dtype)


def test_isin_handles_missing_values():
    assert isin(s(["a", None, "c"]), s(["c", "a", None])).tolist() == [True, False, True]


def test_isin_empty_reference():
    assert isin(s(["a"]), s([])).tolist() == [False]


def test_lookup_first_occurrence_and_missing():
    keys = pd.Series(["b", "z", None, "a"], dtype="string", index=[10, 11, 12, 13])
    out = lookup(keys, s(["a", "b", "b"]), pd.Series([1, 2, 3], dtype="Int64"))
    assert out.index.tolist() == [10, 11, 12, 13]
    assert out.tolist() == [2, pd.NA, pd.NA, 1]


def test_lookup_datetime_values():
    out = lookup(s(["x", "y"]), s(["x"]), pd.Series(pd.to_datetime(["2024-01-02"])))
    assert out.iloc[0] == pd.Timestamp("2024-01-02")
    assert pd.isna(out.iloc[1])
