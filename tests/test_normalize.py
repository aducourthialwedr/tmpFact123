import pandas as pd

from src.load.normalize import (
    label_numbers,
    label_tokens,
    normalize_iban,
    normalize_text,
    number_variants,
    reference_keys,
    tokenize,
)


def matches(label: str, reference: str) -> bool:
    return bool(set(label_numbers(tokenize(label))) & set(reference_keys(reference)))


def test_normalize_text_accents_punctuation_spaces():
    assert normalize_text("  Règlement  fact. n°12/B – Œuvre  ") == "REGLEMENT FACT N 12 B OEUVRE"
    assert normalize_text("Société Générale") == "SOCIETE GENERALE"


def test_normalize_text_missing_values():
    assert normalize_text(None) == ""
    assert normalize_text(pd.NA) == ""
    assert normalize_text(float("nan")) == ""


def test_label_tokens_keeps_alphabetic_only():
    assert label_tokens(tokenize("VIR SEPA FA0012 DUPONT 2024")) == ["VIR", "SEPA", "DUPONT"]


def test_number_variants_prefix_and_leading_zeros():
    assert number_variants("FA0012345") == {"FA0012345", "0012345", "12345"}
    assert number_variants("000") == {"000"}


def test_number_variants_swap_inside_token():
    assert {"FACT123", "123"} <= number_variants("123FACT", swap=True)
    assert "FACT123" not in number_variants("123FACT")


def test_reference_keys():
    assert reference_keys("FA-0012345") == ["0012345", "12345", "FA0012345"]
    assert reference_keys("DUPONT") == ["DUPONT"]
    assert reference_keys(None) == []
    assert reference_keys(" - ") == []


def test_match_exact_reference():
    assert matches("VIR SEPA DUPONT FA0012345", "FA0012345")


def test_match_without_prefix_and_zeros():
    assert matches("VIR DUPONT 12345", "FA0012345")
    assert matches("VIR DUPONT 0012345", "FA0012345")


def test_match_inversion_between_tokens():
    assert matches("REGLEMENT 123 FACT", "FACT123")
    assert matches("REGLEMENT FACT 123", "FACT123")


def test_match_inversion_inside_token():
    assert matches("REGLEMENT 123FACT", "FACT123")


def test_match_reference_split_by_punctuation():
    assert matches("PAIEMENT FACT FA-2024-00123", "FA/2024/00123")
    assert matches("PAIEMENT 2024 00123", "F-2024-00123")


def test_no_match_on_unrelated_numbers():
    assert not matches("VIR 2024 DUPONT", "FA0012345")
    assert not matches("VIR DUPONT 123456", "FA0012345")


def test_window_skips_mostly_alphabetic_triplets():
    keys = label_numbers(tokenize("SARL FACT 001887"))
    assert "SARLFACT001887" not in keys
    assert "FACT001887" in keys


def test_label_numbers_sorted_and_deterministic():
    label = "VIR 123 FACT FA-2024-0099 DUPONT"
    first = label_numbers(tokenize(label))
    assert first == sorted(first)
    assert first == label_numbers(tokenize(label))


def test_normalize_iban():
    s = pd.Series(["fr76 3000-4000 0500", "", None])
    out = normalize_iban(s)
    assert out.iloc[0] == "FR76300040000500"
    assert out.iloc[1:].isna().all()
