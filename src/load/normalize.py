"""Normalisation des libellés, références et noms (brief §3.3, spec §3.1).

Déterministe et versionnée : toute modification du comportement doit
incrémenter `NORMALIZATION_VERSION` et relancer la chaîne depuis l'étape 1.

Vues produites pour un libellé :
- `label_norm`    : texte normalisé (majuscules, sans accents, alphanumérique).
- `label_tokens`  : tokens alphabétiques.
- `label_numbers` : clés de référence candidates — tokens contenant des
  chiffres, fenêtres de tokens adjacents concaténés, inversion bloc lettres /
  bloc chiffres, puis variantes sans préfixe alphabétique et sans zéros de tête.

Une référence de facture est réduite à ses `reference_keys` par la même
fonction de variantes : un match de référence = intersection non vide entre
`label_numbers` et `reference_keys`.

Volumétrie : ~2 M de paiements, ~3 M de factures. La normalisation du texte
est vectorisée (`normalize_series`) et c'est l'unique implémentation — la
version scalaire `normalize_text` l'appelle. Les vues à base de tokens sont
calculées une seule fois par valeur distincte, réparties sur plusieurs
processus si un `Executor` est fourni (résultat identique, ordre conservé).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from concurrent.futures import Executor
from itertools import chain
from typing import Any

import pandas as pd

NORMALIZATION_VERSION = "1.1.0"

# Fenêtre maximale de tokens adjacents concaténés dans un libellé. Le brief
# demande les paires (inversion) ; la fenêtre de 3 couvre en plus les
# références découpées par la ponctuation (`FA-2024-00123` → `FA 2024 00123`).
MAX_JOIN_WINDOW = 3

# « ß » avant la mise en majuscules (sa majuscule simple n'est pas « SS ») ;
# lettres que NFKD ne décompose pas, après.
_BEFORE_UPPER = {"ß": "SS"}
_AFTER_UPPER = {"Œ": "OE", "Æ": "AE", "Ø": "O", "Ð": "D", "Þ": "TH", "Ł": "L"}
_ALPHA_DIGITS = re.compile(r"([A-Z]+)([0-9]+)")
_DIGITS_ALPHA = re.compile(r"([0-9]+)([A-Z]+)")
_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --- Texte ------------------------------------------------------------------


def normalize_series(series: pd.Series) -> pd.Series:
    """Majuscules, sans accents (NFKD), non alphanumérique → espace, espaces compressés.

    Valeur manquante → chaîne vide. Retourne une série de chaînes sans NA.
    """
    s = series.astype("str").fillna("")
    for k, v in _BEFORE_UPPER.items():
        s = s.str.replace(k, v, regex=False)
    s = s.str.normalize("NFKD").str.replace(r"\p{M}+", "", regex=True).str.upper()
    for k, v in _AFTER_UPPER.items():
        s = s.str.replace(k, v, regex=False)
    return s.str.replace(r"[^A-Z0-9]+", " ", regex=True).str.strip()


def normalize_text(value: Any) -> str:
    """Version scalaire de `normalize_series` (même implémentation)."""
    return normalize_series(pd.Series([value], dtype=object)).iloc[0]


def tokenize(value: Any) -> list[str]:
    return normalize_text(value).split()


# En dessous de ce nombre de valeurs distinctes, la parallélisation coûte plus qu'elle ne rapporte.
PARALLEL_MIN_VALUES = 100_000
_CHUNK_SIZE = 50_000


def _apply_chunk(fn: Callable[[Any], Any], values: list[Any]) -> list[Any]:
    return [fn(v) for v in values]


def _map_unique(series: pd.Series, fn: Callable[[Any], Any], executor: Executor | None = None) -> pd.Series:
    """Applique `fn` une fois par valeur distincte (`fn` doit être picklable si `executor`)."""
    codes, uniques = pd.factorize(series, use_na_sentinel=False)
    values = list(uniques)
    if executor is not None and len(values) >= PARALLEL_MIN_VALUES:
        chunks = [values[i:i + _CHUNK_SIZE] for i in range(0, len(values), _CHUNK_SIZE)]
        results = list(chain.from_iterable(executor.map(_apply_chunk, [fn] * len(chunks), chunks)))
    else:
        results = _apply_chunk(fn, values)
    return pd.Series([results[c] for c in codes], index=series.index, dtype=object)


# --- Références ---------------------------------------------------------------


def number_variants(token: str, swap: bool = False) -> set[str]:
    """Variantes d'un token normalisé contenant des chiffres.

    `FA0012345`             → {FA0012345, 0012345, 12345}
    `123FACT`, swap=True    → {123FACT, FACT123, 123}  (inversion dans le token)
    """
    out = {token}
    without_prefix = token.lstrip(_ALPHA)
    if without_prefix:
        out.add(without_prefix)
        without_zeros = without_prefix.lstrip("0")
        if without_zeros:
            out.add(without_zeros)
    if swap:
        for pattern in (_ALPHA_DIGITS, _DIGITS_ALPHA):
            m = pattern.fullmatch(token)
            if m:
                out |= number_variants(m.group(2) + m.group(1))
    return out


def label_tokens(tokens: Iterable[str]) -> list[str]:
    return [t for t in tokens if t.isalpha()]


def label_numbers(tokens: list[str], max_window: int = MAX_JOIN_WINDOW) -> list[str]:
    """Clés de référence candidates extraites d'un libellé tokenisé (triées).

    Une fenêtre de `size` tokens adjacents n'est concaténée que si au moins
    `size − 1` d'entre eux contiennent des chiffres (`FA 2024 00123` oui,
    `SARL FACT 001887` non).
    """
    # Tokens normalisés = [A-Z0-9]+ : « contient un chiffre » ⇔ « pas alphabétique ».
    has_digit = [not t.isalpha() for t in tokens]
    if not any(has_digit):
        return []
    keys: set[str] = set()
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if has_digit[i]:
            keys |= number_variants(tok, swap=True)
        for size in range(2, max_window + 1):
            if i + size > n:
                break
            if sum(has_digit[i:i + size]) < size - 1:
                continue
            window = tokens[i:i + size]
            keys |= number_variants("".join(window))
            if size == 2:
                # Inversion bloc lettres / bloc chiffres : `123 FACT` ↔ `FACT123`.
                keys |= number_variants(window[1] + window[0])
    return sorted(keys)


def _label_tokens_of_norm(norm: str) -> list[str]:
    return label_tokens(norm.split())


def _label_numbers_of_norm(norm: str) -> list[str]:
    return label_numbers(norm.split())


def _keys_of_compact(compact: str) -> list[str]:
    if not compact:
        return []
    if compact.isalpha():
        return [compact]
    return sorted(number_variants(compact))


def reference_keys(reference: Any) -> list[str]:
    """Clés d'une référence de facture (triées). Vide si la référence est vide."""
    return _keys_of_compact(normalize_text(reference).replace(" ", ""))


# --- Application aux tables chargées ---------------------------------------


def enrich_label(series: pd.Series, prefix: str = "label", executor: Executor | None = None) -> pd.DataFrame:
    norm = normalize_series(series)
    return pd.DataFrame(
        {
            f"{prefix}_norm": norm,
            f"{prefix}_tokens": _map_unique(norm, _label_tokens_of_norm),
            f"{prefix}_numbers": _map_unique(norm, _label_numbers_of_norm, executor),
        },
        index=series.index,
    )


def enrich_reference(series: pd.Series, prefix: str, executor: Executor | None = None) -> pd.DataFrame:
    norm = normalize_series(series)
    compact = norm.str.replace(" ", "", regex=False)
    return pd.DataFrame(
        {f"{prefix}_norm": norm, f"{prefix}_keys": _map_unique(compact, _keys_of_compact, executor)},
        index=series.index,
    )


def enrich_name(series: pd.Series, prefix: str = "name") -> pd.DataFrame:
    norm = normalize_series(series)
    return pd.DataFrame(
        {f"{prefix}_norm": norm, f"{prefix}_tokens": _map_unique(norm, str.split)},
        index=series.index,
    )


def normalize_iban(series: pd.Series) -> pd.Series:
    """IBAN sans espaces ni séparateurs, en majuscules ; vide → NA."""
    out = normalize_series(series).str.replace(" ", "", regex=False)
    return out.where(out != "", pd.NA).astype("string")
