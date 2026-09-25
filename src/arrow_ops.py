"""Recherches par clé vectorisées via pyarrow.

Sur les colonnes de chaînes pyarrow, `Series.isin` et `Series.map(Series)` de
pandas repassent par des objets Python : ~10 s sur 2 M de lignes, contre
< 1 s ici. À utiliser pour toute jointure par identifiant à grande échelle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


def to_arrow(values: pd.Series | pd.Index | np.ndarray | list) -> pa.Array:
    if isinstance(values, (pd.Series, pd.Index)):
        values = values.array
    arr = pa.array(values, from_pandas=True)
    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr


def isin(values: pd.Series, reference: pd.Series) -> np.ndarray:
    """Masque booléen : valeur présente dans `reference` (NA → False)."""
    value_set = to_arrow(pd.Series(reference).dropna().drop_duplicates())
    mask = pc.is_in(to_arrow(values), value_set=value_set).fill_null(False)
    return mask.to_numpy(zero_copy_only=False)


def lookup(keys: pd.Series, index_keys: pd.Series, values: pd.Series) -> pd.Series:
    """Pour chaque clé, la valeur de `values` à la première position où `index_keys` vaut la clé.

    Clé absente ou manquante → NA. Résultat aligné sur l'index de `keys`.
    """
    first = ~pd.Series(index_keys).duplicated().to_numpy()
    idx_keys = pd.Series(index_keys)[first]
    vals = pd.Series(values)[first].reset_index(drop=True)
    pos = pc.index_in(to_arrow(keys), value_set=to_arrow(idx_keys)).fill_null(-1)
    pos = pos.to_numpy(zero_copy_only=False)
    found = pos >= 0
    out = vals.iloc[np.where(found, pos, 0)] if len(vals) else pd.Series([pd.NA] * len(pos))
    out = out.reset_index(drop=True).where(found)
    out.index = keys.index
    return out
