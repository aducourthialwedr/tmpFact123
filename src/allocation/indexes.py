"""Index de l'allocation, construits une fois sur les attributs statiques.

Aucun index ne porte d'information temporelle : ils listent toutes les factures,
tous les débiteurs, tous les IBAN connus du référentiel. Ce qui dépend du temps
— la facture existe-t-elle à D, le débiteur est-il connu à D, combien de
débiteurs partagent une clé à D — est filtré à la requête par l'état daté.
Aucune recherche ne parcourt tous les débiteurs : on part toujours du paiement.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np
import pandas as pd


def _flatten(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Colonne de listes → (longueurs, valeurs aplaties en objets)."""
    arr = values.to_numpy() if isinstance(values, pd.Series) else values
    lengths = np.fromiter((0 if v is None or (isinstance(v, float)) else len(v) for v in arr),
                          dtype=np.int64, count=len(arr))
    flat = np.fromiter(chain.from_iterable(v for v in arr if v is not None and not isinstance(v, float)),
                       dtype=object, count=int(lengths.sum()))
    return lengths, flat


@dataclass
class Postings:
    """Listes d'occurrences compactes (CSR) : clé entière → valeurs entières."""

    offsets: np.ndarray
    values: np.ndarray

    @classmethod
    def build(cls, keys: np.ndarray, values: np.ndarray, n_keys: int) -> Postings:
        ok = (keys >= 0) & (values >= 0)
        keys, values = keys[ok], values[ok]
        order = np.lexsort((values, keys))
        keys, values = keys[order], values[order]
        # Déduplication (clé, valeur).
        keep = np.ones(len(keys), dtype=bool)
        keep[1:] = (keys[1:] != keys[:-1]) | (values[1:] != values[:-1])
        keys, values = keys[keep], values[keep]
        offsets = np.searchsorted(keys, np.arange(n_keys + 1))
        return cls(offsets, values)

    @classmethod
    def from_lists(cls, lengths: np.ndarray, flat_ids: np.ndarray) -> Postings:
        """Propriétaire i → ses ids (issus d'une colonne de listes aplatie)."""
        owners = np.repeat(np.arange(len(lengths)), lengths)
        return cls.build(owners, flat_ids, len(lengths))

    def lengths(self, keys: np.ndarray) -> np.ndarray:
        return self.offsets[keys + 1] - self.offsets[keys]

    def gather(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pour chaque clé demandée, ses valeurs : (index de la clé dans `keys`, valeur)."""
        keys = np.asarray(keys, dtype=np.int64)
        lengths = self.lengths(keys)
        total = int(lengths.sum())
        if total == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        owner = np.repeat(np.arange(len(keys)), lengths)
        starts = np.repeat(self.offsets[keys] - (np.cumsum(lengths) - lengths), lengths)
        return owner, self.values[starts + np.arange(total)]


def _hash(values: np.ndarray) -> np.ndarray:
    """Empreintes 64 bits déterministes de chaînes (valeurs nulles : 0, jamais dans un vocabulaire)."""
    values = np.asarray(values, dtype=object)
    if len(values) == 0:
        return np.array([], dtype=np.uint64)
    return pd.util.hash_array(values, categorize=False)


class Vocabulary:
    """Chaînes → identifiants entiers (−1 si inconnue).

    Stocke les empreintes 64 bits triées des chaînes, pas les chaînes : quelques octets par entrée
    au lieu d'un objet Python et d'une table de hachage pandas. Une collision entre deux clés
    distinctes (probabilité ~ n² / 2⁶⁵, soit 10⁻⁵ pour 30 millions de clés) les confondrait.
    """

    def __init__(self, values: np.ndarray):
        values = np.asarray(values, dtype=object)
        valid = ~pd.isna(values)
        hashes = _hash(values[valid])
        self.hashes, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
        self.codes = np.full(len(values), -1, dtype=np.int64)
        self.codes[valid] = inverse
        kept = values[valid][first]
        self.lengths = np.fromiter((len(v) for v in kept), dtype=np.int64, count=len(kept))

    def __len__(self) -> int:
        return len(self.hashes)

    def lookup(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=object)
        out = np.full(len(values), -1, dtype=np.int64)
        if len(values) == 0 or len(self.hashes) == 0:
            return out
        valid = ~pd.isna(values)
        h = _hash(values[valid])
        # Requêtes triées : la recherche dichotomique parcourt alors le vocabulaire dans l'ordre (cache).
        order = np.argsort(h, kind="stable")
        i = np.empty(len(h), dtype=np.int64)
        i[order] = np.searchsorted(self.hashes, h[order])
        i = np.minimum(i, len(self.hashes) - 1)
        out[valid] = np.where(self.hashes[i] == h, i, -1)
        return out


# Les listes de clés / termes des libellés sont traduites par blocs : pas de copie en objets Python
# de tous les libellés de l'historique à la fois.
_PAYMENT_CHUNK = 200_000


def _payment_postings(n: int, ids_of_chunk) -> Postings:
    """Paiement → identifiants, construits par blocs de `_PAYMENT_CHUNK` paiements.

    `ids_of_chunk(start, end)` rend (propriétaire relatif au bloc, identifiant) ; −1 = hors vocabulaire.
    """
    owners, ids = [np.array([], dtype=np.int64)], [np.array([], dtype=np.int64)]
    for start in range(0, n, _PAYMENT_CHUNK):
        o, i = ids_of_chunk(start, min(start + _PAYMENT_CHUNK, n))
        known = i >= 0
        owners.append(o[known] + start)
        ids.append(i[known])
    return Postings.build(np.concatenate(owners), np.concatenate(ids), n)


class ReferenceIndex:
    """Clé de référence → factures ; paiement → clés de son libellé."""

    def __init__(self, invoices: pd.DataFrame, payments: pd.DataFrame):
        parts = []
        for col in ("client_reference_keys", "internal_reference_keys"):
            lengths, flat = _flatten(invoices[col])
            parts.append((np.repeat(np.arange(len(invoices)), lengths), flat))
        inv_pos = np.concatenate([p[0] for p in parts])
        flat = np.concatenate([p[1] for p in parts])
        self.vocab = Vocabulary(flat)
        self.key_invoices = Postings.build(self.vocab.codes, inv_pos, len(self.vocab))
        numbers = payments["label_numbers"]

        def keys_of(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
            lengths, flat = _flatten(numbers.iloc[start:end])
            return np.repeat(np.arange(end - start), lengths), self.vocab.lookup(flat)

        self.payment_keys = _payment_postings(len(payments), keys_of)

    def key_ids(self, keys) -> np.ndarray:
        return self.vocab.lookup(np.asarray(list(keys), dtype=object))


def name_terms(tokens: list[str] | np.ndarray, min_length: int) -> list[str]:
    """Termes indexés d'un nom ou d'un libellé : mots assez longs + bigrammes de mots adjacents."""
    words = [t for t in tokens if len(t) >= min_length and t.isalpha()]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def terms_by_owner(texts: pd.Series, min_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Version vectorisée de `name_terms` sur une colonne de textes normalisés : (propriétaire, terme)."""
    tokens = pd.Series(texts.to_numpy(), dtype=object).fillna("").astype(str).str.split().explode().dropna()
    words = tokens.astype(str)
    keep = (words.str.len() >= min_length).to_numpy() & words.str.isalpha().to_numpy()
    owner = tokens.index.to_numpy(dtype=np.int64)[keep]
    word = words.to_numpy(dtype=object)[keep]
    same = owner[1:] == owner[:-1]
    bigram = (word[:-1] + " " + word[1:])[same] if len(word) > 1 else np.array([], dtype=object)
    return np.concatenate([owner, owner[:-1][same]]), np.concatenate([word, bigram])


class NameIndex:
    """Terme → débiteurs ; débiteur → termes ; paiement → termes de son libellé."""

    def __init__(self, debtors: pd.DataFrame, payments: pd.DataFrame, min_length: int):
        # Toutes les variantes de nom d'un débiteur (lignes multiples en source), sinon son nom.
        variants = debtors["name_variants"] if "name_variants" in debtors.columns \
            else debtors["name_norm"].map(lambda n: [n])
        exploded = pd.Series([v if len(v) else [""] for v in variants]).explode()
        owners, terms = terms_by_owner(exploded.reset_index(drop=True), min_length)
        owners = exploded.index.to_numpy(dtype=np.int64)[owners]
        self.vocab = Vocabulary(terms)
        self.term_debtors = Postings.build(self.vocab.codes, owners, len(self.vocab))
        self.debtor_terms = Postings.build(owners, self.vocab.codes, len(debtors))
        # Termes du libellé, restreints au vocabulaire des noms.
        labels = payments["label_norm"]

        def terms_of(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
            o, t = terms_by_owner(labels.iloc[start:end].reset_index(drop=True), min_length)
            return o, self.vocab.lookup(t)

        self.payment_terms = _payment_postings(len(payments), terms_of)


def party_ibans(debtors: pd.DataFrame, assignors: pd.DataFrame, party_iban: pd.DataFrame | None) -> pd.DataFrame:
    """Couples (rôle, partie, IBAN, bankroll) : table `party_iban` du chargement, sinon colonnes des parties."""
    if party_iban is not None:
        return party_iban
    parts = [df.loc[df["iban"].notna(), ["party_id", "iban", "bankroll_code"]].assign(role=role)
             for role, df in (("assignor", assignors), ("debtor", debtors))]
    return pd.concat(parts, ignore_index=True)


class IbanIndex:
    """IBAN → débiteurs, cédants (tous leurs comptes) ; bankroll par (IBAN, partie) ; comptes techniques."""

    def __init__(self, debtors: pd.DataFrame, assignors: pd.DataFrame, technical: pd.DataFrame | None,
                 party_iban: pd.DataFrame | None = None):
        links = party_ibans(debtors, assignors, party_iban)
        tech = technical["iban"].dropna() if technical is not None else pd.Series([], dtype=object)
        self.vocab = Vocabulary(pd.concat([links["iban"], tech], ignore_index=True).astype(object).to_numpy())
        self.bankroll: dict[str, pd.DataFrame] = {}
        for role, parties, attr in (("debtor", debtors, "debtors"), ("assignor", assignors, "assignors")):
            rows = links[links["role"] == role]
            pos = pd.Index(parties["party_id"].astype(object)).get_indexer(rows["party_id"].astype(object))
            ids = self.vocab.lookup(rows["iban"].astype(object).to_numpy())
            setattr(self, attr, Postings.build(ids, pos.astype(np.int64), len(self.vocab)))
            self.bankroll[role] = pd.DataFrame({"ib": ids, "pos": pos, "br": rows["bankroll_code"].to_numpy()}) \
                .query("ib >= 0 and pos >= 0").drop_duplicates()
        self.technical = np.zeros(len(self.vocab), dtype=bool)
        self.technical[self.vocab.lookup(tech.astype(object).to_numpy())] = True
