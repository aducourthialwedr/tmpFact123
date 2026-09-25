"""Mesure de l'allocation (brief §5, §8) : rappel, précision de l'allocation ferme.

La sonde `AllocationProbe` se branche dans la boucle quotidienne comme un
rapprocheur qui ne décide rien : elle enregistre l'allocation de chaque paiement
à son premier passage (jour d'arrivée) et à son dernier passage dans le lot.

Vérité : le débiteur des factures réellement imputées au paiement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.allocation.allocator import Allocator, FIRM, NONE, SIGNALS
from src.evaluation.metrics import _tuples_by_key
from src.timeline.loop import DayContext, empty_decisions

# Le dernier passage de chaque paiement est dédoublonné tous les N jours : sans cela, les lots successifs
# (reliquat compris) s'accumulent en mémoire.
_COMPACT_EVERY = 5


class AllocationProbe:
    name = "allocation"

    def __init__(self, allocator: Allocator):
        self.allocator = allocator
        self._first: list[pd.DataFrame] = []
        self._last: list[pd.DataFrame] = []

    def process(self, ctx: DayContext) -> pd.DataFrame:
        alloc = self.allocator.allocate(ctx)
        summary = alloc.payments.copy()
        cand = alloc.candidates
        keys, debtors = _tuples_by_key(cand["payment_id"].to_numpy(dtype=object),
                                       cand["debtor_id"].to_numpy(dtype=object))
        _, signals = _tuples_by_key(cand["payment_id"].to_numpy(dtype=object), cand["signals"].to_numpy(dtype=object))
        lists = pd.DataFrame({"payment_id": keys, "candidates": debtors, "candidate_signals": signals})
        summary = summary.merge(lists, on="payment_id", how="left")
        empty = summary["candidates"].isna()
        summary.loc[empty, "candidates"] = pd.Series([()] * int(empty.sum()), index=summary.index[empty], dtype=object)
        summary.loc[empty, "candidate_signals"] = pd.Series([()] * int(empty.sum()), index=summary.index[empty],
                                                            dtype=object)
        summary["day"] = ctx.day
        self._first.append(summary[ctx.batch["is_new"].to_numpy()])
        self._last.append(summary)
        if len(self._last) >= _COMPACT_EVERY:
            self._last = [pd.concat(self._last, ignore_index=True).drop_duplicates("payment_id", keep="last")]
        return empty_decisions()

    def first_pass(self) -> pd.DataFrame:
        return pd.concat(self._first, ignore_index=True) if self._first else pd.DataFrame()

    def last_pass(self) -> pd.DataFrame:
        if not self._last:
            return pd.DataFrame()
        return pd.concat(self._last, ignore_index=True).drop_duplicates("payment_id", keep="last")


def truth_debtors(imputation: pd.DataFrame, invoice: pd.DataFrame) -> pd.DataFrame:
    """Par paiement imputé : débiteurs des factures imputées (tuple trié)."""
    debtor_of = invoice[["invoice_id", "debtor_id"]].drop_duplicates("invoice_id")
    pairs = (imputation[["payment_id", "invoice_id"]].merge(debtor_of, on="invoice_id")
             [["payment_id", "debtor_id"]].dropna().drop_duplicates().sort_values(["payment_id", "debtor_id"]))
    keys, tuples = _tuples_by_key(pairs["payment_id"].to_numpy(dtype=object), pairs["debtor_id"].to_numpy(dtype=object))
    return pd.DataFrame({"payment_id": keys, "truth_debtors": tuples})


def _found(candidates, truth) -> bool:
    return isinstance(truth, tuple) and bool(truth) and set(truth) <= set(candidates)


def allocation_metrics(first: pd.DataFrame, last: pd.DataFrame, truth: pd.DataFrame, target_recall: float,
                       payments: pd.DataFrame | None = None) -> dict:
    """Métriques sur les paiements arrivés dans la période (premier passage) ayant une imputation réelle."""
    p = first.merge(truth, on="payment_id", how="left")
    p = p.merge(last[["payment_id", "candidates", "status", "client_file_id"]]
                .rename(columns={"candidates": "candidates_last", "status": "status_last",
                                 "client_file_id": "client_file_last"}), on="payment_id", how="left")
    has_truth = p["truth_debtors"].map(lambda t: isinstance(t, tuple))
    p["found_first"] = [_found(c, t) for c, t in zip(p["candidates"], p["truth_debtors"])]
    p["found_last"] = [_found(c, t) for c, t in zip(p["candidates_last"], p["truth_debtors"])]
    single = p["truth_debtors"].map(lambda t: isinstance(t, tuple) and len(t) == 1)
    p["top1"] = [s and len(c) > 0 and c[0] == t[0] for s, c, t in zip(single, p["candidates"], p["truth_debtors"])]
    p["firm"] = p["status"] == FIRM
    p["firm_correct"] = [f and s and fd == t[0] for f, s, fd, t in
                         zip(p["firm"], single, p["firm_debtor_id"], p["truth_debtors"])]

    def found_by(c, sigs, t) -> str:
        if not _found(c, t) or len(t) != 1:
            return "non trouvé" if not _found(c, t) else "plusieurs débiteurs"
        return sigs[list(c).index(t[0])]

    p["found_by"] = [found_by(c, s, t) for c, s, t in zip(p["candidates"], p["candidate_signals"], p["truth_debtors"])]
    q = p[has_truth]
    n = len(q)
    firm = q["firm"].sum()
    recall = q["found_first"].mean() if n else float("nan")
    summary = {
        "paiements_périmètre": len(p),
        "avec_imputation_réelle": n,
        "rappel_premier_passage": recall,
        "rappel_dernier_passage": q["found_last"].mean() if n else float("nan"),
        "rappel_cible": target_recall,
        "cible_atteinte": bool(n and recall >= target_recall),
        "top1": q["top1"].mean() if n else float("nan"),
        "taux_ferme": firm / n if n else float("nan"),
        "précision_ferme": q["firm_correct"].sum() / firm if firm else float("nan"),
        "sans_candidat": (q["status"] == NONE).mean() if n else float("nan"),
        "candidats_moyens": q["candidates"].map(len).mean() if n else float("nan"),
        "avec_client_file": q["client_file_last"].notna().mean() if n else float("nan"),
    }

    def rate_table(key: str) -> pd.DataFrame:
        g = q.groupby(key, dropna=False)
        out = pd.DataFrame({"paiements": g.size(), "rappel": g["found_first"].mean(),
                            "rappel_dernier_passage": g["found_last"].mean(), "top1": g["top1"].mean(),
                            "taux_ferme": g["firm"].mean(),
                            "précision_ferme": g["firm_correct"].sum() / g["firm"].sum().where(g["firm"].sum() > 0)})
        out = out.reset_index()
        out["part"] = out["paiements"] / n if n else 0.0
        return out

    by_route = rate_table("iban_route")
    by_status = rate_table("status")
    found = q["found_by"].value_counts().rename_axis("signal").reset_index(name="paiements")
    found["part"] = found["paiements"] / n if n else 0.0
    misses = q[~q["found_first"]].head(200)
    misses = misses[["payment_id", "iban_route", "status", "candidates", "truth_debtors"]].copy()
    if payments is not None:
        misses = misses.merge(payments[["payment_id", "label"]], on="payment_id", how="left")
    for col in ("candidates", "truth_debtors"):
        misses[col] = misses[col].map(lambda v: " ".join(v[:5]) if isinstance(v, tuple) else "")
    return {"summary": summary, "by_route": by_route, "by_status": by_status, "found_by": found,
            "misses": misses, "detail": p}


def signal_order() -> list[str]:
    return list(SIGNALS)
