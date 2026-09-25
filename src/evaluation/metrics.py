"""Évaluation des décisions d'un rejeu contre les imputations réellement prononcées (brief §8).

Unité de mesure : le paiement. Un paiement auto-validé est **correct** si
l'ensemble des factures décidées est exactement celui des factures réellement
imputées à ce paiement (toutes dates confondues). Un paiement sans imputation
réelle qui est auto-validé compte comme une erreur.

Périmètre : les paiements arrivés pendant la période évaluée. Les décisions sur
des paiements arrivés avant (reliquat de démarrage) sont ignorées.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.timeline.loop import AUTO, REVIEW

GROUP_TYPES = ("1↔1", "1↔n", "n↔1", "n↔n", "sans imputation")
_CURVE_POINTS = 200


@dataclass
class Evaluation:
    summary: dict
    by_step: pd.DataFrame
    by_rule: pd.DataFrame
    by_group: pd.DataFrame
    by_month: pd.DataFrame
    curve: pd.DataFrame
    cascade: pd.DataFrame
    payments: pd.DataFrame = field(repr=False)   # détail par paiement du périmètre

    def tables(self) -> dict[str, pd.DataFrame]:
        return {"by_step": self.by_step, "by_rule": self.by_rule, "by_group": self.by_group,
                "by_month": self.by_month, "curve": self.curve, "cascade": self.cascade}


def _tuples_by_key(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
    """Regroupe `values` par `keys` (déjà triés par clé) en tuples, sans groupby Python par groupe."""
    if len(keys) == 0:
        return keys, []
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    return keys[starts], [tuple(values[a:b]) for a, b in zip(starts, ends)]


def ground_truth(imputation: pd.DataFrame) -> pd.DataFrame:
    """Par paiement imputé : factures réellement imputées (tuple trié) et type de groupe.

    Le type se lit sur la composante connexe du graphe paiements–factures :
    1↔1, 1↔n (un paiement, plusieurs factures), n↔1, n↔n.
    """
    pairs = imputation[["payment_id", "invoice_id"]].dropna().drop_duplicates()
    pay_codes, pay_ids = pd.factorize(pairs["payment_id"])
    inv_codes, _ = pd.factorize(pairs["invoice_id"])
    n_pay, n_inv = len(pay_ids), inv_codes.max() + 1 if len(inv_codes) else 0
    graph = coo_matrix((np.ones(len(pairs)), (pay_codes, n_pay + inv_codes)), shape=(n_pay + n_inv,) * 2)
    _, component = connected_components(graph, directed=False)
    comp_pay = np.bincount(component[:n_pay], minlength=component.max() + 1 if len(component) else 0)
    comp_inv = np.bincount(component[n_pay:], minlength=len(comp_pay))
    k_pay, k_inv = comp_pay[component[:n_pay]], comp_inv[component[:n_pay]]
    group_type = np.select([(k_pay == 1) & (k_inv == 1), k_pay == 1, k_inv == 1], ["1↔1", "1↔n", "n↔1"],
                           default="n↔n")
    ordered = pairs.sort_values(["payment_id", "invoice_id"])
    keys, tuples = _tuples_by_key(ordered["payment_id"].to_numpy(dtype=object),
                                  ordered["invoice_id"].to_numpy(dtype=object))
    truth = pd.DataFrame({"payment_id": keys, "truth_invoices": tuples})
    truth["n_invoices"] = [len(t) for t in tuples]
    types = pd.Series(group_type, index=pd.Index(pay_ids.astype(object)))
    truth["group_type"] = types.reindex(truth["payment_id"]).to_numpy()
    return truth


def _proposals(decisions: pd.DataFrame) -> pd.DataFrame:
    """Une proposition par paiement : la première auto-validation, sinon la première mise en revue."""
    cols = ["payment_id", "action", "day", "invoices", "score", "step", "rule_id", "rule_version"]
    d = decisions[decisions["action"].isin([AUTO, REVIEW])]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d.assign(auto=d["action"] == AUTO).dropna(subset=["invoice_id"])
    d = d.drop_duplicates(["payment_id", "day", "auto", "invoice_id"])
    d = d.sort_values(["payment_id", "day", "auto", "invoice_id"]).reset_index(drop=True)
    group = d.groupby(["payment_id", "day", "auto"], sort=False)
    grouped = group.agg(score=("score", "max"), step=("step", "first"), rule_id=("rule_id", "first"),
                        rule_version=("rule_version", "first")).reset_index()
    _, grouped["invoices"] = _tuples_by_key(group.ngroup().to_numpy(), d["invoice_id"].to_numpy(dtype=object))
    # Priorité à l'auto-validation (la plus précoce), sinon la revue la plus précoce.
    grouped = grouped.sort_values(["payment_id", "auto", "day"], ascending=[True, False, True])
    first = grouped.drop_duplicates("payment_id").copy()
    first["action"] = np.where(first["auto"], AUTO, REVIEW)
    return first[cols].reset_index(drop=True)


def _rate_table(df: pd.DataFrame, key: str, in_scope: int | None = None) -> pd.DataFrame:
    """Par valeur de `key` : paiements, auto-validés, corrects, précision, taux d'automatisation."""
    g = df.groupby(key, dropna=False, sort=True)
    out = pd.DataFrame({
        "paiements": g.size(),
        "auto": g["is_auto"].sum(),
        "corrects": g["correct"].sum(),
    }).reset_index()
    out["précision"] = out["corrects"] / out["auto"].where(out["auto"] > 0)
    denominator = in_scope if in_scope is not None else out["paiements"]
    out["taux_automatisation"] = out["auto"] / denominator
    return out


def precision_curve(scores: np.ndarray, correct: np.ndarray, in_scope: int, base_n: int = 0,
                    base_correct: int = 0) -> pd.DataFrame:
    """Courbe automatisation / précision en auto-validant les propositions par score décroissant.

    `base_n` / `base_correct` : décisions acquises quel que soit le seuil (règles de l'étape 4).
    """
    if in_scope == 0 or (len(scores) == 0 and base_n == 0):
        return pd.DataFrame(columns=["seuil", "taux_automatisation", "précision"])
    order = np.argsort(-scores, kind="stable")
    s, c = scores[order], correct[order].astype(np.float64)
    n = base_n + np.arange(1, len(s) + 1)
    prec = (base_correct + np.cumsum(c)) / n
    # Un seuil ne peut couper qu'entre deux scores différents.
    last_of_score = np.r_[s[1:] != s[:-1], True] if len(s) else np.array([], dtype=bool)
    curve = pd.DataFrame({"seuil": s, "taux_automatisation": n / in_scope, "précision": prec})[last_of_score]
    if base_n:
        start = pd.DataFrame({"seuil": [np.inf], "taux_automatisation": [base_n / in_scope],
                              "précision": [base_correct / base_n]})
        curve = pd.concat([start, curve], ignore_index=True)
    if len(curve) > _CURVE_POINTS:
        idx = np.unique(np.linspace(0, len(curve) - 1, _CURVE_POINTS).round().astype(int))
        curve = curve.iloc[idx]
    return curve.reset_index(drop=True)


def automation_at_precision(curve: pd.DataFrame, target: float) -> float:
    ok = curve[curve["précision"] >= target]
    return float(ok["taux_automatisation"].max()) if len(ok) else 0.0


def evaluate(decisions: pd.DataFrame, truth: pd.DataFrame, scope: pd.DataFrame, target_precision: float,
             current_automation_rate: float | None = None) -> Evaluation:
    """`scope` : paiements du périmètre (colonnes payment_id, arrival_day)."""
    props = _proposals(decisions)
    p = (scope[["payment_id", "arrival_day"]]
         .merge(truth, on="payment_id", how="left")
         .merge(props, on="payment_id", how="left"))
    p["group_type"] = p["group_type"].fillna("sans imputation")
    p["is_auto"] = p["action"] == AUTO
    p["is_review"] = p["action"] == REVIEW
    has_truth = p["truth_invoices"].notna()
    same = [isinstance(a, tuple) and isinstance(b, tuple) and a == b
            for a, b in zip(p["invoices"], p["truth_invoices"])]
    p["correct_proposal"] = has_truth & p["action"].notna() & pd.Series(same, index=p.index)
    p["correct"] = p["is_auto"] & p["correct_proposal"]
    p["month"] = pd.to_datetime(p["arrival_day"]).dt.strftime("%Y-%m")
    n = len(p)
    auto = int(p["is_auto"].sum())
    correct = int(p["correct"].sum())

    # Précision / rappel au niveau des paires (paiement, facture) auto-validées.
    decided = {(pid, i) for pid, inv, a in zip(p["payment_id"], p["invoices"], p["is_auto"]) if a for i in inv}
    real = {(pid, i) for pid, inv in zip(p["payment_id"], p["truth_invoices"]) if isinstance(inv, tuple) for i in inv}
    hits = len(decided & real)

    # Les décisions des règles sont acquises ; le seuil ne porte que sur les autres propositions.
    base = p["is_auto"] & (p["step"] == "rules")
    proposed = p[p["action"].notna() & ~base]
    curve = precision_curve(proposed["score"].fillna(0).to_numpy(dtype=np.float64),
                            proposed["correct_proposal"].to_numpy(), n, int(base.sum()),
                            int(p.loc[base, "correct"].sum()))
    at_target = automation_at_precision(curve, target_precision)
    rate = auto / n if n else 0.0
    precision = correct / auto if auto else float("nan")

    summary = {
        "paiements_périmètre": n,
        "avec_imputation_réelle": int(has_truth.sum()),
        "sans_imputation_réelle": int((~has_truth).sum()),
        "auto": auto, "auto_corrects": correct, "revue": int(p["is_review"].sum()),
        "taux_automatisation": rate,
        "précision": precision,
        "taux_automatisation_correct": correct / n if n else 0.0,
        "précision_cible": target_precision,
        "cible_atteinte": bool(auto == 0 or precision >= target_precision),
        "taux_automatisation_à_précision_cible": at_target,
        "précision_paires": hits / len(decided) if decided else float("nan"),
        "rappel_paires": hits / len(real) if real else float("nan"),
        "taux_actuel_référence": current_automation_rate,
    }

    rules_auto = p["is_auto"] & (p["step"] == "rules")
    rules_correct = rules_auto & p["correct"]
    cascade = pd.DataFrame([
        {"niveau": "Algorithme actuel", "taux_automatisation": current_automation_rate, "précision": None},
        {"niveau": "Étape 4 — règles seules", "taux_automatisation": rules_auto.sum() / n if n else 0.0,
         "précision": rules_correct.sum() / rules_auto.sum() if rules_auto.sum() else None},
        {"niveau": "Étapes 4 + 5 — règles puis ML", "taux_automatisation": rate,
         "précision": precision if auto else None},
    ])
    ref = current_automation_rate
    cascade["gain_points"] = [None] + [None if ref is None else 100 * (r - ref)
                                       for r in cascade["taux_automatisation"].iloc[1:]]

    auto_rows = p[p["is_auto"]]
    by_step = _rate_table(auto_rows.assign(step=auto_rows["step"].fillna("?")), "step", n) if auto else \
        pd.DataFrame(columns=["step", "paiements", "auto", "corrects", "précision", "taux_automatisation"])
    by_rule = _rate_table(auto_rows.assign(rule=auto_rows["rule_id"].fillna("—")), "rule", n) if auto else \
        pd.DataFrame(columns=["rule", "paiements", "auto", "corrects", "précision", "taux_automatisation"])
    by_group = _rate_table(p, "group_type")
    by_group["group_type"] = pd.Categorical(by_group["group_type"], categories=GROUP_TYPES, ordered=True)
    by_group = by_group.sort_values("group_type").reset_index(drop=True)
    by_group["group_type"] = by_group["group_type"].astype(str)
    by_month = _rate_table(p, "month")

    detail = p[["payment_id", "arrival_day", "group_type", "action", "day", "step", "rule_id", "score",
                "correct"]].copy()
    return Evaluation(summary, by_step, by_rule, by_group, by_month, curve, cascade, detail)


def ml_diagnostics(candidates: pd.DataFrame, truth: pd.DataFrame, scope: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Diagnostics du scoring sur les paiements passés en ML : rappel des candidats, precision@1, MRR, calibration.

    `candidates` : (payment_id, invoice_id, p) au premier passage en ML de chaque paiement.
    """
    c = candidates[candidates["payment_id"].isin(set(scope["payment_id"]))]
    t = truth.set_index("payment_id")["truth_invoices"]
    c = c[c["payment_id"].isin(t.index)]
    if c.empty:
        return {}, pd.DataFrame()
    pairs = {(pid, inv) for pid, invs in t.items() for inv in invs}
    c = c.assign(label=[(pid, inv) in pairs for pid, inv in zip(c["payment_id"], c["invoice_id"])])
    c = c.sort_values(["payment_id", "p"], ascending=[True, False], kind="mergesort")
    c["rank"] = c.groupby("payment_id").cumcount() + 1
    got = c.groupby("payment_id")["invoice_id"].agg(set)
    recall = np.mean([set(t[pid]) <= got[pid] for pid in got.index])
    first = c[c["label"]].groupby("payment_id")["rank"].min().reindex(got.index)
    edges = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(c["p"].to_numpy(), edges[1:-1]), 0, 9)
    calib = c.assign(bin=b).groupby("bin").agg(paires=("label", "size"), score_moyen=("p", "mean"),
                                               taux_observé=("label", "mean")).reset_index()
    calib["tranche"] = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in calib["bin"]]
    return {
        "paiements_en_ml": int(len(got)),
        "rappel_candidats": float(recall),
        "precision_at_1": float((first == 1).mean()),
        "mrr": float((1.0 / first).fillna(0).mean()),
    }, calib[["tranche", "paires", "score_moyen", "taux_observé"]]


def by_flag(evaluation_payments: pd.DataFrame, flag: pd.Series, name: str) -> pd.DataFrame:
    """Taux d'automatisation et précision selon un indicateur par paiement (ex. client file rattaché)."""
    p = evaluation_payments.assign(**{name: evaluation_payments["payment_id"].map(flag).fillna(False).astype(bool)})
    g = p.groupby(name)
    out = pd.DataFrame({"paiements": g.size(), "auto": g["action"].apply(lambda a: (a == AUTO).sum()),
                        "corrects": g["correct"].sum()}).reset_index()
    out["précision"] = out["corrects"] / out["auto"].where(out["auto"] > 0)
    out["taux_automatisation"] = out["auto"] / out["paiements"]
    return out
