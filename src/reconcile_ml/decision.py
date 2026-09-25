"""Étape 5 — ensembles et décision (brief §7.4-7.5).

Pour chaque paiement, à partir des candidats scorés (score brut de passe 2 : même ordre que la
probabilité calibrée, qui est monotone, mais sans les plateaux de la régression isotonique) :
- passe « candidat unique » : les deux meilleurs candidats capables d'absorber le paiement ;
- passe « somme proche » : sous-ensembles des 25 meilleurs candidats dont la somme est à la
  tolérance près du paiement (5 € ou 3 %), cardinalité minimale préférée ; score = moyenne des
  scores de paire ;
- la meilleure proposition l'emporte ; marge = écart au score de la seconde.
La passe n↔n (paiements d'un même débiteur agrégés) est dans le rapprocheur : elle ne produit
que des propositions en revue.

Décision : auto-validation si score ≥ τ_high et marge ≥ δ ; revue si score ≥ τ_low ; sinon rien.
τ_high se calibre sur la validation pour la précision cible, globalement et par segment si le
volume le permet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.reconcile_rules.subset import near_subsets
from src.settings import DecisionSettings, SetSettings

PROPOSAL_COLUMNS = ["row", "invoices", "amounts", "score", "margin", "kind"]
MIN_KIND_VOLUME = 20


def propose(scored: pd.DataFrame, amount: np.ndarray, cfg: SetSettings) -> pd.DataFrame:
    """`scored` : (row, inv, balance, p) ; `amount` : montant par ligne. Une proposition par ligne."""
    if scored.empty:
        return pd.DataFrame(columns=PROPOSAL_COLUMNS)
    s = scored.sort_values(["row", "p", "inv"], ascending=[True, False, True], kind="mergesort")
    row, inv = s["row"].to_numpy(), s["inv"].to_numpy()
    bal, p = s["balance"].to_numpy(dtype=np.int64), s["p"].to_numpy(dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
    ends = np.r_[starts[1:], len(row)]
    out = []
    for a, b in zip(starts, ends):
        r = int(row[a])
        amt = int(amount[r])
        tol = max(cfg.tolerance_abs_cents, cfg.tolerance_rel * amt)
        props = []
        # Une facture seule n'est recevable que si elle absorbe le paiement (soldée ou partielle).
        absorbs = amt <= bal[a:b] + tol
        if cfg.single_candidate or not cfg.enabled:
            for j in (a + np.flatnonzero(absorbs))[:2]:
                settle = abs(amt - bal[j]) <= tol
                props.append((p[j], (int(inv[j]),), (int(bal[j]) if settle else amt,),
                              "single" if settle else "partial"))
        # Ensembles recherchés aussi quand la meilleure facture seule laisse un écart : un groupe exact
        # entre alors en concurrence (marge faible → revue plutôt qu'une auto-validation hasardeuse).
        exact_single = absorbs[0] and bal[a] == amt
        if cfg.enabled and cfg.near_sum and b - a >= 2 and not exact_single:
            k = min(b - a, cfg.max_candidates)
            sols, _ = near_subsets(bal[a:a + k], amt, int(tol), max_size=cfg.max_invoices,
                                   node_budget=cfg.node_budget, max_solutions=20)
            ranked = sorted(sols, key=lambda sol: (len(sol), -float(np.mean(p[a:a + k][list(sol)]))))
            exact = [sol for sol in ranked if abs(amt - int(bal[a:a + k][list(sol)].sum())) <= cfg.tolerance_abs_cents]
            for sol in list(dict.fromkeys(ranked[:2] + exact[:1])):
                idx = a + np.array(sol)
                props.append((float(np.mean(p[idx])), tuple(int(i) for i in inv[idx]),
                              tuple(int(x) for x in bal[idx]), "set"))
        if not props:
            continue
        props.sort(key=lambda x: (-x[0], len(x[1])))
        best = props[0]
        others = [q for q in props[1:] if set(q[1]) != set(best[1])]
        margin = best[0] - others[0][0] if others else best[0]
        # Une facture seule qui n'explique le paiement qu'à un écart d'escompte près, face à un ensemble qui
        # l'explique exactement (aux frais près) : ambigu, quel que soit l'écart de score → revue.
        if best[3] == "single" and abs(amt - sum(best[2])) > cfg.tolerance_abs_cents and any(
                q[3] == "set" and abs(amt - sum(q[2])) <= cfg.tolerance_abs_cents for q in props[1:]):
            margin = 0.0
        out.append((r, best[1], best[2], best[0], margin, best[3]))
    return pd.DataFrame(out, columns=PROPOSAL_COLUMNS)


def segment_keys(frame: pd.DataFrame, dims: list[str], edges: list[float] | None) -> np.ndarray:
    """Clé de segment par ligne : valeurs des dimensions jointes (montant découpé en quartiles)."""
    parts = []
    for d in dims:
        if d == "amount_bucket":
            e = edges or []
            parts.append(np.digitize(frame["payment_amount"].to_numpy(dtype=np.float64), e).astype(str))
        elif d in frame.columns:
            parts.append(frame[d].astype(str).to_numpy())
    if not parts:
        return np.full(len(frame), "global", dtype=object)
    key = parts[0].astype(object)
    for p in parts[1:]:
        key = key + "|" + p.astype(object)
    return key


def calibrate_thresholds(proposals: pd.DataFrame, correct: np.ndarray, target: float, cfg: DecisionSettings,
                         amount_edges: list[float] | None = None) -> dict:
    """τ_high : plus petit seuil tenant la précision cible parmi les propositions de marge ≥ δ."""

    def one(scores: np.ndarray, ok: np.ndarray) -> float | None:
        if len(scores) == 0:
            return None
        order = np.argsort(-scores, kind="stable")
        s, c = scores[order], ok[order].astype(np.float64)
        prec = np.cumsum(c) / np.arange(1, len(s) + 1)
        last_of_score = np.r_[s[1:] != s[:-1], True]
        valid = np.flatnonzero((prec >= target) & last_of_score)
        return float(s[valid[-1]]) if len(valid) else None

    eligible = proposals["margin"].to_numpy() >= cfg.min_margin
    scores = proposals["score"].to_numpy(dtype=np.float64)
    global_tau = one(scores[eligible], correct[eligible])
    result = {"tau_high": global_tau if global_tau is not None else 1.01, "tau_low": cfg.review_min_score,
              "min_margin": cfg.min_margin, "kinds": {}, "segments": {}, "segment_dims": [],
              "amount_edges": amount_edges}
    # Un seuil par type de proposition (facture soldée, paiement partiel, ensemble) : leurs fiabilités
    # diffèrent (un partiel peut masquer un n↔n). Sans volume suffisant, pas d'auto-validation.
    kinds = proposals["kind"].to_numpy()
    for kind in np.unique(kinds):
        m = (kinds == kind) & eligible
        tau = one(scores[m], correct[m]) if m.sum() >= MIN_KIND_VOLUME else None
        result["kinds"][str(kind)] = tau if tau is not None else 1.01
    if cfg.segmented_thresholds:
        dims = list(cfg.segments)
        keys = segment_keys(proposals, dims, amount_edges)
        result["segment_dims"] = dims
        for key in np.unique(keys):
            m = (keys == key) & eligible
            if m.sum() >= cfg.min_segment_volume:
                tau = one(scores[m], correct[m])
                result["segments"][str(key)] = tau if tau is not None else 1.01
    return result


def decide(proposals: pd.DataFrame, thresholds: dict) -> np.ndarray:
    """Action par proposition : auto / review / '' (pas de décision)."""
    if proposals.empty:
        return np.array([], dtype=object)
    tau = np.full(len(proposals), thresholds["tau_high"], dtype=np.float64)
    if thresholds.get("kinds"):
        by_kind = proposals["kind"].map(thresholds["kinds"]).to_numpy(dtype=np.float64)
        tau = np.where(np.isnan(by_kind), 1.01, by_kind)
    if thresholds.get("segments"):
        keys = segment_keys(proposals, thresholds["segment_dims"], thresholds.get("amount_edges"))
        seg = pd.Series(keys).map(thresholds["segments"]).to_numpy(dtype=np.float64)
        tau = np.where(np.isnan(seg), tau, seg)
    score, margin = proposals["score"].to_numpy(), proposals["margin"].to_numpy()
    auto = (score >= tau) & (margin >= thresholds["min_margin"])
    review = ~auto & (score >= thresholds["tau_low"])
    return np.where(auto, "auto", np.where(review, "review", ""))


def calibrate_online(records: pd.DataFrame, correct: np.ndarray, target: float, cfg: DecisionSettings,
                     grid: int = 2000) -> dict:
    """Seuils calibrés sur la boucle réelle de validation (propositions quotidiennes enregistrées).

    `records` : une ligne par (jour, paiement) — kind, score, margin — dans l'ordre des jours ;
    `correct` : la proposition du jour est-elle exactement la vérité ? En production, un paiement
    en attente est rescoré chaque jour : il est auto-validé le premier jour où son score franchit τ.
    Pour τ donné, la proposition retenue est donc celle du jour qui porte, pour la première fois,
    le maximum courant du score au-dessus de τ. Calibration par type de proposition.
    """
    result = {"tau_high": 1.01, "tau_low": cfg.review_min_score, "min_margin": cfg.min_margin, "kinds": {},
              "segments": {}, "segment_dims": [], "amount_edges": None, "method": "online"}
    r = records.assign(correct=correct)
    for kind in sorted(r["kind"].unique()):
        k = r[(r["kind"] == kind)].copy()
        k["s"] = np.where(k["margin"] >= cfg.min_margin, k["score"], -np.inf)
        k = k.sort_values(["payment_id", "day"], kind="mergesort")
        prev = k.groupby("payment_id")["s"].transform(lambda v: v.cummax().shift(fill_value=-np.inf)).to_numpy()
        s = k["s"].to_numpy()
        setter = s > prev                                  # jours qui portent un nouveau maximum
        s, prev, ok = s[setter], prev[setter], k["correct"].to_numpy()[setter]
        if len(s) < MIN_KIND_VOLUME:
            result["kinds"][kind] = 1.01
            continue
        taus = np.unique(s[np.isfinite(s)])[::-1]
        if len(taus) > grid:
            taus = taus[np.unique(np.linspace(0, len(taus) - 1, grid).round().astype(int))]
        best = None
        for tau in taus:                                   # du plus strict au plus permissif
            m = (prev < tau) & (tau <= s)
            if m.sum() >= MIN_KIND_VOLUME and ok[m].mean() >= target:
                best = float(tau)
        result["kinds"][kind] = best if best is not None else 1.01
    return result
