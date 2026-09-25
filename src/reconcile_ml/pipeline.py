"""Étape 5 — rapprocheur complet (règles puis ML) et construction / entraînement (brief §7).

- `PipelineMatcher` : étape 4 (règles), puis sur le résiduel du jour : candidats, features,
  scoring, ensembles, n↔n (revue), décision par seuils, arbitrage du lot. Les réservations du
  moteur sont partagées entre règles et ML.
- `DatasetRecorder` : même boucle, mais enregistre les features des candidats de chaque paiement
  du résiduel à son jour d'arrivée (état avant les événements du jour), sans décider en ML.
- `fit_ml` : rejoue entraînement + validation, étiquette dans une seconde passe séparée à partir
  des imputations, entraîne, calibre, fixe τ_high et sauvegarde le modèle avec l'empreinte du
  journal et la version de la featurisation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import ground_truth
from src.load.interim import load_interim
from src.reconcile_ml.decision import calibrate_online, calibrate_thresholds, decide, propose
from src.reconcile_ml.features import FEATURIZATION_VERSION, Featurizer, active_features
from src.reconcile_ml.model import GROUP, PairModel, calibration_table, fingerprint, pair_metrics
from src.reconcile_rules.matcher import RulesMatcher
from src.reconcile_rules.subset import near_subsets
from src.settings import RulesConfig, Settings
from src.timeline.loop import AUTO, DECISION_COLUMNS, REVIEW, DayContext, run_replay
from src.timeline.split import assign_period, compute_split
from src.timeline.state import LedgerState, _days

MODEL_DIRNAME = "pair_model"


def _sampled(payment_ids: np.ndarray, share: float) -> np.ndarray:
    """Échantillon déterministe de paiements (hachage de l'identifiant)."""
    if share >= 1:
        return np.ones(len(payment_ids), dtype=bool)
    h = pd.util.hash_array(np.asarray(payment_ids, dtype=object)) % 10_000
    return h < share * 10_000


class _ResidualMixin(RulesMatcher):
    """Candidats et features du résiduel des règles."""

    def _setup_ml(self, settings: Settings, categories: dict | None = None) -> None:
        self.ml = settings.reconcile_ml
        self.featurizer = Featurizer(self.state, self.allocator, self.ml, self.min_key_length, categories)
        self._pay_value = _days(self.state.table("payment")["value_date"])

    def _residual(self, ctx: DayContext, rules_decisions: pd.DataFrame, rows_filter=None):
        batch_ids = ctx.batch["payment_id"].astype(object).to_numpy()
        pos = self.state.pay_pos(pd.Series(batch_ids, dtype=object))
        decided = np.isin(batch_ids, rules_decisions["payment_id"].astype(object).to_numpy())
        keep = ~decided if rows_filter is None else (~decided & rows_filter)
        rows = np.flatnonzero(keep)
        alloc = self.last_allocation
        scope = self._scopes(alloc, batch_ids)[0]
        cands, cited = self.featurizer.candidates(rows, pos, alloc, scope, ctx.as_of, self._claimed)
        X = self.featurizer.features(cands, pos, alloc, batch_ids, ctx.as_of, cited) if len(cands) else None
        return batch_ids, pos, alloc, cands, X


class DatasetRecorder(_ResidualMixin):
    """Rejoue les règles et enregistre les paires candidates du résiduel (jour d'arrivée)."""

    name = "dataset"

    def __init__(self, state: LedgerState, settings: Settings, rules: RulesConfig):
        super().__init__(state, settings, rules)
        self._setup_ml(settings)
        self.features = active_features(self.ml)
        self._frames: list[pd.DataFrame] = []

    def process(self, ctx: DayContext) -> pd.DataFrame:
        decisions = super().process(ctx)
        ids = ctx.batch["payment_id"].astype(object).to_numpy()
        rows_filter = ctx.batch["is_new"].to_numpy() & _sampled(ids, self.ml.training.payment_sample)
        batch_ids, pos, alloc, cands, X = self._residual(ctx, decisions, rows_filter)
        if X is not None and len(X):
            frame = X[self.features].copy()
            frame["payment_id"] = batch_ids[cands["row"].to_numpy()]
            frame["invoice_id"] = self._inv_ids[cands["inv"].to_numpy()]
            frame["balance"] = cands["balance"].to_numpy()
            frame["payment_amount"] = self._pay_amount[pos[cands["row"].to_numpy()]]
            frame["day"] = ctx.day
            self._frames.append(frame)
        return decisions

    def dataset(self) -> pd.DataFrame:
        return pd.concat(self._frames, ignore_index=True) if self._frames else pd.DataFrame()


class PipelineMatcher(_ResidualMixin):
    """Étapes 4 puis 5 dans la boucle quotidienne."""

    name = "pipeline"

    def __init__(self, state: LedgerState, settings: Settings, rules: RulesConfig, model: PairModel,
                 record_proposals: bool = False):
        """`record_proposals` : mode calibration — aucune auto-validation ML, propositions quotidiennes
        enregistrées (`daily_proposals`) pour calibrer les seuils sur la boucle réelle."""
        super().__init__(state, settings, rules)
        self._setup_ml(settings, model.meta.get("categories"))
        self.model = model
        self.record_proposals = record_proposals
        self.thresholds = model.meta.get("thresholds") if not record_proposals else             {"tau_high": 2.0, "tau_low": 0.0, "min_margin": 0.0, "kinds": {}, "segments": {}}
        self._daily: list[pd.DataFrame] = []
        self._diag: list[pd.DataFrame] = []
        self._info: list[pd.DataFrame] = []
        self._seen: set[str] = set()

    def process(self, ctx: DayContext) -> pd.DataFrame:
        rules_decisions = super().process(ctx)
        batch_ids, pos, alloc, cands, X = self._residual(ctx, rules_decisions)
        self._record_info(ctx, alloc, batch_ids)
        if X is None or not len(X):
            return rules_decisions
        amount = self._pay_amount[pos]
        raw = self.model.raw(X, cands["row"].to_numpy())
        scored = cands.assign(p=raw, p_cal=self.model.calibrate(raw))
        self._record_diag(scored, batch_ids)

        props = propose(scored, amount, self.ml.sets)
        props = self._segment_columns(props, pos, alloc, X, cands)
        if self.record_proposals and len(props):
            self._daily.append(pd.DataFrame({
                "day": ctx.day, "payment_id": batch_ids[props["row"].to_numpy()], "kind": props["kind"].to_numpy(),
                "score": props["score"].to_numpy(), "margin": props["margin"].to_numpy(),
                "invoice_ids": [tuple(sorted(self._inv_ids[list(i)])) for i in props["invoices"]]}))
        actions = decide(props, self.thresholds) if len(props) else np.array([], dtype=object)
        props = props.assign(action=actions)
        props = props[props["action"] != ""]
        props = self._arbitrate_ml(props, ctx.as_of)
        nn = self._n_to_n(scored, props, alloc, pos, amount) if self.ml.sets.enabled and self.ml.sets.n_to_n \
            else pd.DataFrame()
        ml_decisions = self._to_decisions(props, nn, batch_ids)
        return pd.concat([rules_decisions, ml_decisions], ignore_index=True) if len(ml_decisions) else rules_decisions

    # --- Étapes ------------------------------------------------------------------------------------------

    def _segment_columns(self, props, pos, alloc, X, cands) -> pd.DataFrame:
        if props.empty:
            return props
        rows = props["row"].to_numpy()
        first_inv = np.array([inv[0] for inv in props["invoices"]], dtype=np.int64)
        market = pd.Series(self.featurizer.inv_market[first_inv]).astype(int).astype(str).to_numpy()
        files = alloc.payments["client_file_id"].to_numpy(dtype=object)
        return props.assign(payment_amount=self._pay_amount[pos[rows]], market=market,
                            has_client_file=pd.notna(files[rows]).astype(int),
                            bankroll_code=self.featurizer.pay_bankroll[pos[rows]].astype(int))

    def _arbitrate_ml(self, props: pd.DataFrame, as_of) -> pd.DataFrame:
        """Auto-validations ML par score décroissant ; une facture déjà soldée ramène la proposition en revue."""
        autos = props[props["action"] == AUTO].sort_values("score", ascending=False, kind="mergesort")
        if autos.empty:
            return props
        invs = np.unique(np.concatenate([np.array(i, dtype=np.int64) for i in autos["invoices"]]))
        remaining = dict(zip(invs.tolist(), (self.state.open_balance_at(invs, as_of) - self._claimed[invs]).tolist()))
        downgrade = []
        for idx, inv, amt in zip(autos.index, autos["invoices"], autos["amounts"]):
            if all(remaining[i] >= a for i, a in zip(inv, amt)):
                for i, a in zip(inv, amt):
                    remaining[i] -= a
            else:
                downgrade.append(idx)
        props = props.copy()
        props.loc[downgrade, "action"] = REVIEW
        return props

    def _n_to_n(self, scored: pd.DataFrame, props: pd.DataFrame, alloc, pos, amount) -> pd.DataFrame:
        """Paiements non auto-validés d'un même débiteur (allocation ferme) proches dans le temps : revue."""
        cfg = self.ml.sets
        firm = alloc.payments["firm_debtor_id"].to_numpy(dtype=object)
        auto_rows = set(props.loc[props["action"] == AUTO, "row"].tolist())
        rows = np.array([r for r in np.unique(scored["row"].to_numpy()) if r not in auto_rows and pd.notna(firm[r])],
                        dtype=np.int64)
        if len(rows) < 2:
            return pd.DataFrame()
        frame = pd.DataFrame({"row": rows, "debtor": firm[rows], "day": self._pay_value[pos[rows]]})
        window = max(cfg.n_to_n_window_hours // 24, 1)
        out = []
        for _, g in frame.groupby("debtor"):
            if len(g) < 2 or g["day"].max() - g["day"].min() > window:
                continue
            pool = scored[scored["row"].isin(g["row"])].drop_duplicates("inv")
            pool = pool.sort_values("p", ascending=False).head(cfg.max_candidates)
            target = int(amount[g["row"].to_numpy()].sum())
            tol = int(max(cfg.tolerance_abs_cents, cfg.tolerance_rel * target))
            sols, _ = near_subsets(pool["balance"].to_numpy(), target, tol, max_size=cfg.max_invoices * 2,
                                   min_size=2, node_budget=cfg.node_budget, max_solutions=2)
            if len(sols) != 1:
                continue
            sel = pool.iloc[list(sols[0])]
            for r in g["row"]:
                out.append({"row": int(r), "invoices": tuple(sel["inv"].tolist()), "score": float(sel["p"].mean())})
        return pd.DataFrame(out)

    def _to_decisions(self, props: pd.DataFrame, nn: pd.DataFrame, batch_ids: np.ndarray) -> pd.DataFrame:
        rows = []
        for r, inv, amt, score, action, kind in zip(props["row"], props["invoices"], props["amounts"], props["score"],
                                                   props["action"], props["kind"]):
            for i, a in zip(inv, amt):
                rows.append((batch_ids[r], self._inv_ids[i], a, action, "ml", f"ML_{kind.upper()}", score))
            if action == AUTO:
                pid = int(self.state.pay_pos(pd.Series([batch_ids[r]], dtype=object))[0])
                self._claims.setdefault(pid, []).extend(zip(inv, amt))
                np.add.at(self._claimed, np.array(inv, dtype=np.int64), np.array(amt, dtype=np.int64))
        done = {r for r, a in zip(props["row"], props["action"]) if a == AUTO}
        for rec in nn.to_dict("records") if len(nn) else []:
            if rec["row"] in done:
                continue
            for i in rec["invoices"]:
                rows.append((batch_ids[rec["row"]], self._inv_ids[i], None, REVIEW, "ml", "ML_NN", rec["score"]))
        if not rows:
            return pd.DataFrame(columns=DECISION_COLUMNS)
        df = pd.DataFrame(rows, columns=["payment_id", "invoice_id", "amount", "action", "step", "rule_id", "score"])
        df["amount"] = pd.array(df["amount"], dtype="Int64")
        df["rule_version"] = pd.array([None] * len(df), dtype="Int64")
        return df[DECISION_COLUMNS]

    # --- Diagnostics pour l'évaluation -----------------------------------------------------------------------

    def _record_info(self, ctx, alloc, batch_ids) -> None:
        new = ctx.batch["is_new"].to_numpy()
        self._info.append(pd.DataFrame({"payment_id": batch_ids[new],
                                        "client_file_id": alloc.payments["client_file_id"].to_numpy()[new],
                                        "allocation_status": alloc.payments["status"].to_numpy()[new]}))

    def _record_diag(self, scored: pd.DataFrame, batch_ids: np.ndarray) -> None:
        """Candidats et scores du premier passage en ML de chaque paiement."""
        s = scored.assign(payment_id=batch_ids[scored["row"].to_numpy()])
        s = s[~s["payment_id"].isin(self._seen)]
        if s.empty:
            return
        self._seen.update(s["payment_id"].unique().tolist())
        self._diag.append(pd.DataFrame({"payment_id": s["payment_id"].to_numpy(),
                                        "invoice_id": self._inv_ids[s["inv"].to_numpy()], "p": s["p_cal"].to_numpy(),
                                        "raw": s["p"].to_numpy()}))

    def daily_proposals(self) -> pd.DataFrame:
        return pd.concat(self._daily, ignore_index=True) if self._daily else pd.DataFrame()

    def side_outputs(self) -> dict[str, pd.DataFrame]:
        out = {"proposals": self.proposals()}
        if self._diag:
            out["ml_candidates"] = pd.concat(self._diag, ignore_index=True)
        if self._info:
            out["payment_info"] = pd.concat(self._info, ignore_index=True).drop_duplicates("payment_id")
        return out


# --- Entraînement ----------------------------------------------------------------------------------------------


def label_pairs(dataset: pd.DataFrame, imputation: pd.DataFrame) -> np.ndarray:
    """Seconde passe, séparée du rejeu : la paire figure-t-elle dans les imputations ?"""
    truth = imputation[["payment_id", "invoice_id"]].drop_duplicates().assign(_hit=1)
    merged = dataset[["payment_id", "invoice_id"]].merge(truth, on=["payment_id", "invoice_id"], how="left")
    return merged["_hit"].fillna(0).to_numpy(dtype=np.int8)


def candidate_recall(dataset: pd.DataFrame, truth: pd.DataFrame) -> float | None:
    """Part des paiements du jeu dont toutes les factures réellement imputées sont candidates."""
    t = truth.set_index("payment_id")["truth_invoices"]
    got = dataset.groupby("payment_id")["invoice_id"].agg(set)
    common = got.index.intersection(t.index)
    if not len(common):
        return None
    return float(np.mean([set(t[p]) <= got[p] for p in common]))


def offline_proposals(frame: pd.DataFrame, scores: np.ndarray, settings: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """Propositions reconstruites hors boucle sur un jeu étiqueté (calibration des seuils)."""
    pay_codes, pay_ids = pd.factorize(frame["payment_id"])
    inv_codes, inv_ids = pd.factorize(frame["invoice_id"])
    scored = pd.DataFrame({"row": pay_codes, "inv": inv_codes, "balance": frame["balance"].to_numpy(), "p": scores})
    amount = frame.groupby(pay_codes)["payment_amount"].first().to_numpy()
    props = propose(scored, amount, settings.reconcile_ml.sets)
    props["payment_id"] = pay_ids[props["row"].to_numpy()]
    props["invoice_ids"] = [tuple(sorted(inv_ids[list(i)])) for i in props["invoices"]]
    first = frame.groupby(pay_codes).first()
    props["payment_amount"] = amount[props["row"].to_numpy()]
    for col in ("market", "has_client_file", "bankroll_code"):
        props[col] = first[col].to_numpy()[props["row"].to_numpy()].astype(int).astype(str) if col in first else "0"
    return props, pd.Series(inv_ids)


def fit_ml(interim_dir: Path, model_dir: Path, settings: Settings, rules: RulesConfig,
           log: Callable[[str], None] = print) -> dict:
    """Rejoue entraînement + validation, entraîne le modèle, calibre les seuils, sauvegarde."""
    timings = {}
    t = time.perf_counter()
    data, journal, meta = load_interim(interim_dir)
    state = LedgerState(data, journal, settings.reconcile_ml.features.behavioral_window_days)
    first, last = state.payment_day_range()
    split = compute_split(settings.split, first.date(), last.date())
    train_p, valid_p = split.period("train"), split.period("validation")
    timings["préparation"] = round(time.perf_counter() - t, 1)

    t = time.perf_counter()
    recorder = DatasetRecorder(state, settings, rules)
    log(f"… rejeu {train_p.start} → {valid_p.end} pour construire le jeu d'entraînement")
    run_replay(state, recorder, train_p.start, valid_p.end, settings.split.retention_days,
               on_day=lambda ctx, row: log(f"  {ctx.day.date()}") if ctx.day.day == 1 else None)
    ds = recorder.dataset()
    timings["construction du jeu"] = round(time.perf_counter() - t, 1)
    if ds.empty:
        raise RuntimeError("jeu d'entraînement vide")

    ds["label"] = label_pairs(ds, data.tables["imputation"])
    ds[GROUP] = ds["payment_id"]
    ds["period"] = assign_period(ds["day"], split).to_numpy()
    ds["month"] = pd.to_datetime(ds["day"]).dt.strftime("%Y-%m")
    out_dir = interim_dir / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out_dir / "dataset", partition_cols=["month"], index=False)
    train, valid = ds[ds["period"] == "train"], ds[ds["period"] == "validation"]
    log(f"  jeu : {len(train):,} paires d'entraînement, {len(valid):,} de validation".replace(",", " "))

    t = time.perf_counter()
    features = recorder.features
    model = PairModel.fit(train, valid, features, settings.reconcile_ml.training,
                          settings.reconcile_ml.second_pass, settings.reconcile_ml.calibration)
    timings["entraînement"] = round(time.perf_counter() - t, 1)

    raw_valid = model.raw(valid, valid[GROUP].to_numpy())
    p_valid = model.calibrate(raw_valid)
    truth = ground_truth(data.tables["imputation"])
    props, _ = offline_proposals(valid, raw_valid, settings)
    t_map = truth.set_index("payment_id")["truth_invoices"]
    correct = np.array([t_map.get(pid) == inv for pid, inv in zip(props["payment_id"], props["invoice_ids"])])
    edges = list(np.quantile(valid.groupby("payment_id")["payment_amount"].first(), [0.25, 0.5, 0.75]))
    offline = calibrate_thresholds(props, correct, settings.evaluation.target_precision,
                                   settings.reconcile_ml.decision, edges)
    auto = pd.Series(decide(props, offline) == AUTO)

    # Calibration sur la boucle réelle de validation : propositions quotidiennes, sans auto-validation ML.
    t = time.perf_counter()
    model.meta = {"thresholds": offline, "categories": recorder.featurizer.categories}
    thresholds = online_thresholds(data, journal, model, settings, rules, valid_p, truth, log, out_dir)
    thresholds["offline"] = {k: offline[k] for k in ("tau_high", "kinds")}
    timings["calibration en ligne"] = round(time.perf_counter() - t, 1)
    metrics = {
        "validation": pair_metrics(valid, p_valid),
        "rappel_candidats_validation": candidate_recall(valid, truth),
        "résiduel_validation_paiements": int(valid["payment_id"].nunique()),
        "auto_validation": int(auto.sum()),
        "précision_auto_validation": float(correct[auto.to_numpy()].mean()) if auto.any() else None,
    }
    model.meta = {
        "journal_sha256": meta["journal_sha256"], "featurization_version": FEATURIZATION_VERSION,
        "rules_version": rules.version, "settings_fingerprint": fingerprint(settings.reconcile_ml.model_dump()),
        "settings": settings.reconcile_ml.model_dump(), "periods": {
            "train": [str(train_p.start), str(train_p.end)], "validation": [str(valid_p.start), str(valid_p.end)]},
        "thresholds": thresholds, "metrics": metrics, "timings_s": timings,
        "categories": recorder.featurizer.categories,
    }
    model.save(model_dir)
    calibration_table(valid["label"].to_numpy(), p_valid).to_csv(model_dir / "calibration_validation.csv", index=False)
    importance = pd.DataFrame({"feature": model.pass1.feature_name(),
                               "gain": model.pass1.feature_importance("gain")}).sort_values("gain", ascending=False)
    importance.to_csv(model_dir / "feature_importance.csv", index=False)
    return model.meta


def online_thresholds(data, journal, model: PairModel, settings: Settings, rules: RulesConfig, valid_p, truth,
                      log: Callable[[str], None], out_dir: Path) -> dict:
    """Rejoue la validation sans auto-validation ML, enregistre les propositions quotidiennes et en déduit τ_high
    par type de proposition (précision réelle de la boucle, rescorage quotidien compris)."""
    log(f"… rejeu de la validation pour calibrer les seuils ({valid_p.start} → {valid_p.end})")
    state = LedgerState(data, journal, settings.reconcile_ml.features.behavioral_window_days)
    probe = PipelineMatcher(state, settings, rules, model, record_proposals=True)
    run_replay(state, probe, valid_p.start, valid_p.end, settings.split.retention_days)
    daily = probe.daily_proposals()
    if daily.empty:
        return dict(model.meta.get("thresholds") or {})
    t_map = truth.set_index("payment_id")["truth_invoices"]
    ok = np.array([t_map.get(pid) == inv for pid, inv in zip(daily["payment_id"], daily["invoice_ids"])])
    daily.assign(correct=ok, invoice_ids=daily["invoice_ids"].map(list)).to_parquet(
        out_dir / "validation_daily_proposals.parquet", index=False)
    return calibrate_online(daily, ok, settings.evaluation.target_precision, settings.reconcile_ml.decision)


def recalibrate_ml(interim_dir: Path, model_dir: Path, settings: Settings, rules: RulesConfig,
                   log: Callable[[str], None] = print) -> dict:
    """Recalcule les seuils d'un modèle existant (après un changement de décision ou de cible), sans réentraîner."""
    data, journal, meta = load_interim(interim_dir)
    model = PairModel.load(model_dir)
    if model.meta.get("journal_sha256") != meta["journal_sha256"]:
        raise RuntimeError("modèle entraîné sur un autre journal : réentraîner")
    state = LedgerState(data, journal, settings.reconcile_ml.features.behavioral_window_days)
    first, last = state.payment_day_range()
    valid_p = compute_split(settings.split, first.date(), last.date()).period("validation")
    t = time.perf_counter()
    offline = model.meta["thresholds"].get("offline", {})
    thresholds = online_thresholds(data, journal, model, settings, rules, valid_p, ground_truth(data.tables["imputation"]),
                                   log, interim_dir / "ml")
    thresholds["offline"] = offline
    model.meta["thresholds"] = thresholds
    model.meta.setdefault("timings_s", {})["recalibration"] = round(time.perf_counter() - t, 1)
    model.save(model_dir)
    return model.meta
