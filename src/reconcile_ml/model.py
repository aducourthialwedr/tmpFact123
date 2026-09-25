"""Modèle de scoring des paires (brief §7.2) : LightGBM binaire en deux passes + calibration isotonique.

- Passe 1 : features des familles actives.
- Features de compétition dérivées des scores de passe 1 au sein d'un même paiement : rang,
  marge au meilleur autre candidat, meilleur score concurrent. En entraînement, les scores de
  passe 1 sont obtenus hors échantillon (validation croisée par paiement) pour ne pas biaiser
  la passe 2.
- Passe 2 : features actives + compétition.
- Calibration isotonique des scores de passe 2 sur la période de validation.

Le modèle est sauvegardé avec l'empreinte du journal, la version de la featurisation et les
paramètres, pour être rejouable à l'identique.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from src.reconcile_ml.features import CATEGORICAL, COMPETITION
from src.settings import TrainingSettings

GROUP = "group"      # identifiant du paiement dans le jeu (paiement × jour de scoring)


def competition_features(scores: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    """Rang, marge au meilleur concurrent et meilleur score concurrent, au sein de chaque groupe."""
    df = pd.DataFrame({"g": group, "s": scores})
    order = np.lexsort((-df["s"].to_numpy(), df["g"].to_numpy()))
    g, s = df["g"].to_numpy()[order], df["s"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]) if len(g) else np.array([], dtype=np.int64)
    sizes = np.diff(np.r_[starts, len(g)])
    first = np.repeat(starts, sizes)
    rank = np.arange(len(g)) - first + 1
    best = s[first]
    second = np.where(sizes > 1, s[np.minimum(starts + 1, len(g) - 1)], 0.0)
    second = np.repeat(second, sizes)
    best_other = np.where(rank == 1, second, best)
    out = np.empty((len(g), 3), dtype=np.float32)
    out[order, 0] = rank
    out[order, 1] = s - best_other
    out[order, 2] = best_other
    return pd.DataFrame(out, columns=COMPETITION)


def _params(cfg: TrainingSettings) -> dict:
    return {"objective": "binary", "learning_rate": cfg.learning_rate, "num_leaves": cfg.num_leaves,
            "min_data_in_leaf": cfg.min_data_in_leaf, "feature_fraction": 0.9, "seed": cfg.seed,
            "deterministic": True, "force_row_wise": True, "verbosity": -1, "num_threads": 0}


def _train(X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame | None, yv: np.ndarray | None,
           cfg: TrainingSettings) -> lgb.Booster:
    cats = [c for c in CATEGORICAL if c in X.columns]
    train = lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=True)
    valid = [lgb.Dataset(Xv, label=yv, categorical_feature=cats, reference=train)] if Xv is not None else []
    callbacks = [lgb.early_stopping(30, verbose=False)] if valid else []
    return lgb.train(_params(cfg), train, num_boost_round=cfg.num_boost_round, valid_sets=valid, callbacks=callbacks)


@dataclass
class PairModel:
    features: list[str]
    pass1: lgb.Booster
    pass2: lgb.Booster | None
    calibrator: IsotonicRegression | None
    meta: dict = field(default_factory=dict)

    # --- Scoring -------------------------------------------------------------------------------------

    def raw(self, X: pd.DataFrame, group: np.ndarray) -> np.ndarray:
        s1 = self.pass1.predict(X[self.features], num_threads=0)
        if self.pass2 is None:
            return s1
        comp = competition_features(s1, group)
        X2 = pd.concat([X[self.features].reset_index(drop=True), comp], axis=1)
        return self.pass2.predict(X2, num_threads=0)

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    def predict(self, X: pd.DataFrame, group: np.ndarray) -> np.ndarray:
        """Probabilité calibrée que la paire soit une vraie imputation."""
        return self.calibrate(self.raw(X, group))

    # --- Entraînement ----------------------------------------------------------------------------------

    @classmethod
    def fit(cls, train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: TrainingSettings,
            second_pass: bool = True, calibration: bool = True, folds: int = 3) -> PairModel:
        y, yv = train["label"].to_numpy(), valid["label"].to_numpy()
        X, Xv = train[features], valid[features]
        pass1 = _train(X, y, Xv, yv, cfg)
        pass2 = None
        if second_pass:
            # Scores de passe 1 hors échantillon sur l'entraînement (plis par paiement).
            fold = (pd.util.hash_array(train[GROUP].astype(str).to_numpy()) % folds).astype(int)
            oof = np.zeros(len(train))
            fixed = cfg.model_copy(update={"num_boost_round": max(pass1.best_iteration, 10)})
            for k in range(folds):
                m = _train(X[fold != k], y[fold != k], None, None, fixed)
                oof[fold == k] = m.predict(X[fold == k], num_threads=0)
            X2 = pd.concat([X.reset_index(drop=True), competition_features(oof, train[GROUP].to_numpy())], axis=1)
            s1v = pass1.predict(Xv, num_threads=0)
            Xv2 = pd.concat([Xv.reset_index(drop=True), competition_features(s1v, valid[GROUP].to_numpy())], axis=1)
            pass2 = _train(X2, y, Xv2, yv, cfg)
        model = cls(features, pass1, pass2, None)
        if calibration:
            raw_v = model.raw(valid, valid[GROUP].to_numpy())
            model.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_v, yv)
        return model

    # --- Persistance -------------------------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.pass1.save_model(str(d / "pass1.txt"))
        if self.pass2 is not None:
            self.pass2.save_model(str(d / "pass2.txt"))
        if self.calibrator is not None:
            (d / "calibrator.pkl").write_bytes(pickle.dumps(self.calibrator))
        meta = {**self.meta, "features": self.features}
        (d / "model.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> PairModel:
        d = Path(directory)
        meta = json.loads((d / "model.json").read_text(encoding="utf-8"))
        pass2 = lgb.Booster(model_file=str(d / "pass2.txt")) if (d / "pass2.txt").exists() else None
        cal = pickle.loads((d / "calibrator.pkl").read_bytes()) if (d / "calibrator.pkl").exists() else None
        return cls(meta["features"], lgb.Booster(model_file=str(d / "pass1.txt")), pass2, cal, meta)


def pair_metrics(df: pd.DataFrame, score: np.ndarray) -> dict:
    """Diagnostics du scoring : AUC, precision@1, MRR (sur les groupes ayant au moins un positif)."""
    d = pd.DataFrame({"g": df[GROUP].to_numpy(), "y": df["label"].to_numpy(), "s": score})
    d = d.sort_values(["g", "s"], ascending=[True, False], kind="mergesort")
    d["rank"] = d.groupby("g").cumcount() + 1
    pos = d[d["y"] == 1]
    first = pos.groupby("g")["rank"].min()
    return {
        "auc": float(roc_auc_score(d["y"], d["s"])) if d["y"].nunique() == 2 else None,
        "precision_at_1": float((first == 1).mean()) if len(first) else None,
        "mrr": float((1.0 / first).mean()) if len(first) else None,
        "groupes_avec_positif": int(len(first)),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y})
    out = df.groupby("bin").agg(paires=("y", "size"), score_moyen=("p", "mean"), taux_observé=("y", "mean"))
    out.index = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in out.index]
    return out.rename_axis("tranche").reset_index()


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
