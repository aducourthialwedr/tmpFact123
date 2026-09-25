"""Point d'entrée unique de la pipeline, pour le notebook et l'interface.

    from src.api import Project
    project = Project(dataset="synthetic")          # ou "real" (sources de config/schema.yaml)
    project.load(n_payments=50_000)                 # étape 1
    project.split()                                 # étape 2 : périodes
    project.measure_allocation("validation")        # étape 3
    project.backtest("test", matcher="rules")       # étape 4 : baseline
    project.train()                                 # étape 5 : modèle
    project.backtest("test", matcher="pipeline")    # étapes 4 + 5 → étape 6
    project.report("pipeline", "test")              # résultats

Chaque méthode écrit ses sorties sur disque (paths de config/settings.yaml) et renvoie des
objets affichables (DataFrame, dict). Les paramètres se lisent dans config/settings.yaml et
config/rules.yaml, éditables à la main ou depuis l'interface.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DEFAULT_SCHEMA_PATH, DEFAULT_SETTINGS_PATH, REPO_ROOT, read_yaml, resolve_path
from src.settings import DEFAULT_RULES_PATH, RulesConfig, Settings, load_rules, load_settings

SYNTHETIC, REAL = "synthetic", "real"
PERIOD_NAMES = ("train", "validation", "test", "all")
MATCHERS = ("null", "rules", "pipeline")


def _stderr(message: str) -> None:
    try:
        print(message, file=sys.stderr, flush=True)
    except UnicodeEncodeError:                  # console Windows non UTF-8
        print(message.encode("ascii", "replace").decode(), file=sys.stderr, flush=True)


def _clean(value: Any) -> Any:
    return None if isinstance(value, float) and math.isnan(value) else value


@dataclass
class Project:
    """Un jeu de données (synthétique ou réel) et ses fichiers de configuration."""

    dataset: str = SYNTHETIC
    settings_path: Path = DEFAULT_SETTINGS_PATH
    schema_path: Path = DEFAULT_SCHEMA_PATH
    rules_path: Path = DEFAULT_RULES_PATH
    log: Callable[[str], None] = field(default=_stderr, repr=False)

    def __post_init__(self) -> None:
        if self.dataset not in (SYNTHETIC, REAL):
            raise ValueError(f"dataset : {SYNTHETIC!r} ou {REAL!r}")
        self.settings_path, self.schema_path, self.rules_path = (
            Path(self.settings_path), Path(self.schema_path), Path(self.rules_path))

    # --- Configuration et chemins -------------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        return load_settings(self.settings_path)

    @property
    def rules(self) -> RulesConfig:
        return load_rules(self.rules_path)

    def _dir(self, base: str) -> Path:
        root = resolve_path(base)
        return root / "synthetic" if self.dataset == SYNTHETIC else root

    @property
    def interim_dir(self) -> Path:
        return self._dir(self.settings.paths.interim_dir)

    @property
    def reports_dir(self) -> Path:
        return self._dir(self.settings.paths.reports_dir)

    @property
    def model_dir(self) -> Path:
        from src.reconcile_ml.pipeline import MODEL_DIRNAME
        return resolve_path(self.settings.paths.models_dir) / self.dataset / MODEL_DIRNAME

    def meta(self) -> dict | None:
        path = self.interim_dir / "journal_meta.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @contextmanager
    def _step(self, name: str, timings: dict[str, float]):
        start = time.perf_counter()
        self.log(f"… {name}")
        yield
        timings[name] = round(time.perf_counter() - start, 1)
        self.log(f"  {name} : {timings[name]} s")

    # --- Étape 1 : chargement ---------------------------------------------------------------------------------

    def load(self, n_payments: int | None = None, seed: int = 42, regenerate: bool = False) -> dict:
        """Charge les sources (ou génère le jeu synthétique), normalise, construit le journal et le profil."""
        from src.load.events import build_journal, derive_imputed_amounts, journal_hash
        from src.load.loader import load_all
        from src.load.normalize import NORMALIZATION_VERSION
        from src.load.profile import build_profile, write_reports
        from src.load.quality import check_quality

        settings = self.settings
        timings: dict[str, float] = {}
        if self.dataset == SYNTHETIC:
            from src.synthetic.generate import SyntheticConfig, write_synthetic
            cfg = SyntheticConfig(seed=seed)
            if n_payments:
                cfg = replace(cfg, n_payments=n_payments)
            # Un répertoire par variante : changer de volume n'écrase pas les autres jeux.
            source_dir = resolve_path(settings.paths.synthetic_dir) / f"{cfg.n_payments}_seed{cfg.seed}"
            if regenerate or not (source_dir / "payment.csv").exists():
                with self._step(f"génération synthétique ({cfg.n_payments:,} paiements visés)", timings):
                    write_synthetic(cfg, source_dir)
            schema_cfg = read_yaml(REPO_ROOT / "config" / "schema.synthetic.yaml")
            source = f"synthetic/{source_dir.name}"
        else:
            schema_cfg = read_yaml(self.schema_path)
            source_dir = resolve_path(schema_cfg.get("base_dir") or ".")
            source = str(self.schema_path)

        with self._step("chargement et normalisation", timings):
            data = load_all(schema_cfg, source_dir, workers=settings.load.workers)
        with self._step("journal", timings):
            journal, journal_issues = build_journal(data)
        with self._step("contrôles qualité", timings):
            derived = derive_imputed_amounts(data.tables["imputation"], data.tables["invoice"])
            issues = data.issues + journal_issues + check_quality(data, derived)
        interim = self.interim_dir
        with self._step("écriture parquet", timings):
            interim.mkdir(parents=True, exist_ok=True)
            for name, df in data.tables.items():
                df.to_parquet(interim / f"{name}.parquet", index=False)
            journal.to_parquet(interim / "journal.parquet", index=False)
        with self._step("empreinte du journal", timings):
            digest = journal_hash(journal)
        meta = {"normalization_version": NORMALIZATION_VERSION, "journal_sha256": digest,
                "journal_events": len(journal), "source": source}
        (interim / "journal_meta.json").write_text(
            json.dumps({**meta, "mapped_fields": data.mapped_fields, "timings_s": timings}, indent=2),
            encoding="utf-8")
        profile = build_profile(data, journal, issues)
        write_reports(profile, meta, self.reports_dir)
        return {"meta": {**meta, "timings_s": timings}, **profile}

    # --- Étape 2 : temps ----------------------------------------------------------------------------------------

    def _state(self):
        from src.load.interim import load_interim
        from src.timeline.state import LedgerState
        data, journal, meta = load_interim(self.interim_dir)
        state = LedgerState(data, journal, self.settings.reconcile_ml.features.behavioral_window_days)
        return data, state, meta

    def split(self):
        """Périodes train / validation / test sur la plage des paiements chargés."""
        from src.timeline.split import compute_split
        pay = pd.read_parquet(self.interim_dir / "payment.parquet", columns=["value_date", "booking_date"])
        days = pay["booking_date"].fillna(pay["value_date"])
        return compute_split(self.settings.split, days.min().date(), days.max().date())

    def _period(self, state, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        from src.timeline.split import compute_split
        first, last = state.payment_day_range()
        if period == "all":
            return first, last
        p = compute_split(self.settings.split, first.date(), last.date()).period(period)
        return pd.Timestamp(p.start), pd.Timestamp(p.end)

    def _matcher(self, name: str, state, settings: Settings):
        from src.timeline.loop import NullMatcher
        if name == "null":
            return NullMatcher()
        if name == "rules":
            from src.reconcile_rules.matcher import RulesMatcher
            return RulesMatcher(state, settings, self.rules)
        if name == "pipeline":
            from src.reconcile_ml.model import PairModel
            from src.reconcile_ml.pipeline import PipelineMatcher
            if not (self.model_dir / "model.json").exists():
                raise FileNotFoundError("aucun modèle entraîné : lancer d'abord project.train()")
            return PipelineMatcher(state, settings, self.rules, PairModel.load(self.model_dir))
        raise ValueError(f"rapprocheur inconnu : {name} ({', '.join(MATCHERS)})")

    def replay(self, period: str = "test", matcher: str = "null") -> dict:
        """Rejoue la période jour par jour avec un rapprocheur ; écrit décisions et statistiques quotidiennes."""
        from src.timeline.loop import run_replay
        settings, timings = self.settings, {}
        with self._step("lecture étape 1 et état", timings):
            data, state, meta = self._state()
        with self._step("préparation du rapprocheur", timings):
            m = self._matcher(matcher, state, settings)
        start, end = self._period(state, period)
        self.log(f"… rejeu {period} du {start.date()} au {end.date()} ({matcher})")

        def progress(ctx, row):
            if ctx.day.day == 1 or ctx.day == end:
                self.log(f"  {ctx.day.date()} : lot {row['batch']:,} (nouveaux {row['new']:,})".replace(",", " "))

        result = run_replay(state, m, start.date(), end.date(), settings.split.retention_days, on_day=progress)
        timings["rejeu"] = result.seconds
        tag = f"replay_{matcher}_{period}"
        out, rep = self.interim_dir / "replay", self.reports_dir
        out.mkdir(parents=True, exist_ok=True)
        rep.mkdir(parents=True, exist_ok=True)
        result.decisions.to_parquet(out / f"{tag}_decisions.parquet", index=False)
        result.daily.to_csv(rep / f"{tag}_daily.csv", index=False)
        side = m.side_outputs() if hasattr(m, "side_outputs") else (
            {"proposals": m.proposals()} if hasattr(m, "proposals") else {})
        for name, frame in side.items():
            if "invoices" in frame.columns:
                frame = frame.assign(invoices=frame["invoices"].map(list))
            frame.to_parquet(out / f"{tag}_{name}.parquet", index=False)
        summary = {"matcher": matcher, "period": period, "start": str(start.date()), "end": str(end.date()),
                   "days": len(result.daily), "journal_sha256": meta["journal_sha256"],
                   "batch_rows": int(result.daily["batch"].sum()), "new_payments": int(result.daily["new"].sum()),
                   "expired": int(result.daily["expired"].sum()), "auto_payments": int(result.daily["auto"].sum()),
                   "decisions": len(result.decisions), "timings_s": timings}
        (rep / f"{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"summary": summary, "daily": result.daily, "decisions": result.decisions}

    # --- Étape 3 : allocation --------------------------------------------------------------------------------------

    def measure_allocation(self, period: str = "validation") -> dict:
        """Rejoue l'allocation sur la période et mesure le rappel (cible du brief : 99 %)."""
        from src.allocation.allocator import Allocator
        from src.allocation.evaluate import AllocationProbe, allocation_metrics, truth_debtors
        from src.timeline.loop import run_replay
        settings, timings = self.settings, {}
        with self._step("lecture étape 1 et état", timings):
            data, state, meta = self._state()
        with self._step("construction des index", timings):
            probe = AllocationProbe(Allocator(state, settings.allocation))
        start, end = self._period(state, period)
        self.log(f"… allocation {period} du {start.date()} au {end.date()}")
        result = run_replay(state, probe, start.date(), end.date(), settings.split.retention_days,
                            on_day=lambda ctx, row: self.log(f"  {ctx.day.date()}") if ctx.day.day == 1 else None)
        timings["rejeu"] = result.seconds
        truth = truth_debtors(data.tables["imputation"], data.tables["invoice"])
        metrics = allocation_metrics(probe.first_pass(), probe.last_pass(), truth, settings.allocation.target_recall,
                                     data.tables["payment"][["payment_id", "label"]])
        tag, rep = f"allocation_{period}", self.reports_dir
        rep.mkdir(parents=True, exist_ok=True)
        for name in ("by_route", "by_status", "found_by", "misses"):
            metrics[name].to_csv(rep / f"{tag}_{name}.csv", index=False)
        summary = {k: _clean(v) for k, v in metrics["summary"].items()}
        payload = {"context": {"period": period, "start": str(start.date()), "end": str(end.date()),
                               "journal_sha256": meta["journal_sha256"], "settings": settings.allocation.model_dump(),
                               "timings_s": timings}, "summary": summary}
        (rep / f"{tag}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
        return {"summary": summary, **{k: metrics[k] for k in ("by_route", "by_status", "found_by", "misses")}}

    # --- Étape 5 : apprentissage ----------------------------------------------------------------------------------

    def train(self) -> dict:
        """Rejoue entraînement + validation, entraîne le modèle, calibre les seuils, sauvegarde."""
        from src.reconcile_ml.pipeline import fit_ml
        return fit_ml(self.interim_dir, self.model_dir, self.settings, self.rules, log=self.log)

    def calibrate(self) -> dict:
        """Recalcule les seuils de décision du modèle existant sur la validation, sans réentraîner."""
        from src.reconcile_ml.pipeline import recalibrate_ml
        return recalibrate_ml(self.interim_dir, self.model_dir, self.settings, self.rules, log=self.log)

    # --- Étape 6 : évaluation -----------------------------------------------------------------------------------------

    def evaluate(self, period: str = "test", matcher: str = "null") -> dict:
        """Compare les décisions d'un rejeu aux imputations réelles ; écrit les rapports."""
        from src.evaluation.metrics import by_flag, evaluate, ground_truth, ml_diagnostics
        from src.evaluation.report import render_markdown
        settings = self.settings
        tag = f"replay_{matcher}_{period}"
        replay_dir, rep = self.interim_dir / "replay", self.reports_dir
        context_path = rep / f"{tag}.json"
        if not context_path.exists():
            raise FileNotFoundError(f"aucun rejeu {matcher}/{period} : lancer d'abord project.replay()")
        context = json.loads(context_path.read_text(encoding="utf-8"))
        if context["journal_sha256"] != self.meta()["journal_sha256"]:
            raise RuntimeError("le rejeu a été calculé sur un autre journal : le relancer")
        imputation = pd.read_parquet(self.interim_dir / "imputation.parquet", columns=["payment_id", "invoice_id"])
        journal = pd.read_parquet(self.interim_dir / "journal.parquet", columns=["ts", "event_type", "entity_id"])
        start = pd.Timestamp(context["start"])
        end = pd.Timestamp(context["end"]) + pd.Timedelta(days=1)
        arrivals = journal[(journal["event_type"] == "PAYMENT_RECEIVED") & (journal["ts"] >= start)
                           & (journal["ts"] < end)]
        scope = pd.DataFrame({"payment_id": arrivals["entity_id"].to_numpy(),
                              "arrival_day": arrivals["ts"].dt.normalize().to_numpy()})
        truth = ground_truth(imputation)
        result = evaluate(pd.read_parquet(replay_dir / f"{tag}_decisions.parquet"), truth, scope,
                          settings.evaluation.target_precision, settings.evaluation.current_automation_rate)
        tables = result.tables()
        if (replay_dir / f"{tag}_proposals.parquet").exists():
            from src.reconcile_rules.matcher import rule_alone_metrics
            proposals = pd.read_parquet(replay_dir / f"{tag}_proposals.parquet")
            proposals["invoices"] = proposals["invoices"].map(tuple)
            tables["by_rule_alone"] = rule_alone_metrics(proposals, truth, scope, self.rules.rules)
        if (replay_dir / f"{tag}_ml_candidates.parquet").exists():
            diag, calib = ml_diagnostics(pd.read_parquet(replay_dir / f"{tag}_ml_candidates.parquet"), truth, scope)
            result.summary.update({f"ml_{k}": v for k, v in diag.items()})
            tables["ml_calibration"] = calib
        if (replay_dir / f"{tag}_payment_info.parquet").exists():
            info = pd.read_parquet(replay_dir / f"{tag}_payment_info.parquet")
            tables["by_client_file"] = by_flag(result.payments, info.set_index("payment_id")["client_file_id"].notna(),
                                               "client_file")
        summary = {k: _clean(v) for k, v in result.summary.items()}
        out = f"evaluation_{matcher}_{period}"
        for name, df in tables.items():
            df.to_csv(rep / f"{out}_{name}.csv", index=False)
        (rep / f"{out}.json").write_text(json.dumps({"context": context, "summary": summary}, indent=2,
                                                    ensure_ascii=False, default=lambda v: None), encoding="utf-8")
        (rep / f"{out}.md").write_text(render_markdown(summary, tables, context), encoding="utf-8")
        return {"summary": summary, **tables}

    def backtest(self, period: str = "test", matcher: str = "pipeline") -> dict:
        """Rejeu puis évaluation (étape 6)."""
        self.replay(period, matcher)
        return self.evaluate(period, matcher)

    def report(self, matcher: str = "pipeline", period: str = "test") -> dict:
        """Relit une évaluation déjà calculée (sans recalcul)."""
        rep = self.reports_dir
        name = f"evaluation_{matcher}_{period}"
        payload = json.loads((rep / f"{name}.json").read_text(encoding="utf-8"))
        tables = {p.stem.removeprefix(f"{name}_"): pd.read_csv(p) for p in rep.glob(f"{name}_*.csv")}
        return {"summary": payload["summary"], **tables}


def run_task(task: str, kwargs: dict) -> None:
    """Exécute une méthode de `Project` dans un processus séparé (utilisé par l'interface)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    project_args = {k: kwargs.pop(k) for k in ("dataset", "settings_path", "schema_path", "rules_path") if k in kwargs}
    project = Project(**project_args, log=lambda m: print(m, flush=True))
    result = getattr(project, task)(**kwargs)
    summary = result.get("summary") if isinstance(result, dict) else None
    if summary is not None:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
