"""Éléments partagés par les pages de l'UI.

- Registre des étapes et de leur état d'implémentation.
- Formulaires générés à partir des modèles de `src/settings.py` : l'UI et la
  pipeline lisent les mêmes paramètres, avec la même validation.
- Exécution des étapes en sous-processus (même CLI qu'en ligne de commande),
  journal affiché en direct.

Chemins de config surchargeables par variables d'environnement
(`RECON_SETTINGS`, `RECON_SCHEMA`, `RECON_RULES`) — utilisé par les tests pour ne
jamais toucher la config ni les données du dépôt.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
import typing
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import BaseModel, ValidationError

from src.config import DEFAULT_SCHEMA_PATH, DEFAULT_SETTINGS_PATH, REPO_ROOT, resolve_path
from src.settings import DEFAULT_RULES_PATH, Settings, load_settings

# --- Registre des étapes ------------------------------------------------------------

DONE, PARTIAL, TODO = "done", "partial", "todo"
STATUS_BADGE = {
    DONE: ("Opérationnel", "green", ":material/check_circle:"),
    PARTIAL: ("Partiel", "orange", ":material/timelapse:"),
    TODO: ("À venir", "gray", ":material/schedule:"),
}


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    icon: str
    status: str
    summary: str
    note: str = ""


STEPS: dict[str, Step] = {s.key: s for s in [
    Step("config", "Configuration", ":material/settings:", DONE,
         "Mapping des sources réelles (schema.yaml) et paramètres généraux."),
    Step("load", "1 · Chargement", ":material/database:", DONE,
         "Chargement, normalisation, journal d'événements, contrôles qualité."),
    Step("split", "2 · Découpage temporel", ":material/date_range:", DONE,
         "Périodes train / validation / test, état du grand livre à date et boucle quotidienne sans fuite."),
    Step("allocation", "3 · Allocation", ":material/account_tree:", DONE,
         "Rattachement de chaque paiement à une liste classée de débiteurs candidats, rappel mesuré."),
    Step("rules", "4 · Réconciliation algorithmique", ":material/rule:", DONE,
         "Règles déterministes, validation seulement si la solution est unique : la baseline à battre."),
    Step("ml", "5 · Réconciliation ML", ":material/model_training:", DONE,
         "Candidats, scoring LightGBM en deux passes, ensembles, décision par seuils sur le résiduel de l'étape 4."),
    Step("evaluation", "6 · Évaluation", ":material/monitoring:", DONE,
         "Backtest, taux d'automatisation à précision fixée, cascade, découpages par étape, règle, mois, groupe."),
]}


def step_header(key: str) -> Step:
    step = STEPS[key]
    label, color, icon = STATUS_BADGE[step.status]
    st.title(step.title)
    st.badge(label, icon=icon, color=color)
    st.caption(step.summary)
    if step.note:
        (st.info if step.status == TODO else st.caption)(step.note)
    return step


# --- Chemins -------------------------------------------------------------------------

REAL, SYNTHETIC = "réel", "synthétique"


def settings_path() -> Path:
    return Path(os.environ.get("RECON_SETTINGS", DEFAULT_SETTINGS_PATH))


def schema_path() -> Path:
    return Path(os.environ.get("RECON_SCHEMA", DEFAULT_SCHEMA_PATH))


def rules_path() -> Path:
    return Path(os.environ.get("RECON_RULES", DEFAULT_RULES_PATH))


def current_settings() -> Settings:
    return load_settings(settings_path())


def dataset() -> str:
    return st.session_state.get("dataset", SYNTHETIC)


def interim_dir(settings: Settings | None = None, ds: str | None = None) -> Path:
    base = resolve_path((settings or current_settings()).paths.interim_dir)
    return base / "synthetic" if (ds or dataset()) == SYNTHETIC else base


def reports_dir(settings: Settings | None = None, ds: str | None = None) -> Path:
    base = resolve_path((settings or current_settings()).paths.reports_dir)
    return base / "synthetic" if (ds or dataset()) == SYNTHETIC else base


def load_meta(settings: Settings | None = None, ds: str | None = None) -> dict[str, Any] | None:
    path = interim_dir(settings, ds) / "journal_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def require_loaded_data() -> dict[str, Any] | None:
    meta = load_meta()
    if meta is None:
        st.warning(f"Aucune donnée chargée pour le jeu **{dataset()}** : lancer d'abord l'étape 1 "
                   "(page Chargement).")
    return meta


# --- Formulaires générés depuis les modèles ------------------------------------------


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    args = typing.get_args(annotation)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType) and type(None) in args:
        rest = [a for a in args if a is not type(None)]
        return rest[0], True
    return annotation, False


def render_model(model: BaseModel, key: str, disabled: bool = False, skip: tuple[str, ...] = ()) -> dict[str, Any]:
    """Un widget par champ (label = description, aide = clé YAML). Retourne les valeurs saisies.

    Si le modèle a un champ `enabled` désactivé, ses autres champs sont grisés.
    """
    values: dict[str, Any] = {}
    fields = type(model).model_fields
    if "enabled" in fields and "enabled" not in skip:
        values["enabled"] = field_widget(model, "enabled", key, disabled)
        disabled = disabled or not values["enabled"]
    for name, info in fields.items():
        if name in values or name in skip:
            continue
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            st.markdown(f"**{info.description or name}**")
            with st.container(border=True):
                values[name] = render_model(value, f"{key}.{name}", disabled)
        else:
            values[name] = field_widget(model, name, key, disabled)
    return values


def field_widget(model: BaseModel, name: str, key: str, disabled: bool = False) -> Any:
    info = type(model).model_fields[name]
    value = getattr(model, name)
    label = info.description or name
    wkey = f"{key}.{name}"
    help_ = f"Clé YAML : `{wkey}`"
    annotation, optional = _unwrap_optional(info.annotation)

    if annotation is bool:
        return st.toggle(label, value=value, key=wkey, help=help_, disabled=disabled)
    if optional:
        active = st.checkbox(f"{label}", value=value is not None, key=f"{wkey}.set", help=help_,
                             disabled=disabled)
        if not active:
            return None
        default = value if value is not None else (info.default if info.default is not None else None)
        return _scalar_widget(annotation, "↳ valeur", default, wkey, help_, disabled)
    if typing.get_origin(annotation) is list:
        options = list(dict.fromkeys([*(info.default_factory() if info.default_factory else []), *value]))
        return st.multiselect(label, options=options, default=value, key=wkey, help=help_, disabled=disabled)
    return _scalar_widget(annotation, label, value, wkey, help_, disabled)


def _scalar_widget(annotation: Any, label: str, value: Any, wkey: str, help_: str, disabled: bool) -> Any:
    if annotation is int:
        return int(st.number_input(label, value=int(value if value is not None else 0), step=1,
                                   key=wkey, help=help_, disabled=disabled))
    if annotation is float:
        return float(st.number_input(label, value=float(value if value is not None else 0.0), step=0.001,
                                     format="%.4f", key=wkey, help=help_, disabled=disabled))
    if annotation is date:
        return st.date_input(label, value=value or date.today(), key=wkey, help=help_, disabled=disabled)
    text = st.text_input(label, value="" if value is None else str(value), key=wkey, help=help_,
                         disabled=disabled)
    return text or None


def validation_errors(exc: ValidationError) -> list[str]:
    return [f"`{'.'.join(str(p) for p in e['loc'])}` : {e['msg']}" for e in exc.errors()]


def save_section(settings: Settings, section: str, values: dict[str, Any], key: str) -> None:
    """Boutons Enregistrer / Valeurs par défaut pour une section de `Settings`."""
    left, right = st.columns([1, 1])
    if left.button("Enregistrer", type="primary", key=f"{key}.save", icon=":material/save:"):
        try:
            updated = Settings.model_validate({**settings.model_dump(), section: values})
        except ValidationError as exc:
            st.error("Paramètres invalides :\n\n" + "\n".join(f"- {m}" for m in validation_errors(exc)))
            return
        from src.settings import save_settings
        save_settings(updated, settings_path())
        st.toast(f"Paramètres « {section} » enregistrés dans {settings_path().name}", icon=":material/check:")
    if right.button("Valeurs par défaut", key=f"{key}.reset", icon=":material/restart_alt:"):
        from src.settings import save_settings
        defaults = Settings.model_validate({**settings.model_dump(), section: {}})
        save_settings(defaults, settings_path())
        for k in [k for k in st.session_state if str(k).startswith(f"{key}.")]:
            del st.session_state[k]
        st.rerun()


# --- Exécution en sous-processus -------------------------------------------------------


def run_task(task: str, label: str, **kwargs) -> tuple[int, list[str]]:
    """Exécute `Project.<task>(**kwargs)` (src/api.py) dans un processus séparé, journal en direct.

    Même code que dans le notebook ; le processus séparé garde l'interface réactive et sa mémoire
    indépendante des 2 M de lignes traitées.
    """
    project = {"dataset": "synthetic" if dataset() == SYNTHETIC else "real", "settings_path": str(settings_path()),
               "schema_path": str(schema_path()), "rules_path": str(rules_path())}
    payload = json.dumps({**project, **kwargs})
    code_line = "import json, sys; from src.api import run_task; run_task(sys.argv[1], json.loads(sys.argv[2]))"
    cmd = [sys.executable, "-c", code_line, task, payload]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    lines: list[str] = []
    with st.status(label, expanded=True) as status:
        shown = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        st.code(f"Project(dataset={project['dataset']!r}).{task}({shown})", language="python")
        log = st.empty()
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())
            log.code("\n".join(lines[-30:]), language="text")
        code = proc.wait()
        status.update(label=f"{label} — {'terminé' if code == 0 else f'échec (code {code})'}",
                      state="complete" if code == 0 else "error", expanded=code != 0)
    return code, lines


# --- Lecture des sorties -------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _read_parquet_cached(path: str, columns: tuple[str, ...] | None, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Lecture mise en cache, invalidée quand le fichier change."""
    return _read_parquet_cached(str(path), tuple(columns) if columns else None, path.stat().st_mtime)


def read_report(name: str) -> pd.DataFrame | None:
    path = reports_dir() / name
    return pd.read_csv(path) if path.exists() else None
