"""Interface de pilotage — version en un fichier.

    streamlit run reconciliation_ui.py

Exige reconciliation.py dans le même dossier. Écoute locale,
télémétrie désactivée par .streamlit/config.toml (créé par reconciliation.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import altair as alt
import json
import os
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import subprocess
import sys
import types
import typing
import yaml
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import Any

from reconciliation import (
    DEFAULT_RULES_PATH,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_SETTINGS_PATH,
    FIELD_NOTES,
    GLOBAL_KEYS,
    OUTSIDE,
    PERIODS,
    PURGE,
    REPO_ROOT,
    RulesConfig,
    Settings,
    SplitError,
    SplitSettings,
    SyntheticConfig,
    TABLES,
    TABLE_NOTES,
    VALUE_MAP_FIELDS,
    _check_config,
    assign_period,
    compute_split,
    label_numbers,
    load_rules,
    load_schema,
    load_settings,
    normalize_text,
    reference_keys,
    resolve_path,
    save_rules,
    save_schema,
    save_settings,
    scalar,
)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# Éléments communs
# ════════════════════════════════════════════════════════════════════════════════════════════════════

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
        save_settings(updated, settings_path())
        st.toast(f"Paramètres « {section} » enregistrés dans {settings_path().name}", icon=":material/check:")
    if right.button("Valeurs par défaut", key=f"{key}.reset", icon=":material/restart_alt:"):
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
    code_line = "import json, sys; from reconciliation import run_task; run_task(sys.argv[1], json.loads(sys.argv[2]))"
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


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# Pages
# ════════════════════════════════════════════════════════════════════════════════════════════════════


def page_home() -> None:
    st.title("Rapprochement automatique paiements / factures")
    st.caption("POC — gain de taux d'automatisation à précision fixée (cible ≥ 99,5 %). "
               "Chaque étape ne traite que ce que la précédente n'a pas résolu.")

    meta = load_meta()
    cols = st.columns(3)
    cols[0].metric("Jeu de données", dataset())
    cols[1].metric("Événements du journal", f"{meta['journal_events']:,}".replace(",", " ") if meta else "—")
    cols[2].metric("Normalisation", f"v{meta['normalization_version']}" if meta else "—")

    st.subheader("Étapes")
    for step in STEPS.values():
        label, color, icon = STATUS_BADGE[step.status]
        with st.container(border=True):
            left, right = st.columns([5, 1], vertical_alignment="center")
            left.markdown(f"**{step.title}**  \n{step.summary}")
            right.badge(label, icon=icon, color=color)


def page_config() -> None:
    step_header("config")
    NONE_LABEL = "— non mappé —"


    @st.cache_data(show_spinner=False)
    def _header(path: str, opts_json: str, mtime: float) -> list[str]:
        if path.lower().endswith(".parquet"):
            return pq.read_schema(path).names
        return pd.read_csv(path, nrows=0, **json.loads(opts_json)).columns.tolist()


    def _read(path: Path, opts: dict, **kwargs) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path, columns=kwargs.get("usecols"))
            return df.head(kwargs["nrows"]) if "nrows" in kwargs else df
        return pd.read_csv(path, dtype=str, **opts, **kwargs)


    tab_sources, tab_general = st.tabs(["Sources — schema.yaml", "Paramètres généraux"])

    # --- Sources ------------------------------------------------------------------------------
    with tab_sources:
        st.caption(f"Fichier `{schema_path()}`. Associe chaque champ du modèle canonique à une colonne réelle. "
                   "Rien n'est deviné : chaque colonne se choisit dans l'en-tête du fichier source.")
        saved = load_schema(schema_path())
        cfg: dict = {}

        st.markdown("**Conventions des sources**")
        c = st.columns(3)
        cfg["amount_unit"] = c[0].selectbox("Unité des montants", [None, "cents", "units"],
                                            index=[None, "cents", "units"].index(saved["amount_unit"]),
                                            format_func=lambda v: "— à renseigner —" if v is None else v,
                                            help=GLOBAL_KEYS["amount_unit"], key="schema.amount_unit")
        cfg["decimal_separator"] = c[1].selectbox("Séparateur décimal", [None, ".", ","],
                                                  index=[None, ".", ","].index(saved["decimal_separator"]),
                                                  format_func=lambda v: "— à renseigner —" if v is None else f"« {v} »",
                                                  help=GLOBAL_KEYS["decimal_separator"], key="schema.decimal_separator")
        for col, key, label in [(c[2], "source_timezone", "Fuseau des horodatages"),
                                (c[0], "date_format", "Format des dates"),
                                (c[1], "timestamp_format", "Format des horodatages"),
                                (c[2], "base_dir", "Répertoire des sources")]:
            cfg[key] = col.text_input(label, saved[key] or "", help=GLOBAL_KEYS[key], key=f"schema.{key}") or None
        base = resolve_path(cfg["base_dir"] or ".")

        st.markdown("**Tables**")
        cfg["tables"] = {}
        for name, table in TABLES.items():
            tcfg = saved["tables"][name]
            missing = [f.name for f in table.fields if f.required and not tcfg["columns"].get(f.name)]
            configured = bool(tcfg["source"])
            if configured and not missing:
                icon = ":material/check_circle:"
            elif not configured and not table.required:
                icon = ":material/radio_button_unchecked:"
            else:
                icon = ":material/error:"
            mapped = sum(1 for v in tcfg["columns"].values() if v)
            kind = "requise" if table.required else "optionnelle"
            with st.expander(f"{name} — table {kind} · {mapped}/{len(table.fields)} champs mappés", icon=icon):
                if name in TABLE_NOTES:
                    st.caption(TABLE_NOTES[name])
                k = f"schema.{name}"
                left, right = st.columns([2, 1])
                source = left.text_input("Fichier source (csv ou parquet, relatif au répertoire des sources)",
                                         tcfg["source"] or "", key=f"{k}.source") or None
                opts_text = right.text_input("Options de lecture (YAML)", scalar(tcfg["read_options"]),
                                             help="Passées à pandas.read_csv, ex. {sep: ';', encoding: latin-1}",
                                             key=f"{k}.read_options")
                try:
                    opts = yaml.safe_load(opts_text) or {}
                    assert isinstance(opts, dict)
                except Exception:
                    st.error("Options de lecture : YAML invalide (attendu un dictionnaire).")
                    opts = tcfg["read_options"]

                header: list[str] | None = None
                path = base / source if source else None
                if path is not None:
                    if path.exists():
                        try:
                            header = _header(str(path), json.dumps(opts, sort_keys=True), path.stat().st_mtime)
                            st.caption(f"{len(header)} colonnes lues dans l'en-tête de `{path.name}`.")
                        except Exception as exc:  # fichier illisible avec ces options
                            st.error(f"Lecture de l'en-tête impossible : {exc}")
                    else:
                        st.warning(f"Fichier introuvable : `{path}`")

                columns = {}
                grid = st.columns(3)
                for i, f in enumerate(table.fields):
                    current = tcfg["columns"].get(f.name)
                    note = FIELD_NOTES.get((name, f.name))
                    label = f"{f.name}{' *' if f.required else ''}"
                    help_ = f"Type : {f.type.value}. {'Requis.' if f.required else 'Optionnel.'} {note or ''}"
                    cell = grid[i % 3]
                    if header is not None:
                        options = [None, *header] + ([current] if current and current not in header else [])
                        columns[f.name] = cell.selectbox(
                            label, options, index=options.index(current) if current in options else 0,
                            format_func=lambda v, h=header: NONE_LABEL if v is None
                            else (v if v in h else f"{v} (absente du fichier)"),
                            help=help_, key=f"{k}.col.{f.name}")
                    else:
                        columns[f.name] = cell.text_input(label, current or "", help=help_,
                                                          key=f"{k}.col.{f.name}") or None
                entry = {"source": source, "read_options": opts, "columns": columns}
                for opt in ("date_format", "timestamp_format"):
                    if tcfg.get(opt):
                        entry[opt] = tcfg[opt]

                if name in VALUE_MAP_FIELDS:
                    st.markdown("**Traduction des valeurs** — valeurs réelles séparées par des virgules")
                    entry["value_maps"] = {}
                    for fld in VALUE_MAP_FIELDS[name]:
                        current_map = tcfg.get("value_maps", {}).get(fld, {}) or {}
                        mcols = st.columns(3)
                        mapping = {}
                        for j, canonical_value in enumerate(("FULL", "PARTIAL")):
                            real = current_map.get(canonical_value, [])
                            real = real if isinstance(real, list) else [real]
                            text = mcols[j].text_input(f"{fld} = {canonical_value}", ", ".join(map(str, real)),
                                                       key=f"{k}.vm.{fld}.{canonical_value}")
                            values = [v.strip() for v in text.split(",") if v.strip()]
                            if values:
                                mapping[canonical_value] = values
                        entry["value_maps"][fld] = mapping
                        real_col = columns.get(fld)
                        if path is not None and path.exists() and real_col and \
                                mcols[2].button("Valeurs distinctes", key=f"{k}.vm.{fld}.distinct"):
                            counts = _read(path, opts, usecols=[real_col])[real_col].value_counts(dropna=False)
                            st.dataframe(counts.head(30).rename("lignes"), width="content")

                if path is not None and path.exists() and st.toggle("Aperçu (10 lignes)", key=f"{k}.preview"):
                    st.dataframe(_read(path, opts, nrows=10), hide_index=True)
                cfg["tables"][name] = entry

        problems = _check_config(cfg, base)
        if problems:
            with st.expander(f"{len(problems)} point(s) à compléter avant de pouvoir charger les données réelles",
                             icon=":material/warning:", expanded=False):
                st.markdown("\n".join(f"- {p}" for p in problems))
        else:
            st.success("Configuration complète : le chargement des données réelles peut être lancé.",
                       icon=":material/check_circle:")

        if st.button("Enregistrer schema.yaml", type="primary", icon=":material/save:", key="schema.save"):
            save_schema(cfg, schema_path())
            st.toast("schema.yaml enregistré", icon=":material/check:")
            st.rerun()

    # --- Paramètres généraux -----------------------------------------------------------------------
    with tab_general:
        settings = current_settings()
        st.subheader("Chemins")
        paths = render_model(settings.paths, "paths")
        save_section(settings, "paths", paths, "paths")
        st.divider()
        st.subheader("Chargement")
        load = render_model(settings.load, "load")
        save_section(settings, "load", load, "load")


def page_load() -> None:
    step_header("load")
    settings = current_settings()
    ds = dataset()

    # --- Lancement -----------------------------------------------------------------------------
    st.subheader("Lancer le chargement")
    kwargs = {}
    if ds == SYNTHETIC:
        default = SyntheticConfig()
        c = st.columns([2, 1, 1])
        n_payments = c[0].number_input("Paiements visés", min_value=1_000, max_value=5_000_000,
                                       value=default.n_payments, step=100_000, key="load.n_payments",
                                       help="Volume réel : environ 2 millions de paiements sur un an.")
        seed = c[1].number_input("Seed", min_value=0, value=default.seed, step=1, key="load.seed")
        regenerate = c[2].checkbox("Régénérer", key="load.regenerate",
                                   help="Sinon, le jeu déjà généré pour ce volume et cette seed est réutilisé.")
        variant_dir = resolve_path(settings.paths.synthetic_dir) / f"{int(n_payments)}_seed{int(seed)}"
        st.caption(f"Jeu : `{variant_dir}` — {'existe déjà' if (variant_dir / 'payment.csv').exists() else 'à générer'}. "
                   "Comptez ≈ 1 min 30 de génération et ≈ 2 min de chargement pour 2 M de paiements.")
        kwargs = {"n_payments": int(n_payments), "seed": int(seed), "regenerate": bool(regenerate)}
    else:
        st.caption(f"Sources décrites par `{schema_path()}` (page Configuration).")

    if st.button("Lancer l'étape 1", type="primary", icon=":material/play_arrow:", key="run_load"):
        code, lines = run_task("load", "Étape 1 — chargement", **kwargs)
        if code != 0:
            st.error("Échec du chargement :\n\n" + "\n".join(lines[-8:]))
        else:
            st.rerun()

    # --- Résultats -------------------------------------------------------------------------------
    meta = load_meta(settings)
    if meta is None:
        st.info(f"Aucun chargement pour le jeu **{ds}**.")
        st.stop()

    st.subheader("Dernier chargement")
    volumes = read_report("load_volumes.csv")
    counts = dict(zip(volumes["table"], volumes["rows"])) if volumes is not None else {}
    fmt = lambda n: f"{int(n):,}".replace(",", " ")  # noqa: E731
    m = st.columns(5)
    m[0].metric("Paiements", fmt(counts.get("payment", 0)))
    m[1].metric("Factures", fmt(counts.get("invoice", 0)))
    m[2].metric("Imputations", fmt(counts.get("imputation", 0)))
    m[3].metric("Événements", fmt(meta["journal_events"]))
    issues = read_report("load_issues.csv")
    n_issues = 0 if issues is None else len(issues[issues["count"] > 0])
    m[4].metric("Anomalies", n_issues)
    st.caption(f"Source `{meta['source']}` · normalisation v{meta['normalization_version']} · "
               f"journal `{meta['journal_sha256'][:16]}…`")

    tabs = st.tabs(["Anomalies", "Champs manquants", "Plages de dates", "Journal", "Temps d'exécution", "Tables"])
    with tabs[0]:
        if issues is None or issues.empty or n_issues == 0:
            st.success("Aucune anomalie détectée.", icon=":material/check_circle:")
        else:
            st.dataframe(issues[issues["count"] > 0], hide_index=True, width="stretch")
    with tabs[1]:
        missing = read_report("load_missing_fields.csv")
        if missing is not None:
            only = st.toggle("Seulement les champs non mappés ou incomplets", value=True, key="load.only_missing")
            view = missing[(~missing["mapped"]) | (missing["null"] > 0)] if only else missing
            st.dataframe(view, hide_index=True, width="stretch",
                         column_config={"null_pct": st.column_config.ProgressColumn(
                             "null %", min_value=0, max_value=100, format="%.1f %%")})
    with tabs[2]:
        dates = read_report("load_date_ranges.csv")
        if dates is not None:
            st.dataframe(dates, hide_index=True, width="stretch")
    with tabs[3]:
        events = read_report("load_events.csv")
        if events is not None:
            st.dataframe(events, hide_index=True, width="stretch")
    with tabs[4]:
        timings = meta.get("timings_s", {})
        st.dataframe(pd.DataFrame({"étape": list(timings), "secondes": list(timings.values())}),
                     hide_index=True, width="content")
    with tabs[5]:
        if volumes is not None:
            st.dataframe(volumes, hide_index=True, width="content")

    # --- Données normalisées ------------------------------------------------------------------------
    st.subheader("Libellés normalisés")
    pay_path = interim_dir(settings) / "payment.parquet"
    if pay_path.exists():
        pf = pq.ParquetFile(pay_path)
        sample = pf.read_row_group(0, columns=["payment_id", "label", "label_norm", "label_numbers"]).to_pandas()
        sample = sample.sample(min(200, len(sample)), random_state=0).sort_values("payment_id")
        sample["label_numbers"] = sample["label_numbers"].map(lambda v: " · ".join(v))
        st.caption(f"Échantillon de {len(sample)} paiements sur {pf.metadata.num_rows:,}.".replace(",", " "))
        st.dataframe(sample, hide_index=True, width="stretch", height=280)

    st.subheader("Testeur de normalisation")
    st.caption("Vérifie si une référence de facture est retrouvée dans un libellé, avec la normalisation courante.")
    c = st.columns(2)
    label = c[0].text_input("Libellé bancaire", "VIR SEPA DUPONT SARL REGLT 12345 FA", key="load.test_label")
    ref = c[1].text_input("Référence de facture", "FA0012345", key="load.test_ref")
    numbers = label_numbers(normalize_text(label).split())
    keys = reference_keys(ref)
    common = sorted(set(numbers) & set(keys))
    c[0].markdown(f"Normalisé : `{normalize_text(label)}`  \nClés : " + " ".join(f"`{k}`" for k in numbers))
    c[1].markdown("Clés : " + " ".join(f"`{k}`" for k in keys))
    if common:
        st.success(f"Référence retrouvée via : {', '.join(common)}", icon=":material/check_circle:")
    else:
        st.warning("Référence non retrouvée dans le libellé.", icon=":material/search_off:")


def page_split() -> None:
    step_header("split")
    settings = current_settings()

    # Palette de référence : 3 premières teintes catégorielles, gris neutres pour purge / hors période.
    DOMAIN = [*PERIODS, PURGE, OUTSIDE]
    COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a8983", "#c9c8c1"]
    LABELS = {"train": "Entraînement", "validation": "Validation", "test": "Test (backtest)",
              PURGE: "Purge", OUTSIDE: "Hors période"}

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Paramètres")
        values = render_model(settings.split, "split")
        save_section(settings, "split", values, "split")

    with right:
        st.subheader("Aperçu sur les données chargées")
        meta = require_loaded_data()
        if meta is None:
            st.stop()
        try:
            cfg = SplitSettings.model_validate(values)
        except ValidationError as exc:
            st.error("\n".join(f"- {m}" for m in validation_errors(exc)))
            st.stop()

        base = interim_dir(settings)
        pay = read_parquet(base / "payment.parquet", ["value_date", "booking_date"])
        imp = read_parquet(base / "imputation.parquet", ["updated_at"])
        # Jour de connaissance du paiement, comme dans le journal.
        pay_day = pay["booking_date"].fillna(pay["value_date"]).dt.normalize()
        daily = pay_day.value_counts().sort_index().rename_axis("jour").reset_index(name="paiements")

        try:
            split = compute_split(cfg, daily["jour"].min().date(), daily["jour"].max().date())
        except SplitError as exc:
            st.error(f"Découpage impossible : {exc}", icon=":material/error:")
            st.stop()
        for w in split.warnings:
            st.warning(w, icon=":material/warning:")

        daily["période"] = assign_period(daily["jour"], split)
        daily["libellé"] = daily["période"].map(LABELS)
        imp_period = assign_period(imp["updated_at"].dt.normalize(), split)
        rows = []
        for p in split.periods:
            rows.append({
                "période": LABELS[p.name], "début": p.start, "fin": p.end, "jours": p.days,
                "paiements": int(daily.loc[daily["période"] == p.name, "paiements"].sum()),
                "imputations": int((imp_period == p.name).sum()),
            })
        table = pd.DataFrame(rows)
        table["part des paiements"] = table["paiements"] / len(pay_day)
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "début": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "fin": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "paiements": st.column_config.NumberColumn(format="localized"),
            "imputations": st.column_config.NumberColumn(format="localized"),
            "part des paiements": st.column_config.NumberColumn(format="percent"),
        })
        excluded = int(daily.loc[daily["période"].isin([PURGE, OUTSIDE]), "paiements"].sum())
        st.caption(f"{excluded:,} paiements tombent en purge ou hors période.".replace(",", " "))

        chart = alt.Chart(daily).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
            x=alt.X("jour:T", title=None, axis=alt.Axis(format="%m/%Y", labelAngle=0)),
            y=alt.Y("paiements:Q", title="Paiements par jour"),
            color=alt.Color("libellé:N", title=None,
                            scale=alt.Scale(domain=[LABELS[d] for d in DOMAIN], range=COLORS),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("jour:T", title="Jour", format="%d/%m/%Y"),
                     alt.Tooltip("libellé:N", title="Période"),
                     alt.Tooltip("paiements:Q", title="Paiements", format=",")],
        ).properties(height=320)
        st.altair_chart(chart, width="stretch")
        st.caption("L'aperçu suit les valeurs saisies, même non enregistrées. Les paiements sont datés par "
                   "leur date de connaissance (booking_date si disponible, sinon value_date).")

    # --- Rejeu quotidien ------------------------------------------------------------------------------
    st.divider()
    st.subheader("Rejeu quotidien")
    st.caption("Chaque jour D, l'état est figé à la veille (événements < D) et le lot contient les paiements du jour "
               "plus le reliquat non résolu, dans la limite de la rétention. Un paiement sort du reliquat quand une "
               "imputation réelle est prononcée, quand le rapprocheur l'auto-valide, ou à expiration.")

    period = st.segmented_control("Période", options=[*PERIODS, "all"], default="test", key="replay.period",
                                  format_func=lambda p: {"all": "Tout l'historique", **LABELS}[p]) or "test"
    if st.button("Rejouer avec le rapprocheur vide", icon=":material/replay:", key="replay.run"):
        code, lines = run_task("replay", f"Rejeu — {period}", period=period, matcher="null")
        if code != 0:
            st.error("\n".join(lines[-5:]))

    summary_path = reports_dir(settings) / f"replay_null_{period}.json"
    daily_path = reports_dir(settings) / f"replay_null_{period}_daily.csv"
    if not summary_path.exists():
        st.info("Aucun rejeu pour cette période.")
        st.stop()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["journal_sha256"] != meta["journal_sha256"]:
        st.warning("Ce rejeu a été calculé sur un autre journal : le relancer.", icon=":material/warning:")
    fmt = lambda n: f"{int(n):,}".replace(",", "\u202f")  # noqa: E731
    m = st.columns(5)
    m[0].metric("Jours", summary["days"])
    m[1].metric("Paiements arrivés", fmt(summary["new_payments"]))
    m[2].metric("Lignes de lot", fmt(summary["batch_rows"]))
    m[3].metric("Sortis par expiration", fmt(summary["expired"]))
    m[4].metric("Durée du rejeu", f"{summary['timings_s']['rejeu']} s")
    timings = " · ".join(f"{k} {v} s" for k, v in summary["timings_s"].items())
    st.caption(f"Du {summary['start']} au {summary['end']} · rapprocheur « {summary['matcher']} » · {timings}")

    daily = pd.read_csv(daily_path, parse_dates=["day"])
    long = daily.melt(id_vars="day", value_vars=["new", "carried"], var_name="origine", value_name="paiements")
    long["origine"] = long["origine"].map({"new": "Arrivés le jour même", "carried": "Reliquat"})
    bars = alt.Chart(long).mark_bar().encode(
        x=alt.X("day:T", title=None, axis=alt.Axis(format="%d/%m", labelAngle=0)),
        y=alt.Y("paiements:Q", title="Paiements dans le lot", stack=True),
        color=alt.Color("origine:N", title=None, scale=alt.Scale(domain=["Arrivés le jour même", "Reliquat"],
                                                                   range=COLORS[:2]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("day:T", title="Jour", format="%d/%m/%Y"), alt.Tooltip("origine:N", title="Origine"),
                 alt.Tooltip("paiements:Q", title="Paiements", format=",")],
    ).properties(height=280)
    st.altair_chart(bars, width="stretch")
    with st.expander("Statistiques quotidiennes", icon=":material/table:"):
        st.dataframe(daily, hide_index=True, width="stretch")


def page_allocation() -> None:
    step_header("allocation")
    settings = current_settings()
    alloc = settings.allocation
    PERIOD_LABELS = {"train": "Entraînement", "validation": "Validation", "test": "Test (backtest)",
                     "all": "Tout l'historique"}

    # --- Paramètres -------------------------------------------------------------------------------------
    with st.expander("Signaux et paramètres", icon=":material/tune:", expanded=False):
        st.caption("Du plus fort au plus faible. Un débiteur manqué ici est perdu pour la suite : "
                   "désactiver un signal se mesure sur le rappel d'allocation.")
        signals = {}
        titles = {"client_file": "1 · Client file", "reference": "2 · Référence", "iban": "3 · IBAN", "name": "4 · Nom",
                  "amount": "5 · Montant"}
        cols = st.columns(len(titles))
        for col, (name, title) in zip(cols, titles.items()):
            with col.container(border=True):
                st.markdown(f"**{title}**")
                signals[name] = render_model(getattr(alloc.signals, name), f"allocation.signals.{name}")
        values = {"signals": signals, **render_model(alloc, "allocation", skip=("signals",))}
        if not any(s["enabled"] for s in signals.values()):
            st.error("Aucun signal actif : aucun paiement ne pourra être alloué.", icon=":material/error:")
        save_section(settings, "allocation", values, "allocation")
        st.caption("Le signal montant est une extension du brief (clé « montant exact » de la spec) : il départage "
                   "les paiements sans référence ni IBAN exploitable.")

    # --- Mesure ---------------------------------------------------------------------------------------------
    st.subheader("Mesure du rappel")
    c = st.columns([3, 1], vertical_alignment="bottom")
    period = c[0].segmented_control("Période", options=[*PERIODS, "all"], default="validation",
                                    key="allocation.period", format_func=PERIOD_LABELS.get) or "validation"
    if c[1].button("Mesurer l'allocation", type="primary", icon=":material/play_arrow:", key="allocation.run"):
        code, lines = run_task("measure_allocation", f"Allocation — {PERIOD_LABELS[period]}", period=period)
        if code != 0:
            st.error("\n".join(lines[-5:]))

    out = reports_dir(settings)
    tag = f"allocation_{period}"
    if not (out / f"{tag}.json").exists():
        st.info("Aucune mesure pour cette période. Comptez ≈ 4 min pour deux mois à 2 M de paiements.")
        st.stop()
    payload = json.loads((out / f"{tag}.json").read_text(encoding="utf-8"))
    s, ctx = payload["summary"], payload["context"]
    meta = load_meta(settings)
    if meta and ctx["journal_sha256"] != meta["journal_sha256"]:
        st.warning("Mesure calculée sur un autre journal : la relancer.", icon=":material/warning:")
    if ctx["settings"] != alloc.model_dump():
        st.warning("Les paramètres ont changé depuis cette mesure : la relancer pour les évaluer.",
                   icon=":material/sync_problem:")

    pct = lambda v: "—" if v is None else f"{v:.2%}"  # noqa: E731
    recall = s["rappel_premier_passage"]
    if s["cible_atteinte"]:
        st.success(f"Rappel {pct(recall)} ≥ cible {pct(s['rappel_cible'])} : l'étape 4 peut s'appuyer sur l'allocation.",
                   icon=":material/check_circle:")
    else:
        st.error(f"Rappel {pct(recall)} < cible {pct(s['rappel_cible'])} : ne pas avancer à l'étape 4 (brief §9).",
                 icon=":material/block:")
    m = st.columns(5)
    m[0].metric("Rappel (1er passage)", pct(recall), help="Le vrai débiteur figure dans la liste le jour d'arrivée.")
    m[1].metric("Vrai débiteur en tête", pct(s["top1"]))
    m[2].metric("Allocation ferme", pct(s["taux_ferme"]))
    m[3].metric("Précision ferme", pct(s["précision_ferme"]))
    m[4].metric("Sans candidat", pct(s["sans_candidat"]), help="Partent en file analyste.")
    st.caption(f"{PERIOD_LABELS[period]} du {ctx['start']} au {ctx['end']} · "
               f"{s['avec_imputation_réelle']:,} paiements avec imputation réelle · ".replace(",", " ")
               + f"{s['candidats_moyens']:.1f} candidats en moyenne · client file rattaché : {pct(s['avec_client_file'])} · "
               f"rappel au dernier passage {pct(s['rappel_dernier_passage'])} · "
               f"index {ctx['timings_s'].get('construction des index', '—')} s, rejeu {ctx['timings_s'].get('rejeu', '—')} s")

    percent = st.column_config.NumberColumn(format="percent")
    cfg = {k: percent for k in ("rappel", "rappel_dernier_passage", "top1", "taux_ferme", "précision_ferme", "part")}
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Par route IBAN**")
        st.dataframe(pd.read_csv(out / f"{tag}_by_route.csv"), hide_index=True, width="stretch", column_config=cfg)
        st.markdown("**Par statut d'allocation**")
        st.dataframe(pd.read_csv(out / f"{tag}_by_status.csv"), hide_index=True, width="stretch", column_config=cfg)
    with right:
        st.markdown("**Signaux ayant trouvé le vrai débiteur**")
        st.dataframe(pd.read_csv(out / f"{tag}_found_by.csv"), hide_index=True, width="stretch", height=360,
                     column_config=cfg)
    with st.expander("Paiements manqués (échantillon)", icon=":material/search_off:"):
        st.dataframe(pd.read_csv(out / f"{tag}_misses.csv"), hide_index=True, width="stretch")


def page_rules() -> None:
    step_header("rules")
    saved = load_rules(rules_path())

    c = st.columns(3)
    c[0].metric("Version du jeu de règles", saved.version)
    c[1].metric("Règles actives", f"{len(saved.active())} / {len(saved.rules)}")
    min_precision = c[2].number_input("Précision individuelle minimale", value=float(saved.min_precision),
                                      step=0.001, format="%.3f", key="rules.min_precision",
                                      help="Une règle mesurée sous ce seuil est désactivée.")
    st.caption(f"Fichier `{rules_path()}`. Règles appliquées par priorité croissante ; validation seulement si la "
               "solution est unique, sinon le paiement passe à l'étape 5.")

    rules = []
    for rule in sorted(saved.rules, key=lambda r: r.priority):
        k = f"rules.{rule.id}"
        with st.container(border=True):
            head = st.columns([3, 1, 1], vertical_alignment="center")
            enabled = head[0].toggle(f"**{rule.name}** · `{rule.id}`", value=rule.enabled, key=f"{k}.enabled")
            priority = head[1].number_input("Priorité", min_value=1, value=rule.priority, step=1, key=f"{k}.priority")
            head[2].badge("active" if enabled else "désactivée", color="green" if enabled else "gray")
            st.caption(rule.description)
            params = {}
            if rule.params:
                pcols = st.columns(max(len(rule.params), 2))
                for col, (name, value) in zip(pcols, rule.params.items()):
                    if isinstance(value, int):
                        params[name] = int(col.number_input(name, value=value, step=1, key=f"{k}.{name}",
                                                            disabled=not enabled))
                    else:
                        params[name] = float(col.number_input(name, value=float(value), step=0.001, format="%.4f",
                                                              key=f"{k}.{name}", disabled=not enabled))
            rules.append({**rule.model_dump(), "enabled": enabled, "priority": int(priority), "params": params})

    candidate = {"version": saved.version, "min_precision": min_precision, "rules": rules}


    def _comparable(cfg: dict) -> dict:
        return {**cfg, "rules": sorted(cfg["rules"], key=lambda r: r["id"])}


    changed = _comparable(candidate) != _comparable(saved.model_dump())

    if st.button("Enregistrer les règles", type="primary", icon=":material/save:", key="rules.save", disabled=not changed):
        try:
            # Toute modification crée une nouvelle version, tracée dans les décisions.
            updated = RulesConfig.model_validate({**candidate, "version": saved.version + 1})
        except ValidationError as exc:
            st.error("Règles invalides :\n\n" + "\n".join(f"- {m}" for m in validation_errors(exc)))
        else:
            save_rules(updated, rules_path())
            st.toast(f"Règles enregistrées — version {updated.version}", icon=":material/check:")
            st.rerun()
    if changed:
        st.caption(f"Modifications non enregistrées : l'enregistrement créera la version {saved.version + 1}.")

    # --- Mesure ---------------------------------------------------------------------------------------------
    st.subheader("Mesure de la baseline")
    PERIOD_LABELS = {"train": "Entraînement", "validation": "Validation", "test": "Test (backtest)",
                     "all": "Tout l'historique"}
    c = st.columns([3, 1], vertical_alignment="bottom")
    period = c[0].segmented_control("Période", options=[*PERIODS, "all"], default="validation", key="rules.period",
                                    format_func=PERIOD_LABELS.get) or "validation"
    if c[1].button("Rejouer et évaluer", type="primary", icon=":material/play_arrow:", key="rules.run"):
        code, lines = run_task("backtest", f"Règles — {PERIOD_LABELS[period]}", period=period, matcher="rules")
        if code != 0:
            st.error("\n".join(lines[-5:]))

    out = reports_dir(current_settings())
    name = f"evaluation_rules_{period}"
    if not (out / f"{name}.json").exists():
        st.info("Aucune mesure pour cette période. Comptez ≈ 5 min pour deux mois à 2 M de paiements.")
        st.stop()
    payload = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
    summary = payload["summary"]
    pct = lambda v: "—" if v is None else f"{v:.2%}"  # noqa: E731
    m = st.columns(4)
    m[0].metric("Taux d'automatisation", pct(summary["taux_automatisation"]))
    m[1].metric("Précision", pct(summary["précision"]))
    m[2].metric(f"Automatisation à précision ≥ {summary['précision_cible']:.1%}",
                pct(summary["taux_automatisation_à_précision_cible"]))
    current = current_settings().evaluation.current_automation_rate
    m[3].metric("Taux actuel (référence)", pct(current),
                delta=None if current is None else f"{100 * (summary['taux_automatisation'] - current):+.1f} pts")

    cascade = pd.read_csv(out / f"{name}_by_rule.csv").rename(columns={
        "rule": "rule_id", "auto": "décidés (cascade)", "précision": "précision (cascade)",
        "taux_automatisation": "couverture (cascade)"})[["rule_id", "décidés (cascade)", "précision (cascade)",
                                                         "couverture (cascade)"]]
    alone_path = out / f"{name}_by_rule_alone.csv"
    table = pd.read_csv(alone_path).rename(columns={"précision": "précision (seule)", "couverture": "couverture (seule)",
                                                    "propositions": "propositions (seule)"})     if alone_path.exists() else pd.DataFrame({"rule_id": [r.id for r in saved.rules]})
    table = table.merge(cascade, on="rule_id", how="left")
    below = table[table["précision (seule)"].notna() & (table["précision (seule)"] < saved.min_precision)]
    percent = st.column_config.NumberColumn(format="percent")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        c: percent for c in ("précision (seule)", "couverture (seule)", "précision (cascade)", "couverture (cascade)")})
    st.caption("« Seule » : chaque règle évaluée indépendamment sur tous les paiements (R4, coûteuse, sur le résiduel "
               "des règles précédentes). « Cascade » : ce que chaque règle décide effectivement, après arbitrage.")
    if len(below):
        st.warning(f"{len(below)} règle(s) sous le seuil de précision {saved.min_precision:.1%} : "
                   + ", ".join(below["rule_id"]), icon=":material/warning:")
        if st.button("Désactiver ces règles", icon=":material/block:", key="rules.disable_below"):
            updated = saved.model_copy(deep=True)
            for r in updated.rules:
                if r.id in set(below["rule_id"]):
                    r.enabled = False
            updated.version += 1
            save_rules(updated, rules_path())
            st.rerun()
    else:
        st.success(f"Toutes les règles mesurées tiennent la précision minimale {saved.min_precision:.1%}.",
                   icon=":material/check_circle:")


def page_ml() -> None:
    step_header("ml")
    settings = current_settings()
    ml = settings.reconcile_ml
    st.caption("Ne traite que le résiduel de l'étape 4. Chaque brique est mesurée et retirée si elle n'apporte rien.")

    values: dict = {}
    tabs = st.tabs(["Candidats", "Features", "Scoring", "Ensembles", "Décision", "LLM sur libellés"])
    with tabs[0]:
        st.caption("Clés de génération des candidats (union). Filtres durs toujours actifs : devise, facture ouverte, "
                   "agreement actif.")
        values["candidates"] = render_model(ml.candidates, "reconcile_ml.candidates")
    with tabs[1]:
        st.caption("Familles de features du modèle de paires (LightGBM binaire).")
        families = {}
        cols = st.columns(2)
        for i, name in enumerate(type(ml.features).model_fields):
            with cols[i % 2]:
                families[name] = field_widget(ml.features, name, "reconcile_ml.features")
        values["features"] = families
    with tabs[2]:
        values.update(render_model(ml, "reconcile_ml", skip=("candidates", "features", "sets", "decision", "llm_labels")))
    with tabs[3]:
        values["sets"] = render_model(ml.sets, "reconcile_ml.sets")
    with tabs[4]:
        st.caption(f"Auto-validation si p ≥ τ_high et marge au second ≥ δ. τ_high est calibré pour une précision de "
                   f"{settings.evaluation.target_precision:.1%} sur la validation (page Évaluation).")
        values["decision"] = render_model(ml.decision, "reconcile_ml.decision")
    with tabs[5]:
        values["llm_labels"] = render_model(ml.llm_labels, "reconcile_ml.llm_labels")

    save_section(settings, "reconcile_ml", values, "reconcile_ml")

    # --- Entraînement et backtest -------------------------------------------------------------------------------
    st.subheader("Modèle")
    model_dir = resolve_path(settings.paths.models_dir) / ("synthetic" if dataset() == SYNTHETIC else "real") / "pair_model"
    c = st.columns(3)
    if c[0].button("Entraîner (train + validation)", type="primary", icon=":material/model_training:", key="ml.fit"):
        code, lines = run_task("train", "Entraînement")
        if code != 0:
            st.error("\n".join(lines[-5:]))
    if c[1].button("Recalibrer les seuils", icon=":material/tune:", key="ml.calibrate",
                   disabled=not (model_dir / "model.json").exists(),
                   help="Après un changement de la marge δ ou de la précision cible : seuils recalculés sans réentraîner."):
        code, lines = run_task("calibrate", "Recalibration des seuils")
        if code != 0:
            st.error("\n".join(lines[-5:]))
    if c[2].button("Backtest (période de test)", icon=":material/play_arrow:", key="ml.backtest",
                   disabled=not (model_dir / "model.json").exists()):
        code, lines = run_task("backtest", "Backtest règles + ML", period="test", matcher="pipeline")
        if code != 0:
            st.error("\n".join(lines[-5:]))
        else:
            st.success("Backtest terminé : résultats dans la page Évaluation (rapprocheur « Règles + ML »).",
                       icon=":material/check_circle:")

    if not (model_dir / "model.json").exists():
        st.info("Aucun modèle entraîné pour ce jeu de données. Comptez ≈ 30 min à 2 M de paiements.")
        st.stop()
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    current = load_meta(settings)
    if current and meta.get("journal_sha256") != current["journal_sha256"]:
        st.warning("Modèle entraîné sur un autre journal : le réentraîner.", icon=":material/warning:")
    if meta.get("settings") != ml.model_dump():
        st.warning("Les paramètres ML ont changé depuis l'entraînement.", icon=":material/sync_problem:")
    metrics, th = meta.get("metrics", {}), meta.get("thresholds", {})
    val = metrics.get("validation", {})
    pct = lambda v: "—" if v is None else f"{v:.2%}"  # noqa: E731
    m = st.columns(4)
    m[0].metric("AUC (validation)", "—" if val.get("auc") is None else f"{val['auc']:.4f}")
    m[1].metric("Precision@1", pct(val.get("precision_at_1")))
    m[2].metric("MRR", "—" if val.get("mrr") is None else f"{val['mrr']:.3f}")
    m[3].metric("Rappel des candidats", pct(metrics.get("rappel_candidats_validation")))
    st.caption(f"Journal `{meta.get('journal_sha256', '')[:16]}…` · featurisation v{meta.get('featurization_version')} · "
               f"règles v{meta.get('rules_version')} · entraînement {meta.get('periods', {}).get('train')} · "
               f"validation {meta.get('periods', {}).get('validation')}")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Seuils τ_high par type de proposition** (calibrés sur la boucle de validation)")
        kinds = th.get("kinds", {})
        st.dataframe(pd.DataFrame({"type": list(kinds), "τ_high": list(kinds.values())}), hide_index=True,
                     width="stretch")
        st.caption(f"τ_low = {th.get('tau_low')} · marge minimale δ = {th.get('min_margin')} · un seuil > 1 signifie "
                   "« jamais d'auto-validation » (précision cible non atteinte ou volume insuffisant).")
        calib = model_dir / "calibration_validation.csv"
        if calib.exists():
            st.markdown("**Calibration (validation)**")
            st.dataframe(pd.read_csv(calib), hide_index=True, width="stretch")
    with right:
        imp = model_dir / "feature_importance.csv"
        if imp.exists():
            st.markdown("**Features les plus utiles (gain, passe 1)**")
            st.dataframe(pd.read_csv(imp).head(20), hide_index=True, width="stretch", height=560)


def page_evaluation() -> None:
    step_header("evaluation")
    settings = current_settings()
    ev = settings.evaluation
    PERIOD_LABELS = {"train": "Entraînement", "validation": "Validation", "test": "Test (backtest)",
                     "all": "Tout l'historique"}
    MATCHERS = {"pipeline": "Règles + ML (étapes 4 et 5)", "rules": "Règles (étape 4)", "null": "Rapprocheur vide"}

    with st.expander("Cible et chiffres de référence", icon=":material/tune:"):
        values = render_model(ev, "evaluation")
        save_section(settings, "evaluation", values, "evaluation")

    # --- Lancement --------------------------------------------------------------------------------
    c = st.columns([2, 2, 1], vertical_alignment="bottom")
    period = c[0].segmented_control("Période", options=[*PERIODS, "all"], default="test", key="evaluation.period",
                                    format_func=PERIOD_LABELS.get) or "test"
    matcher = c[1].selectbox("Rapprocheur", list(MATCHERS), format_func=MATCHERS.get, key="evaluation.matcher",
                             help="« Règles + ML » exige un modèle entraîné (page 5).")
    if c[2].button("Rejouer et évaluer", type="primary", icon=":material/play_arrow:", key="evaluation.run"):
        code, lines = run_task("backtest", f"Rejeu et évaluation — {PERIOD_LABELS[period]}", period=period,
                               matcher=matcher)
        if code != 0:
            st.error("\n".join(lines[-5:]))

    out = reports_dir(settings)
    name = f"evaluation_{matcher}_{period}"
    if not (out / f"{name}.json").exists():
        st.info("Aucune évaluation pour cette période et ce rapprocheur.")
        st.stop()

    payload = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
    s, ctx = payload["summary"], payload["context"]
    meta = load_meta(settings)
    if meta and ctx["journal_sha256"] != meta["journal_sha256"]:
        st.warning("Évaluation calculée sur un autre journal : la relancer.", icon=":material/warning:")


    def table(kind: str) -> pd.DataFrame:
        return pd.read_csv(out / f"{name}_{kind}.csv")


    pct = lambda v: "—" if v is None else f"{v:.1%}"  # noqa: E731
    num = lambda v: f"{int(v):,}".replace(",", " ")  # noqa: E731

    # --- Indicateurs --------------------------------------------------------------------------------
    st.subheader(f"Taux d'automatisation à précision ≥ {s['précision_cible']:.1%}")
    m = st.columns(4)
    m[0].metric("À précision cible", pct(s["taux_automatisation_à_précision_cible"]),
                help="Part des paiements du périmètre auto-validables en gardant la précision cible, "
                     "en triant les propositions par score.")
    m[1].metric("Taux d'automatisation effectif", pct(s["taux_automatisation"]))
    m[2].metric("Précision des auto-validations", pct(s["précision"]))
    m[3].metric("Paiements du périmètre", num(s["paiements_périmètre"]))
    st.caption(f"{PERIOD_LABELS[period]} du {ctx['start']} au {ctx['end']} · {MATCHERS.get(matcher, matcher)} · "
               f"{num(s['auto'])} auto-validés dont {num(s['auto_corrects'])} corrects · {num(s['revue'])} en revue · "
               f"{num(s['sans_imputation_réelle'])} paiements sans imputation réelle.")
    if s["auto"] and not s["cible_atteinte"]:
        st.error(f"Précision {pct(s['précision'])} sous la cible {pct(s['précision_cible'])}.", icon=":material/error:")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Tableau en cascade**")
        st.dataframe(table("cascade"), hide_index=True, width="stretch", column_config={
            "taux_automatisation": st.column_config.NumberColumn("taux d'automatisation", format="percent"),
            "précision": st.column_config.NumberColumn(format="percent"),
            "gain_points": st.column_config.NumberColumn("gain (points)", format="%.1f"),
        })
        if ev.current_automation_rate is None:
            st.caption("Renseigner le taux actuel (Cible et chiffres de référence) pour mesurer le gain.")
    with right:
        st.markdown("**Par type de groupe**")
        st.dataframe(table("by_group"), hide_index=True, width="stretch", column_config={
            "group_type": "type", "paiements": st.column_config.NumberColumn(format="localized"),
            "précision": st.column_config.NumberColumn(format="percent"),
            "taux_automatisation": st.column_config.NumberColumn("taux d'automatisation", format="percent"),
        })

    curve = table("curve")
    st.markdown("**Courbe automatisation / précision**")
    if curve.empty:
        st.info("Aucune proposition : la courbe apparaîtra avec un rapprocheur qui décide.", icon=":material/info:")
    else:
        line = alt.Chart(curve).mark_line(strokeWidth=2, color="#2a78d6").encode(
            x=alt.X("taux_automatisation:Q", title="Taux d'automatisation", axis=alt.Axis(format="%")),
            y=alt.Y("précision:Q", title="Précision", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("seuil:Q", format=".3f"), alt.Tooltip("taux_automatisation:Q", format=".1%"),
                     alt.Tooltip("précision:Q", format=".2%")])
        target = alt.Chart(pd.DataFrame({"y": [s["précision_cible"]]})).mark_rule(
            strokeDash=[4, 4], color="#8a8983").encode(y="y:Q")
        st.altair_chart((line + target).properties(height=300), width="stretch")

    if s.get("ml_paiements_en_ml") is not None:
        st.markdown("**Scoring ML sur la période** (premier passage de chaque paiement en ML)")
        mm = st.columns(4)
        mm[0].metric("Paiements passés en ML", num(s["ml_paiements_en_ml"]))
        mm[1].metric("Rappel des candidats", pct(s.get("ml_rappel_candidats")))
        mm[2].metric("Precision@1", pct(s.get("ml_precision_at_1")))
        mm[3].metric("MRR", "—" if s.get("ml_mrr") is None else f"{s['ml_mrr']:.3f}")

    kinds = [("by_month", "Par mois"), ("by_step", "Par étape"), ("by_rule", "Par règle"),
             ("by_client_file", "Avec / sans client file"), ("ml_calibration", "Calibration ML")]
    kinds = [(k, label) for k, label in kinds if (out / f"{name}_{k}.csv").exists()]
    tabs = st.tabs([label for _, label in kinds])
    for tab, (kind, _) in zip(tabs, kinds):
        with tab:
            df = table(kind)
            if df.empty:
                st.caption("Aucune auto-validation.")
            else:
                st.dataframe(df, hide_index=True, width="stretch", column_config={
                    "précision": st.column_config.NumberColumn(format="percent"),
                    "taux_automatisation": st.column_config.NumberColumn("taux d'automatisation", format="percent"),
                })

    with st.expander("Rapport markdown", icon=":material/description:"):
        st.markdown((out / f"{name}.md").read_text(encoding="utf-8"))

    st.info("Métriques propres à l'allocation (rappel, précision ferme), aux règles et au ML (rappel des candidats, "
            "precision@1, MRR, calibration) : ajoutées avec chaque étape.", icon=":material/schedule:")


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# Navigation
# ════════════════════════════════════════════════════════════════════════════════════════════════════


def main() -> None:
    st.set_page_config(page_title="Rapprochement automatique", page_icon=":material/join:", layout="wide")

    def page(key: str, fn) -> st.Page:
        step = STEPS[key]
        return st.Page(fn, title=step.title, icon=step.icon, url_path=key)

    nav = st.navigation({
        "": [st.Page(page_home, title="Vue d'ensemble", icon=":material/home:", default=True)],
        "Préparation": [page("config", page_config), page("load", page_load), page("split", page_split)],
        "Rapprochement": [page("allocation", page_allocation), page("rules", page_rules), page("ml", page_ml)],
        "Mesure": [page("evaluation", page_evaluation)],
    })
    with st.sidebar:
        st.radio("Jeu de données", options=[SYNTHETIC, REAL], key="dataset", horizontal=True,
                 help="Synthétique : données générées (seed fixe). Réel : sources de config/schema.yaml.")
        meta = load_meta()
        if meta:
            st.caption(f"Dernier chargement : `{meta['source']}` · {meta['journal_events']:,} événements · "
                       f"normalisation v{meta['normalization_version']}".replace(",", "\u202f"))
        else:
            st.caption("Aucune donnée chargée pour ce jeu.")
    nav.run()


if __name__ == "__main__":
    main()
