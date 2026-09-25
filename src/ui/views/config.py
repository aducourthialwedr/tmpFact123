import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import yaml

from src.config import resolve_path
from src.load.canonical import TABLES
from src.load.loader import _check_config
from src.load.schema_io import FIELD_NOTES, GLOBAL_KEYS, TABLE_NOTES, VALUE_MAP_FIELDS, load_schema, save_schema
from src.ui.common import current_settings, render_model, save_section, schema_path, step_header
from src.yaml_io import scalar

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
