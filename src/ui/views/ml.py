import json

import pandas as pd
import streamlit as st

from src.config import resolve_path
from src.ui.common import (
    SYNTHETIC, current_settings, dataset, field_widget, load_meta, render_model, rules_path, run_task, save_section,
    settings_path, step_header,
)

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
