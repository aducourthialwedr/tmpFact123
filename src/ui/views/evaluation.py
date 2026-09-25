import json

import altair as alt
import pandas as pd
import streamlit as st

from src.timeline.split import PERIODS
from src.ui.common import (
    SYNTHETIC, current_settings, dataset, load_meta, render_model, reports_dir, rules_path, run_task, save_section,
    settings_path, step_header,
)

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
