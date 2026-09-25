import json

import altair as alt
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src.settings import SplitSettings
from src.timeline.split import OUTSIDE, PERIODS, PURGE, SplitError, assign_period, compute_split
from src.ui.common import (
    SYNTHETIC, current_settings, dataset, interim_dir, read_parquet, render_model, reports_dir,
    require_loaded_data, run_task, save_section, settings_path, step_header, validation_errors,
)

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
