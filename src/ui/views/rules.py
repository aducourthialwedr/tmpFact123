import json

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src.settings import RulesConfig, load_rules, save_rules
from src.timeline.split import PERIODS
from src.ui.common import (
    SYNTHETIC, current_settings, dataset, reports_dir, rules_path, run_task, settings_path, step_header,
    validation_errors,
)

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
