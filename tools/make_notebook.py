"""Génère notebooks/pipeline.ipynb (outil de développement, hors livrable)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    ("md", """# Rapprochement automatique paiements / factures — pipeline

Chaque étape du brief est une méthode de `Project` (`src/api.py`). Les résultats sont écrits sur disque
(`data/interim`, `reports/`, `models/`) et renvoyés sous forme de tableaux.

- **Paramètres** : `config/settings.yaml` (par étape) et `config/rules.yaml` (règles de l'étape 4),
  éditables à la main, depuis l'interface (`streamlit run src/ui/app.py`) ou depuis ce notebook.
- **Données réelles** : renseigner `config/schema.yaml`, puis `Project(dataset="real")`.
- **Données synthétiques** : `Project(dataset="synthetic")` — volume réglable (réel : ~2 M de paiements ;
  commencer par 50 000 pour une exécution de quelques minutes)."""),
    ("code", """import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
from IPython.display import Markdown, display

from src.api import Project

pd.set_option("display.max_columns", 50)
project = Project(dataset="synthetic", log=print)
N_PAYMENTS = 50_000          # 2_000_000 pour le volume réel (≈ 1 h de bout en bout)"""),
    ("md", "## Paramètres\n\nLecture et modification des paramètres depuis le notebook (écrits dans `config/settings.yaml`)."),
    ("code", """from src.settings import save_settings

settings = project.settings
print(settings.split)
print(settings.allocation.signals)
# Exemple : settings.split.purge_days = 7 ; save_settings(settings, project.settings_path)"""),
    ("md", "## Étape 1 — chargement, normalisation, journal"),
    ("code", """step1 = project.load(n_payments=N_PAYMENTS)
display(step1["volumes"])
display(step1["issues"][step1["issues"]["count"] > 0])
display(step1["missing_fields"][~step1["missing_fields"]["mapped"] | (step1["missing_fields"]["null"] > 0)])
step1["meta"]"""),
    ("md", "## Étape 2 — découpage temporel et boucle quotidienne"),
    ("code", """split = project.split()
display(pd.DataFrame([{"période": p.name, "début": p.start, "fin": p.end, "jours": p.days} for p in split.periods]))

replay = project.replay("test", "null")        # rapprocheur vide : taille des lots et reliquat
daily = replay["daily"].melt(id_vars="day", value_vars=["new", "carried"], var_name="origine", value_name="paiements")
alt.Chart(daily).mark_bar().encode(x="day:T", y="paiements:Q", color="origine:N").properties(height=240)"""),
    ("md", "## Étape 3 — allocation (rappel cible ≥ 99 %)"),
    ("code", """alloc = project.measure_allocation("validation")
display(pd.Series(alloc["summary"]))
display(alloc["by_route"])
display(alloc["found_by"].head(10))"""),
    ("md", "## Étape 4 — réconciliation par règles (baseline)"),
    ("code", """rules = project.backtest("test", matcher="rules")
display(pd.Series(rules["summary"]))
display(rules["by_rule_alone"])
display(rules["by_rule"])"""),
    ("md", """## Étape 5 — réconciliation ML (entraînement sur train, seuils sur la validation)

`project.train()` rejoue entraînement + validation, entraîne le modèle et calibre les seuils sur la boucle
réelle de validation. Après un changement des paramètres de décision (marge δ, précision cible),
`project.calibrate()` recalcule seulement les seuils, sans réentraîner."""),
    ("code", """meta = project.train()
display(pd.Series(meta["metrics"]["validation"]))
display(pd.Series(meta["thresholds"]["kinds"], name="τ_high par type"))
pd.read_csv(project.model_dir / "feature_importance.csv").head(15)"""),
    ("md", "## Étape 6 — backtest complet et tableau en cascade"),
    ("code", """ev = project.backtest("test", matcher="pipeline")
display(pd.Series(ev["summary"]))
display(ev["cascade"])
display(ev["by_group"])
display(ev["by_step"])"""),
    ("code", """curve = ev["curve"].replace([float("inf")], None).dropna()
target = ev["summary"]["précision_cible"]
line = alt.Chart(curve).mark_line().encode(x=alt.X("taux_automatisation:Q", axis=alt.Axis(format="%")),
                                           y=alt.Y("précision:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="%")))
rule = alt.Chart(pd.DataFrame({"y": [target]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
(line + rule).properties(height=260, title="Automatisation / précision (règles acquises, seuil ML variable)")"""),
    ("code", """for name in ("by_month", "by_client_file", "ml_calibration"):
    if name in ev:
        display(Markdown(f"**{name}**"))
        display(ev[name])
display(Markdown((project.reports_dir / "evaluation_pipeline_test.md").read_text(encoding="utf-8")))"""),
]


def main() -> None:
    cells = []
    for kind, text in CELLS:
        lines = text.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out = ROOT / "notebooks" / "pipeline.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
