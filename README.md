# POC — rapprochement automatique paiements / factures

Moteur de rapprochement pour l'affacturage : chaque paiement reçu est rattaché à un débiteur, puis
aux factures qu'il règle, par des règles déterministes puis un modèle de scoring, avec un objectif
de **taux d'automatisation à précision ≥ 99,5 %**. Tout est rejoué jour par jour, sans fuite
d'information future : un seul code sert au backtest et à la production.

**Pour comprendre le projet et ses subtilités : [`GUIDE.md`](GUIDE.md).** Le brief est dans `CLAUDE.md`, la spécification détaillée dans `spec_rapprochement_automatique.md`.

## Installation

Python 3.11 ou plus.

```bash
pip install -r requirements.txt          # ou : uv sync --extra notebook --extra ui
```

## Utilisation dans un notebook

Ouvrir `notebooks/pipeline.ipynb` dans JupyterLab (`jupyter lab`), puis exécuter les cellules dans
l'ordre. Chaque étape est une méthode de `Project` :

```python
from src.api import Project

project = Project(dataset="synthetic")               # ou "real"
project.load(n_payments=50_000)                      # 1 · chargement, normalisation, journal
project.split()                                      # 2 · périodes train / validation / test
project.measure_allocation("validation")             # 3 · allocation (rappel ≥ 99 %)
project.backtest("test", matcher="rules")            # 4 · baseline par règles
project.train()                                      # 5 · modèle (train) et seuils (validation)
project.calibrate()                                  #     seuils seuls, sans réentraîner
project.backtest("test", matcher="pipeline")         # 6 · backtest complet, cascade
```

Chaque méthode renvoie des tableaux (`pandas.DataFrame`) et écrit ses sorties sur disque :
`data/interim/` (tables normalisées, journal, décisions), `reports/` (CSV, JSON, markdown),
`models/` (modèle et seuils). Si le notebook n'est pas lancé depuis la racine ou `notebooks/`,
ajouter la racine du projet à `sys.path`.

Volumes indicatifs (16 cœurs) : 50 000 paiements ≈ 5 min de bout en bout ; 2 M ≈ 1 h.

## Données réelles

1. Renseigner `config/schema.yaml` : fichier source et colonne réelle de chaque champ du modèle
   (rien n'est deviné), unité des montants, fuseau horaire, valeurs des statuts d'imputation.
2. `Project(dataset="real").load()` — le profil de chargement liste ce qui manque ou pose problème.

## Paramètres

- `config/settings.yaml` : un bloc par étape (découpage, allocation, ML, évaluation), commenté.
- `config/rules.yaml` : les règles de l'étape 4, versionnées (activation, priorité, tolérances).

Modifiables à la main, depuis le notebook (`src.settings.load_settings` / `save_settings`) ou depuis
l'interface.

## Interface (optionnelle)

```bash
streamlit run src/ui/app.py
```

Une page par étape : configuration, chargement, découpage, allocation, règles, ML, évaluation. Les
boutons exécutent les mêmes méthodes de `Project` que le notebook. Écoute locale uniquement,
télémétrie désactivée (`.streamlit/config.toml`).

## Organisation du code

| Dossier | Rôle |
|---|---|
| `src/api.py` | Point d'entrée : `Project`, une méthode par étape |
| `src/load/` | Étape 1 — chargement, normalisation, journal d'événements, contrôles qualité |
| `src/timeline/` | Étape 2 — état du grand livre à date, boucle quotidienne, découpage |
| `src/allocation/` | Étape 3 — allocation des paiements aux débiteurs |
| `src/reconcile_rules/` | Étape 4 — règles déterministes |
| `src/reconcile_ml/` | Étape 5 — candidats, features, modèle, ensembles, décision |
| `src/evaluation/` | Étape 6 — métriques et rapports |
| `src/synthetic/` | Générateur de données synthétiques (seed fixe) |
| `src/ui/` | Interface Streamlit |

## Garanties

- Toute lecture de l'état exige une date `as_of` ; une facture future ne peut pas être citée
  (erreur `LeakError`).
- Montants en entiers de centimes ; aucun appel réseau.
- Le modèle est sauvegardé avec l'empreinte du journal et la version de la featurisation.
