"""Moteur de rapprochement automatique paiements / factures — version en un fichier.

Chaque section reprend un module du dépôt d'origine (bannières « src/... »), ce qui permet de
suivre GUIDE.md.

    from reconciliation import Project
    project = Project(dataset="synthetic")
    project.load(n_payments=50_000)
    ...

Au premier import, les fichiers de configuration manquants sont créés à côté de ce fichier
(config/settings.yaml, config/rules.yaml, config/schema.yaml, config/schema.synthetic.yaml,
.streamlit/config.toml).
"""

from __future__ import annotations

import copy
import csv
import gc
import hashlib
import json
import lightgbm as lgb
import math
import numpy as np
import pandas as pd
import pickle
import pyarrow as pa
import pyarrow.compute as pc
import re
import sys
import textwrap
import threading
import time
import yaml
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from enum import Enum
from itertools import chain
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from types import SimpleNamespace
from typing import Any, Protocol


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/config.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Lecture des fichiers de configuration YAML."""




REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA_PATH = REPO_ROOT / "config" / "schema.yaml"
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    return content or {}


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    """Chemin absolu : relatif à `base` (racine du dépôt par défaut)."""
    p = Path(path)
    return p if p.is_absolute() else base / p


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/yaml_io.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Écriture YAML commentée à partir de modèles pydantic.

Les fichiers de config sont édités à la fois à la main et par l'UI : on les
régénère avec les descriptions des champs en commentaires, pour que l'écriture
par l'UI ne perde pas la documentation.
"""





def scalar(value: Any) -> str:
    """Représentation YAML d'une valeur scalaire, liste ou dict courts (style flow)."""
    if isinstance(value, date):
        return value.isoformat()
    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, sort_keys=False, width=10_000)
    return text.removesuffix("\n...\n").strip()


def _comment(text: str | None, pad: str) -> list[str]:
    if not text:
        return []
    return [f"{pad}# {line}" for line in textwrap.wrap(text, width=96 - len(pad))]


def model_lines(model: BaseModel, indent: int = 0, comments: bool = True) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    for name, info in type(model).model_fields.items():
        value = getattr(model, name)
        if comments:
            lines += _comment(info.description, pad)
        if isinstance(value, BaseModel):
            lines.append(f"{pad}{name}:")
            lines += model_lines(value, indent + 2, comments)
        elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
            lines.append(f"{pad}{name}:")
            for item in value:
                item_lines = model_lines(item, indent + 4, comments=False)
                first = item_lines[0].lstrip()
                lines.append(f"{pad}  - {first}")
                lines += item_lines[1:]
        elif isinstance(value, dict) and value:
            lines.append(f"{pad}{name}:")
            lines += [f"{pad}  {k}: {scalar(v)}" for k, v in value.items()]
        else:
            lines.append(f"{pad}{name}: {scalar(value)}")
        if comments and indent == 0:
            lines.append("")
    return lines


def dump_model(model: BaseModel, header: str = "") -> str:
    head = [f"# {line}" if line else "#" for line in header.strip().splitlines()] if header else []
    body = model_lines(model)
    return "\n".join(head + ([""] if head else []) + body).rstrip() + "\n"


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/settings.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Paramètres de la pipeline, par étape — source unique pour la pipeline et l'UI.

- `config/settings.yaml` : `Settings` (chemins, chargement, découpage,
  allocation, réconciliation ML, évaluation).
- `config/rules.yaml`    : `RulesConfig`, les règles déterministes de l'étape 4
  (déclaratives et versionnées, brief §6).

Les valeurs par défaut marquées « à calibrer » sont des points de départ, pas
des résultats : elles se fixent sur la période de validation.
"""





DEFAULT_RULES_PATH = REPO_ROOT / "config" / "rules.yaml"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --- Chemins et chargement (étape 1) -------------------------------------------


class PathsSettings(_Model):
    interim_dir: str = Field("data/interim", description="Tables normalisées et journal produits par l'étape 1.")
    reports_dir: str = Field("reports", description="Rapports CSV et markdown.")
    synthetic_dir: str = Field("data/synthetic", description="Jeux synthétiques, un sous-dossier par volume et seed.")
    models_dir: str = Field("models", description="Modèles de l'étape 5 (un sous-dossier par jeu de données).")


class LoadSettings(_Model):
    workers: int = Field(12, ge=1, description="Processus pour la normalisation des gros volumes "
                         "(1 = séquentiel). Sans effet sur le résultat.")


# --- Découpage temporel (étape 2) ------------------------------------------------


class SplitSettings(_Model):
    test_months: int = Field(2, ge=1, description="Durée de la période de test (backtest), en mois : "
                             "les derniers mois de l'historique.")
    validation_months: int = Field(2, ge=1, description="Durée de la période de validation, en mois, "
                                   "juste avant le test.")
    train_months: int | None = Field(None, ge=1, description="Durée de la période d'entraînement, en mois. "
                                     "null = tout l'historique disponible avant la validation.")
    purge_days: int = Field(5, ge=0, description="Jours exclus entre deux périodes (purge), pour éviter "
                            "qu'un groupe d'imputations chevauche deux blocs.")
    anchor_end: date | None = Field(None, description="Dernier jour de la période de test. null = dernier "
                                    "jour de paiement présent dans les données.")
    retention_days: int = Field(60, ge=1, description="Durée pendant laquelle un paiement non résolu reste "
                                "dans le reliquat retraité chaque jour.")


# --- Allocation (étape 3) --------------------------------------------------------


class ClientFileSignal(_Model):
    enabled: bool = Field(True, description="Rattacher le paiement au client file correspondant "
                          "(référence de virement, montant total, date, IBAN).")
    date_tolerance_days: int = Field(5, ge=0, description="Écart maximal entre la date du client file et la "
                                     "date de valeur du paiement pour corroborer le rattachement.")


class ReferenceSignal(_Model):
    enabled: bool = Field(True, description="Débiteur de la facture dont la référence figure dans le libellé.")
    min_key_length: int = Field(4, ge=1, description="Longueur minimale d'une clé de référence utilisée "
                                "(les clés courtes sont du bruit : années, numéros de rue).")
    strong_min_key_length: int = Field(6, ge=1, description="Longueur minimale d'une clé désignant un seul "
                                       "débiteur pour constituer un signal fort (allocation ferme).")
    max_debtors_per_key: int = Field(5, ge=1, description="Une clé partagée par plus de débiteurs (factures "
                                     "existantes à D) est ignorée.")


class IbanSignal(_Model):
    enabled: bool = Field(True, description="Routage IBAN (DEBTOR_DIRECT / ASSIGNOR / TECHNICAL_ACCOUNT / "
                          "UNKNOWN) arbitré par bankroll_code.")


class NameSignal(_Model):
    enabled: bool = Field(True, description="Similarité entre le libellé et le nom du débiteur (index inversé).")
    min_similarity: float = Field(0.5, ge=0, le=1, description="Score de nom minimal : part du nom retrouvée "
                                  "dans le libellé (pondérée par la rareté des mots, bigrammes inclus), ou "
                                  "spécificité du meilleur mot (1 / nombre de débiteurs le portant). À calibrer.")
    max_token_share: float = Field(0.01, gt=0, le=1, description="Élagage : un mot présent dans plus que "
                                   "cette part des noms de débiteurs est ignoré. À calibrer.")
    max_debtors_per_term: int = Field(200, ge=1, description="Élagage absolu : un mot partagé par plus de "
                                      "débiteurs est ignoré, quelle que soit leur part.")
    min_token_length: int = Field(3, ge=1, description="Longueur minimale d'un mot indexé.")


class AmountSignal(_Model):
    enabled: bool = Field(True, description="Débiteurs ayant une facture ouverte à D dont le restant dû égale "
                          "exactement le montant du paiement. Extension du brief (clé K3 de la spec).")
    max_debtors_per_amount: int = Field(3, ge=1, description="Un montant partagé par plus de débiteurs "
                                        "(factures ouvertes à D) est ignoré, sauf confirmation par le nom.")
    max_debtors_with_name_hint: int = Field(30, ge=1, description="Jusqu'à ce nombre de débiteurs partageant le "
                                            "montant, ceux dont un mot du nom (même fréquent) figure dans le "
                                            "libellé sont retenus.")


class AllocationSignals(_Model):
    client_file: ClientFileSignal = Field(default_factory=ClientFileSignal, description="Signal 1 (le plus fort).")
    reference: ReferenceSignal = Field(default_factory=ReferenceSignal, description="Signal 2.")
    iban: IbanSignal = Field(default_factory=IbanSignal, description="Signal 3.")
    name: NameSignal = Field(default_factory=NameSignal, description="Signal 4.")
    amount: AmountSignal = Field(default_factory=AmountSignal, description="Signal 5 (le plus faible).")


class AllocationSettings(_Model):
    signals: AllocationSignals = Field(default_factory=AllocationSignals,
                                       description="Signaux d'allocation, du plus fort au plus faible.")
    max_candidates: int = Field(10, ge=1, description="Taille maximale de la liste classée de débiteurs "
                                "candidats transmise aux étapes suivantes.")
    target_recall: float = Field(0.99, gt=0, le=1, description="Rappel d'allocation visé : le vrai débiteur "
                                 "doit figurer dans la liste. Bloquant pour passer à l'étape 4.")


# --- Réconciliation ML (étape 5) ---------------------------------------------------


class CandidateSettings(_Model):
    allocated_debtors: bool = Field(True, description="Factures ouvertes des débiteurs candidats de l'allocation.")
    reference_no_window: bool = Field(True, description="Factures dont la référence figure dans le libellé, "
                                      "sans fenêtre temporelle.")
    amount_exact: bool = Field(True, description="Factures dont le montant ouvert égale le paiement.")
    amount_window_days: int = Field(90, ge=0, description="Fenêtre ± jours de la clé montant.")
    client_file_cited: bool = Field(True, description="Factures citées dans le client file rattaché.")
    debtor_window_before_days: int = Field(180, ge=0, description="Clé débiteur : échéances jusqu'à ce nombre "
                                           "de jours avant la date de valeur.")
    debtor_window_after_days: int = Field(30, ge=0, description="Clé débiteur : échéances jusqu'à ce nombre "
                                          "de jours après la date de valeur.")
    max_debtors: int = Field(3, ge=1, description="Clé débiteur : factures des N premiers débiteurs de "
                             "l'allocation seulement (les gros débiteurs ont des milliers de factures ouvertes).")
    max_per_payment: int = Field(60, ge=1, description="Candidats maximum par paiement : clés référence, montant "
                                 "et client file d'abord, puis factures du débiteur les plus proches en montant "
                                 "ou en échéance.")


class FeatureFamilies(_Model):
    amount: bool = Field(True, description="Montant : écarts, escompte, frais bancaires, retenue de garantie.")
    temporal: bool = Field(True, description="Temporel : délai à l'échéance, ancienneté de la facture.")
    textual: bool = Field(True, description="Textuel : référence exacte / partielle, similarité de nom.")
    identity: bool = Field(True, description="Identité : route IBAN, bankroll, canal, contrat.")
    behavioral: bool = Field(True, description="Comportemental : agrégats du débiteur, fenêtre strictement antérieure.")
    behavioral_window_days: int = Field(180, ge=1, description="Fenêtre glissante des agrégats comportementaux.")
    contract: bool = Field(True, description="Contexte contrat : market, product, recourse.")
    allocation: bool = Field(True, description="Score et signal d'allocation du débiteur.")
    client_file: bool = Field(True, description="Présence et concordance du client file.")


class SetSettings(_Model):
    enabled: bool = Field(True, description="Reconstitution des ensembles (1↔n, n↔n) par DFS borné.")
    near_sum: bool = Field(True, description="Passe 1 : sous-ensemble dont la somme est proche du paiement.")
    n_to_n: bool = Field(True, description="Passe 2 : agrégation des paiements d'un même débiteur.")
    n_to_n_window_hours: int = Field(72, ge=1, description="Fenêtre d'agrégation des paiements n↔n.")
    single_candidate: bool = Field(True, description="Passe 3 : candidat unique.")
    max_invoices: int = Field(5, ge=1, description="Cardinalité maximale d'un ensemble.")
    max_candidates: int = Field(25, ge=1, description="Candidats les mieux scorés en entrée du DFS.")
    tolerance_abs_cents: int = Field(500, ge=0, description="Tolérance absolue sur la somme (centimes).")
    tolerance_rel: float = Field(0.03, ge=0, le=1, description="Tolérance relative sur la somme.")
    node_budget: int = Field(100_000, ge=1, description="Budget de nœuds explorés ; au-delà → revue.")


class TrainingSettings(_Model):
    payment_sample: float = Field(0.5, gt=0, le=1, description="Part des paiements du résiduel conservés pour "
                                  "l'entraînement (tous leurs candidats sont gardés : pas d'échantillonnage des "
                                  "négatifs).")
    num_boost_round: int = Field(400, ge=10, description="Nombre maximal d'arbres (arrêt précoce sur la validation).")
    learning_rate: float = Field(0.05, gt=0, le=1, description="Taux d'apprentissage LightGBM.")
    num_leaves: int = Field(63, ge=2, description="Feuilles par arbre.")
    min_data_in_leaf: int = Field(50, ge=1, description="Observations minimales par feuille.")
    seed: int = Field(42, description="Graine (reproductibilité).")


class DecisionSettings(_Model):
    review_min_score: float = Field(0.3, ge=0, le=1, description="τ_low : sous ce score, pas de proposition "
                                    "(le paiement reste en attente) ; entre τ_low et τ_high, revue.")
    min_margin: float = Field(0.05, ge=0, le=1, description="δ : marge minimale au second candidat pour "
                              "l'auto-validation. À calibrer.")
    segmented_thresholds: bool = Field(False, description="Seuils par segment si le volume le permet.")
    segments: list[str] = Field(default_factory=lambda: ["market", "bankroll_code", "amount_bucket",
                                                          "has_client_file"],
                                description="Dimensions de segmentation des seuils.")
    min_segment_volume: int = Field(1000, ge=1, description="Volume de validation minimal pour qu'un "
                                    "segment ait son propre seuil.")


class LLMLabelSettings(_Model):
    enabled: bool = Field(False, description="Extraction LLM de références bruitées sur le résiduel "
                          "(optionnel, gardé seulement si le gain est mesurable).")
    endpoint: str | None = Field(None, description="URL de l'API LLM on-premise. Aucun autre appel réseau.")
    cache_dir: str = Field("data/cache/llm", description="Cache des réponses LLM.")


class MLSettings(_Model):
    candidates: CandidateSettings = Field(default_factory=CandidateSettings, description="Génération de candidats.")
    features: FeatureFamilies = Field(default_factory=FeatureFamilies, description="Familles de features actives.")
    second_pass: bool = Field(True, description="Deuxième passe avec features de compétition "
                              "(rang, marge au second, nombre de candidats).")
    calibration: bool = Field(True, description="Calibration isotonique des scores sur la validation.")
    training: TrainingSettings = Field(default_factory=TrainingSettings, description="Entraînement.")
    sets: SetSettings = Field(default_factory=SetSettings, description="Résolution des ensembles.")
    decision: DecisionSettings = Field(default_factory=DecisionSettings, description="Décision et seuils.")
    llm_labels: LLMLabelSettings = Field(default_factory=LLMLabelSettings, description="LLM sur libellés.")


# --- Évaluation (étape 6) ------------------------------------------------------------


class EvaluationSettings(_Model):
    target_precision: float = Field(0.995, gt=0, le=1, description="Précision visée : fixe τ_high et définit "
                                    "le taux d'automatisation à précision fixée.")
    current_automation_rate: float | None = Field(None, ge=0, le=1, description="Taux d'automatisation de "
                                                  "l'algorithme actuel (chiffre de référence client).")
    reference_total_volume: int | None = Field(None, ge=0, description="Volume total de paiements de référence.")
    reference_manual_volume: int | None = Field(None, ge=0, description="Volume traité en manuel de référence.")


class Settings(_Model):
    paths: PathsSettings = Field(default_factory=PathsSettings, description="Chemins.")
    load: LoadSettings = Field(default_factory=LoadSettings, description="Étape 1 — chargement.")
    split: SplitSettings = Field(default_factory=SplitSettings, description="Étape 2 — découpage temporel.")
    allocation: AllocationSettings = Field(default_factory=AllocationSettings, description="Étape 3 — allocation.")
    reconcile_ml: MLSettings = Field(default_factory=MLSettings, description="Étape 5 — réconciliation ML.")
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings, description="Étape 6 — évaluation.")


# --- Règles déterministes (étape 4) ------------------------------------------------


class RuleConfig(_Model):
    id: str
    name: str
    description: str
    enabled: bool = True
    priority: int = Field(ge=1)
    params: dict[str, int | float] = Field(default_factory=dict)


def _default_rules() -> list[RuleConfig]:
    # firm_only = 1 : la règle ne porte que sur le débiteur d'une allocation ferme ;
    # 0 : sur tous les débiteurs candidats de l'allocation.
    return [
        RuleConfig(id="R1_CLIENT_FILE", name="Client file concordant", priority=1,
                   description="Chaque ligne du client file se résout vers une seule facture ouverte, "
                               "montants concordants dans la tolérance : groupe entier validé.",
                   params={"tolerance_abs_cents": 0, "tolerance_rel": 0.0, "firm_only": 0}),
        RuleConfig(id="R2_REFERENCE_UNIQUE", name="Référence exacte unique", priority=2,
                   description="Une seule facture ouverte correspond à une référence du libellé, "
                               "montant égal au restant dû (dans la tolérance).",
                   params={"tolerance_abs_cents": 0, "firm_only": 0}),
        RuleConfig(id="R3_AMOUNT_UNIQUE", name="Montant exact unique", priority=3,
                   description="Une seule facture ouverte du débiteur a exactement le montant du paiement.",
                   params={"firm_only": 1}),
        RuleConfig(id="R4_EXACT_SUM", name="Somme exacte", priority=4,
                   description="Un sous-ensemble unique de factures ouvertes du débiteur somme exactement "
                               "au paiement (DFS borné).",
                   params={"max_invoices": 5, "node_budget": 100_000, "max_open_invoices": 30, "firm_only": 1}),
        RuleConfig(id="R5_PARTIAL_REFERENCED", name="Paiement partiel sur facture référencée", priority=5,
                   description="Référence unique, montant inférieur au restant dû : imputation PARTIAL.",
                   params={"firm_only": 0}),
    ]


class RulesConfig(_Model):
    version: int = Field(1, ge=1, description="Version du jeu de règles, tracée dans chaque décision. "
                         "À incrémenter à toute modification.")
    min_precision: float = Field(0.995, gt=0, le=1, description="Précision individuelle minimale : une règle "
                                 "mesurée en dessous est désactivée.")
    rules: list[RuleConfig] = Field(default_factory=_default_rules, description="Règles, appliquées par "
                                    "priorité croissante. Validation seulement si la solution est unique.")

    @model_validator(mode="after")
    def _unique(self) -> RulesConfig:
        ids = [r.id for r in self.rules]
        prios = [r.priority for r in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError("identifiants de règles dupliqués")
        if len(set(prios)) != len(prios):
            raise ValueError("priorités de règles dupliquées")
        return self

    def active(self) -> list[RuleConfig]:
        return sorted((r for r in self.rules if r.enabled), key=lambda r: r.priority)


# --- Lecture / écriture ------------------------------------------------------------

_SETTINGS_HEADER = """
Paramètres de la pipeline, par étape. Édité à la main, depuis le notebook ou par l'UI (streamlit run reconciliation_ui.py).
Les règles de l'étape 4 sont dans config/rules.yaml.
"""

_RULES_HEADER = """
Règles déterministes de l'étape 4 (réconciliation algorithmique) — déclaratives et versionnées.
Chaque décision garde l'identifiant de la règle et la version de ce fichier.
"""


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    p = Path(path)
    return Settings.model_validate(read_yaml(p) if p.exists() else {})


def save_settings(settings: Settings, path: str | Path = DEFAULT_SETTINGS_PATH) -> None:
    Path(path).write_text(dump_model(settings, _SETTINGS_HEADER), encoding="utf-8")


def load_rules(path: str | Path = DEFAULT_RULES_PATH) -> RulesConfig:
    p = Path(path)
    return RulesConfig.model_validate(read_yaml(p) if p.exists() else {})


def save_rules(rules: RulesConfig, path: str | Path = DEFAULT_RULES_PATH) -> None:
    Path(path).write_text(dump_model(rules, _RULES_HEADER), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/arrow_ops.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Recherches par clé vectorisées via pyarrow.

Sur les colonnes de chaînes pyarrow, `Series.isin` et `Series.map(Series)` de
pandas repassent par des objets Python : ~10 s sur 2 M de lignes, contre
< 1 s ici. À utiliser pour toute jointure par identifiant à grande échelle.
"""




def to_arrow(values: pd.Series | pd.Index | np.ndarray | list) -> pa.Array:
    if isinstance(values, (pd.Series, pd.Index)):
        values = values.array
    arr = pa.array(values, from_pandas=True)
    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr


def isin(values: pd.Series, reference: pd.Series) -> np.ndarray:
    """Masque booléen : valeur présente dans `reference` (NA → False)."""
    value_set = to_arrow(pd.Series(reference).dropna().drop_duplicates())
    mask = pc.is_in(to_arrow(values), value_set=value_set).fill_null(False)
    return mask.to_numpy(zero_copy_only=False)


def lookup(keys: pd.Series, index_keys: pd.Series, values: pd.Series) -> pd.Series:
    """Pour chaque clé, la valeur de `values` à la première position où `index_keys` vaut la clé.

    Clé absente ou manquante → NA. Résultat aligné sur l'index de `keys`.
    """
    first = ~pd.Series(index_keys).duplicated().to_numpy()
    idx_keys = pd.Series(index_keys)[first]
    vals = pd.Series(values)[first].reset_index(drop=True)
    pos = pc.index_in(to_arrow(keys), value_set=to_arrow(idx_keys)).fill_null(-1)
    pos = pos.to_numpy(zero_copy_only=False)
    found = pos >= 0
    out = vals.iloc[np.where(found, pos, 0)] if len(vals) else pd.Series([pd.NA] * len(pos))
    out = out.reset_index(drop=True).where(found)
    out.index = keys.index
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/memory.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Suivi de la consommation mémoire pendant l'exécution (notebook, pod à mémoire limitée).

    from src.memory import MemoryMonitor
    monitor = MemoryMonitor("reports/memory.csv").start()
    project = Project(dataset="real", log=monitor.log)     # chaque message suivi de la mémoire
    ...
    monitor.by_phase()                                    # pic par étape / sous-étape

Un fil d'arrière-plan relève chaque seconde :
- la mémoire résidente du processus (RSS) et son pic ;
- la mémoire du conteneur (cgroup v1 ou v2) : consommée, anonyme (ce que l'OOM killer compte
  réellement, hors cache disque) et limite du pod ;
- la phase en cours : étape de `Project`, jour du rejeu, sous-étape (signal d'allocation, règle,
  candidats ML...), posée par `mark_step` / `mark_day` / `mark` depuis le code de la pipeline.

Chaque relevé est **écrit et vidé immédiatement** dans un CSV : si le noyau est tué (OOM), le fichier
reste, et `by_phase(path)` / `chart(path)` montrent, dans un nouveau noyau, la phase où la mémoire a
explosé. `tail -f` sur ce fichier depuis un terminal JupyterLab donne un suivi en direct.

`mark*` ne coûtent rien sans moniteur actif. Aucun appel réseau.
"""




GB = 2 ** 30
COLUMNS = ["time", "elapsed_s", "rss_gb", "rss_peak_gb", "pod_used_gb", "pod_anon_gb", "pod_limit_gb",
           "step", "day", "sub"]

_START = "<démarrage du suivi>"                  # ligne séparant les sessions dans le CSV

_ACTIVE: MemoryMonitor | None = None


# --- Relevés -------------------------------------------------------------------------------------------

def process_rss() -> int:
    """Mémoire résidente du processus, en octets (psutil si présent, sinon /proc)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        pass
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _read_int(path: str) -> int | None:
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return None
    return None if not text or text == "max" else int(text)


def _stat(path: str, key: str) -> int | None:
    try:
        for line in Path(path).read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == key:
                return int(value)
    except OSError:
        pass
    return None


def pod_memory() -> dict[str, int | None]:
    """Mémoire du conteneur (cgroup v2, sinon v1) : consommée, anonyme, limite. None hors conteneur."""
    if Path("/sys/fs/cgroup/memory.current").exists():                      # cgroup v2
        return {"used": _read_int("/sys/fs/cgroup/memory.current"),
                "anon": _stat("/sys/fs/cgroup/memory.stat", "anon"),
                "limit": _read_int("/sys/fs/cgroup/memory.max")}
    base = "/sys/fs/cgroup/memory/"                                          # cgroup v1
    if Path(base + "memory.usage_in_bytes").exists():
        limit = _read_int(base + "memory.limit_in_bytes")
        return {"used": _read_int(base + "memory.usage_in_bytes"),
                "anon": _stat(base + "memory.stat", "total_rss"),
                "limit": limit if limit and limit < 2 ** 60 else None}      # « illimité » = très grand nombre
    return {"used": None, "anon": None, "limit": None}


def release_memory() -> None:
    """Rend au système la mémoire libérée : ramasse-miettes, puis `malloc_trim` (glibc, Linux).

    Sans `malloc_trim`, la mémoire libérée par pandas / numpy reste souvent réservée au processus : le
    RSS ne redescend pas d'un jour ou d'une étape à l'autre et le pod finit par être tué.
    """
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


# --- Phases (appelées depuis la pipeline) ----------------------------------------------------------------

def mark(sub: str) -> None:
    """Sous-étape en cours (signal d'allocation, règle, candidats ML...)."""
    if _ACTIVE is not None:
        _ACTIVE.sub = sub


def mark_step(step: str) -> None:
    """Étape en cours (méthode de `Project`) ; remet à zéro jour et sous-étape."""
    if _ACTIVE is not None:
        _ACTIVE.step, _ACTIVE.day, _ACTIVE.sub = step, "", ""


def mark_day(day, batch: int) -> None:
    """Début d'un jour du rejeu (appelé par `run_replay`)."""
    if _ACTIVE is not None:
        _ACTIVE.day, _ACTIVE.sub = f"{pd.Timestamp(day).date()} (lot {batch})", ""


def end_day() -> None:
    """Fin d'un jour du rejeu : mémoire rendue au système si le moniteur le demande."""
    if _ACTIVE is not None and _ACTIVE.trim_daily:
        release_memory()


# --- Moniteur ---------------------------------------------------------------------------------------------

def _gb(value: int | None) -> float | None:
    return None if value is None else round(value / GB, 3)


class MemoryMonitor:
    """Relevé périodique de la mémoire dans un CSV (vidé à chaque ligne) et suivi des phases.

    `trim_daily` : rend la mémoire libérée au système à la fin de chaque jour du rejeu.
    `warn_ratio` : part de la limite du pod au-delà de laquelle les messages portent une alerte.
    """

    def __init__(self, path: str | Path = "reports/memory.csv", interval: float = 1.0, trim_daily: bool = True,
                 warn_ratio: float = 0.85, echo: Callable[[str], None] = print):
        self.path = Path(path)
        self.interval = interval
        self.trim_daily = trim_daily
        self.warn_ratio = warn_ratio
        self.echo = echo
        self.step = self.day = self.sub = ""
        self.peak = 0
        self.last: dict = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._file = None
        self._t0 = time.time()
        self._live = None

    # Démarrage / arrêt ------------------------------------------------------------------------------------

    def start(self) -> MemoryMonitor:
        global _ACTIVE
        if _ACTIVE is not None and _ACTIVE is not self:
            _ACTIVE.stop()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists() or self.path.stat().st_size == 0
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if fresh:
            self._writer.writerow(COLUMNS)
        self._writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), *[""] * 6, _START, "", ""])
        self._t0 = time.time()
        self._stop.clear()
        _ACTIVE = self
        self.sample()
        self._thread = threading.Thread(target=self._run, name="memory-monitor", daemon=True)
        self._thread.start()
        limit = self.last.get("pod_limit_gb")
        self.echo(f"[mémoire] suivi → {self.path.resolve()} ; limite du pod : "
                  f"{f'{limit:.1f} Go' if limit else 'non détectée (hors conteneur ?)'}")
        return self

    def stop(self) -> None:
        global _ACTIVE
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            if self._file is not None and not self._file.closed:
                self._file.close()
        if _ACTIVE is self:
            _ACTIVE = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
                if self._live is not None:
                    self._live.update(self._badge())
            except Exception:                    # le suivi ne doit jamais interrompre la pipeline
                pass

    def sample(self) -> dict:
        rss = process_rss()
        self.peak = max(self.peak, rss)
        pod = pod_memory()
        row = {"time": time.strftime("%H:%M:%S"), "elapsed_s": round(time.time() - self._t0, 1),
               "rss_gb": _gb(rss), "rss_peak_gb": _gb(self.peak), "pod_used_gb": _gb(pod["used"]),
               "pod_anon_gb": _gb(pod["anon"]), "pod_limit_gb": _gb(pod["limit"]),
               "step": self.step, "day": self.day, "sub": self.sub}
        self.last = row
        with self._lock:
            if self._file is not None and not self._file.closed:
                self._writer.writerow([row[c] for c in COLUMNS])
                self._file.flush()
        return row

    # Affichage ---------------------------------------------------------------------------------------------

    def status(self) -> str:
        r = self.last or self.sample()
        text = f"RSS {r['rss_gb']:.2f} Go (pic {r['rss_peak_gb']:.2f})"
        if r["pod_used_gb"] is not None:
            limit = r["pod_limit_gb"]
            anon = f", anonyme {r['pod_anon_gb']:.2f}" if r["pod_anon_gb"] is not None else ""
            text += f" · pod {r['pod_used_gb']:.2f}{f'/{limit:.1f}' if limit else ''} Go{anon}"
            used = r["pod_anon_gb"] if r["pod_anon_gb"] is not None else r["pod_used_gb"]
            if limit and used >= self.warn_ratio * limit:
                text += " ⚠ PROCHE DE LA LIMITE"
        return text

    def log(self, message: str) -> None:
        """À passer comme `log` de `Project` : chaque message est suivi de l'état mémoire."""
        if message.startswith("… "):
            mark_step(message[2:])
        self.echo(f"{message}   [{self.status()}]")

    def _badge(self):
        from IPython.display import HTML
        phase = " › ".join(p for p in (self.step, self.day, self.sub) if p) or "—"
        return HTML(f"<code>{self.last.get('time', '')} · {self.status()}<br>{phase}</code>")

    def live(self) -> None:
        """Pastille mise à jour chaque seconde dans la sortie de la cellule qui l'appelle (Jupyter)."""
        from IPython.display import display
        self._live = display(self._badge(), display_id=True)

    # Analyse -----------------------------------------------------------------------------------------------

    def history(self) -> pd.DataFrame:
        return read_history(self.path)

    def by_phase(self) -> pd.DataFrame:
        return by_phase(self.path)

    def chart(self):
        return chart(self.path)


# --- Analyse, y compris après un arrêt brutal du noyau -------------------------------------------------------

def read_history(path: str | Path = "reports/memory.csv") -> pd.DataFrame:
    """Relevés du CSV ; une session par démarrage du suivi (ligne « démarrage » écrite par `start`)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["session"] = (df["step"] == _START).cumsum()
    df = df[df["step"] != _START].copy()
    for c in COLUMNS:
        if c.endswith("_gb") or c == "elapsed_s":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def _session(df: pd.DataFrame, session: int | None) -> pd.DataFrame:
    if session is None or df.empty:
        return df
    return df[df["session"] == df["session"].unique()[session]]


def by_phase(path: str | Path = "reports/memory.csv", session: int | None = -1) -> pd.DataFrame:
    """Pic de mémoire par (étape, sous-étape), dans l'ordre d'exécution, pour la dernière session
    (`session=None` : toutes). Mesure : mémoire anonyme du pod si disponible, sinon RSS du processus."""
    df = _session(read_history(path), session)
    mem = "pod_anon_gb" if df["pod_anon_gb"].notna().any() else "rss_gb"
    keys = ["step", "sub"]
    out = (df.groupby(keys, sort=False)
           .agg(début=("time", "first"), fin=("time", "last"), relevés=("time", "size")).reset_index())
    at_peak = df.sort_values(mem, ascending=False, kind="stable").drop_duplicates(keys)[[*keys, mem, "day"]]
    return out.merge(at_peak, on=keys, how="left").rename(
        columns={"step": "étape", "sub": "sous-étape", mem: f"pic_{mem}", "day": "jour_du_pic"})


def chart(path: str | Path = "reports/memory.csv", session: int = -1):
    """Courbe mémoire de la session (altair) ; limite du pod en rouge, phases en infobulle."""
    import altair as alt
    df = _session(read_history(path), session)
    cols = [c for c in ("rss_gb", "pod_anon_gb", "pod_used_gb") if df[c].notna().any()]
    long = df.melt(id_vars=["elapsed_s", "step", "day", "sub"], value_vars=cols, var_name="mesure",
                   value_name="Go")
    layers = [alt.Chart(long).mark_line().encode(
        x=alt.X("elapsed_s:Q", title="secondes"), y=alt.Y("Go:Q"), color="mesure:N",
        tooltip=["elapsed_s", "mesure", "Go", "step", "day", "sub"])]
    limit = df["pod_limit_gb"].dropna()
    if len(limit):
        layers.append(alt.Chart(pd.DataFrame({"y": [limit.iloc[-1]]}))
                      .mark_rule(color="red", strokeDash=[4, 4]).encode(y="y:Q"))
    return alt.layer(*layers).properties(height=260, title="Mémoire (limite du pod en rouge)")


# Espace de noms « memory » (src/memory.py dans le dépôt).
memory = SimpleNamespace(**{n: globals()[n] for n in (
    'MemoryMonitor', 'by_phase', 'chart', 'end_day', 'mark', 'mark_day', 'mark_step',
    'pod_memory', 'process_rss', 'read_history', 'release_memory')})


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/load/canonical.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Modèle de données canonique du POC.

Ce sont les noms *internes* (ceux du brief et de la spec §2). Les noms réels des
tables et colonnes sont fournis par `config/schema.yaml`, jamais déduits ici.

`required=True` sur un champ : la pipeline ne peut pas fonctionner sans, le
mapping doit être renseigné. Un champ optionnel non mappé est ajouté vide et
signalé dans le rapport de chargement.
"""




class FieldType(str, Enum):
    ID = "id"                # identifiant, stocké en chaîne
    TEXT = "text"
    DATE = "date"            # date sans heure (minuit)
    TIMESTAMP = "timestamp"  # horodatage, converti en UTC naïf
    AMOUNT = "amount"        # entier de centimes


@dataclass(frozen=True)
class Field:
    name: str
    type: FieldType
    required: bool = True


@dataclass(frozen=True)
class Table:
    name: str
    fields: tuple[Field, ...]
    required: bool = True
    primary_key: tuple[str, ...] = ()

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"{self.name}.{name}")


T = FieldType

PAYMENT = Table(
    "payment",
    (
        Field("payment_id", T.ID),
        Field("value_date", T.DATE),
        # Date de comptabilisation / de connaissance, si elle existe (§3.5).
        # Si mappée, c'est elle qui ordonne le journal.
        Field("booking_date", T.DATE, required=False),
        Field("amount", T.AMOUNT),
        Field("currency", T.TEXT),
        Field("iban_debtor", T.TEXT, required=False),
        Field("iban_creditor", T.TEXT, required=False),
        Field("label", T.TEXT),
        Field("channel", T.TEXT, required=False),
        Field("payment_type", T.TEXT, required=False),
        Field("bankroll_code", T.TEXT, required=False),
    ),
    primary_key=("payment_id",),
)

INVOICE = Table(
    "invoice",
    (
        Field("invoice_id", T.ID),
        Field("client_reference", T.TEXT),
        Field("internal_reference", T.TEXT, required=False),
        Field("creation_date", T.DATE),
        Field("due_date", T.DATE),
        Field("initial_amount", T.AMOUNT),
        # Restant dû *final* : fuite directe. Chargé uniquement pour l'audit de
        # cohérence des imputations, retiré de la table `invoice` au chargement.
        Field("current_amount", T.AMOUNT, required=False),
        Field("currency", T.TEXT),
        Field("debtor_id", T.ID),
        Field("agreement_id", T.ID),
    ),
    primary_key=("invoice_id",),
)

IMPUTATION = Table(
    "imputation",
    (
        Field("payment_id", T.ID),
        Field("invoice_id", T.ID),
        Field("status", T.TEXT),              # FULL / PARTIAL après value_maps
        Field("updated_at", T.TIMESTAMP),
        Field("residual_amount", T.AMOUNT),   # solde de la facture APRÈS la ligne
    ),
)


def _party(name: str) -> Table:
    return Table(
        name,
        (
            Field("party_id", T.ID),
            Field("bankroll_code", T.TEXT, required=False),
            Field("iban", T.TEXT, required=False),
            Field("name", T.TEXT),
            Field("opened_at", T.DATE, required=False),
            Field("closed_at", T.DATE, required=False),
        ),
        primary_key=("party_id",),
    )


ASSIGNOR = _party("assignor")
DEBTOR = _party("debtor")

AGREEMENT = Table(
    "agreement",
    (
        Field("agreement_id", T.ID),
        Field("debtor_id", T.ID),
        Field("client_id", T.ID),
        Field("contract_number", T.TEXT, required=False),
        Field("created_at", T.DATE),
        Field("disabled_at", T.DATE, required=False),
        Field("market", T.TEXT, required=False),
        Field("product", T.TEXT, required=False),
        Field("recourse", T.TEXT, required=False),
    ),
    primary_key=("agreement_id",),
)

# Référentiel des IBAN de comptes techniques (§3.5) — optionnel.
TECHNICAL_ACCOUNT = Table(
    "technical_account",
    (
        Field("iban", T.TEXT),
        Field("bankroll_code", T.TEXT, required=False),
        Field("description", T.TEXT, required=False),
    ),
    required=False,
    primary_key=("iban",),
)

# Client files (§3.2) — format réel à confirmer. Seule la voie tabulaire
# (fichiers déjà structurés en lignes) est branchée ici.
CLIENT_FILE = Table(
    "client_file",
    (
        Field("file_id", T.ID),
        Field("received_at", T.TIMESTAMP),
        Field("source_format", T.TEXT, required=False),
        Field("payment_reference", T.TEXT, required=False),
        Field("total_amount", T.AMOUNT, required=False),
        Field("payment_date", T.DATE, required=False),
        Field("iban", T.TEXT, required=False),
        Field("issuer_name", T.TEXT, required=False),
    ),
    required=False,
    primary_key=("file_id",),
)

CLIENT_FILE_LINE = Table(
    "client_file_line",
    (
        Field("file_id", T.ID),
        Field("line_no", T.ID, required=False),
        Field("invoice_reference", T.TEXT),
        Field("amount", T.AMOUNT, required=False),
        Field("gap_reason", T.TEXT, required=False),
    ),
    required=False,
)

TABLES: dict[str, Table] = {
    t.name: t
    for t in (
        PAYMENT, INVOICE, IMPUTATION, ASSIGNOR, DEBTOR, AGREEMENT,
        TECHNICAL_ACCOUNT, CLIENT_FILE, CLIENT_FILE_LINE,
    )
}

IMPUTATION_STATUSES = ("FULL", "PARTIAL")


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/load/normalize.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Normalisation des libellés, références et noms (brief §3.3, spec §3.1).

Déterministe et versionnée : toute modification du comportement doit
incrémenter `NORMALIZATION_VERSION` et relancer la chaîne depuis l'étape 1.

Vues produites pour un libellé :
- `label_norm`    : texte normalisé (majuscules, sans accents, alphanumérique).
- `label_tokens`  : tokens alphabétiques.
- `label_numbers` : clés de référence candidates — tokens contenant des
  chiffres, fenêtres de tokens adjacents concaténés, inversion bloc lettres /
  bloc chiffres, puis variantes sans préfixe alphabétique et sans zéros de tête.

Une référence de facture est réduite à ses `reference_keys` par la même
fonction de variantes : un match de référence = intersection non vide entre
`label_numbers` et `reference_keys`.

Volumétrie : ~2 M de paiements, ~3 M de factures. La normalisation du texte
est vectorisée (`normalize_series`) et c'est l'unique implémentation — la
version scalaire `normalize_text` l'appelle. Les vues à base de tokens sont
calculées une seule fois par valeur distincte, réparties sur plusieurs
processus si un `Executor` est fourni (résultat identique, ordre conservé).
"""




NORMALIZATION_VERSION = "1.1.0"

# Fenêtre maximale de tokens adjacents concaténés dans un libellé. Le brief
# demande les paires (inversion) ; la fenêtre de 3 couvre en plus les
# références découpées par la ponctuation (`FA-2024-00123` → `FA 2024 00123`).
MAX_JOIN_WINDOW = 3

# « ß » avant la mise en majuscules (sa majuscule simple n'est pas « SS ») ;
# lettres que NFKD ne décompose pas, après.
_BEFORE_UPPER = {"ß": "SS"}
_AFTER_UPPER = {"Œ": "OE", "Æ": "AE", "Ø": "O", "Ð": "D", "Þ": "TH", "Ł": "L"}
_ALPHA_DIGITS = re.compile(r"([A-Z]+)([0-9]+)")
_DIGITS_ALPHA = re.compile(r"([0-9]+)([A-Z]+)")
_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --- Texte ------------------------------------------------------------------


def normalize_series(series: pd.Series) -> pd.Series:
    """Majuscules, sans accents (NFKD), non alphanumérique → espace, espaces compressés.

    Valeur manquante → chaîne vide. Retourne une série de chaînes sans NA.
    """
    s = series.astype("str").fillna("")
    for k, v in _BEFORE_UPPER.items():
        s = s.str.replace(k, v, regex=False)
    s = s.str.normalize("NFKD").str.replace(r"\p{M}+", "", regex=True).str.upper()
    for k, v in _AFTER_UPPER.items():
        s = s.str.replace(k, v, regex=False)
    return s.str.replace(r"[^A-Z0-9]+", " ", regex=True).str.strip()


def normalize_text(value: Any) -> str:
    """Version scalaire de `normalize_series` (même implémentation)."""
    return normalize_series(pd.Series([value], dtype=object)).iloc[0]


def tokenize(value: Any) -> list[str]:
    return normalize_text(value).split()


# En dessous de ce nombre de valeurs distinctes, la parallélisation coûte plus qu'elle ne rapporte.
PARALLEL_MIN_VALUES = 100_000
_CHUNK_SIZE = 50_000


def _apply_chunk(fn: Callable[[Any], Any], values: list[Any]) -> list[Any]:
    return [fn(v) for v in values]


def _map_unique(series: pd.Series, fn: Callable[[Any], Any], executor: Executor | None = None) -> pd.Series:
    """Applique `fn` une fois par valeur distincte (`fn` doit être picklable si `executor`)."""
    codes, uniques = pd.factorize(series, use_na_sentinel=False)
    values = list(uniques)
    if executor is not None and len(values) >= PARALLEL_MIN_VALUES:
        chunks = [values[i:i + _CHUNK_SIZE] for i in range(0, len(values), _CHUNK_SIZE)]
        results = list(chain.from_iterable(executor.map(_apply_chunk, [fn] * len(chunks), chunks)))
    else:
        results = _apply_chunk(fn, values)
    return pd.Series([results[c] for c in codes], index=series.index, dtype=object)


# --- Références ---------------------------------------------------------------


def number_variants(token: str, swap: bool = False) -> set[str]:
    """Variantes d'un token normalisé contenant des chiffres.

    `FA0012345`             → {FA0012345, 0012345, 12345}
    `123FACT`, swap=True    → {123FACT, FACT123, 123}  (inversion dans le token)
    """
    out = {token}
    without_prefix = token.lstrip(_ALPHA)
    if without_prefix:
        out.add(without_prefix)
        without_zeros = without_prefix.lstrip("0")
        if without_zeros:
            out.add(without_zeros)
    if swap:
        for pattern in (_ALPHA_DIGITS, _DIGITS_ALPHA):
            m = pattern.fullmatch(token)
            if m:
                out |= number_variants(m.group(2) + m.group(1))
    return out


def label_tokens(tokens: Iterable[str]) -> list[str]:
    return [t for t in tokens if t.isalpha()]


def label_numbers(tokens: list[str], max_window: int = MAX_JOIN_WINDOW) -> list[str]:
    """Clés de référence candidates extraites d'un libellé tokenisé (triées).

    Une fenêtre de `size` tokens adjacents n'est concaténée que si au moins
    `size − 1` d'entre eux contiennent des chiffres (`FA 2024 00123` oui,
    `SARL FACT 001887` non).
    """
    # Tokens normalisés = [A-Z0-9]+ : « contient un chiffre » ⇔ « pas alphabétique ».
    has_digit = [not t.isalpha() for t in tokens]
    if not any(has_digit):
        return []
    keys: set[str] = set()
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if has_digit[i]:
            keys |= number_variants(tok, swap=True)
        for size in range(2, max_window + 1):
            if i + size > n:
                break
            if sum(has_digit[i:i + size]) < size - 1:
                continue
            window = tokens[i:i + size]
            keys |= number_variants("".join(window))
            if size == 2:
                # Inversion bloc lettres / bloc chiffres : `123 FACT` ↔ `FACT123`.
                keys |= number_variants(window[1] + window[0])
    return sorted(keys)


def _label_tokens_of_norm(norm: str) -> list[str]:
    return label_tokens(norm.split())


def _label_numbers_of_norm(norm: str) -> list[str]:
    return label_numbers(norm.split())


def _keys_of_compact(compact: str) -> list[str]:
    if not compact:
        return []
    if compact.isalpha():
        return [compact]
    return sorted(number_variants(compact))


def reference_keys(reference: Any) -> list[str]:
    """Clés d'une référence de facture (triées). Vide si la référence est vide."""
    return _keys_of_compact(normalize_text(reference).replace(" ", ""))


# --- Application aux tables chargées ---------------------------------------


def enrich_label(series: pd.Series, prefix: str = "label", executor: Executor | None = None) -> pd.DataFrame:
    norm = normalize_series(series)
    return pd.DataFrame(
        {
            f"{prefix}_norm": norm,
            f"{prefix}_tokens": _map_unique(norm, _label_tokens_of_norm),
            f"{prefix}_numbers": _map_unique(norm, _label_numbers_of_norm, executor),
        },
        index=series.index,
    )


def enrich_reference(series: pd.Series, prefix: str, executor: Executor | None = None) -> pd.DataFrame:
    norm = normalize_series(series)
    compact = norm.str.replace(" ", "", regex=False)
    return pd.DataFrame(
        {f"{prefix}_norm": norm, f"{prefix}_keys": _map_unique(compact, _keys_of_compact, executor)},
        index=series.index,
    )


def enrich_name(series: pd.Series, prefix: str = "name") -> pd.DataFrame:
    norm = normalize_series(series)
    return pd.DataFrame(
        {f"{prefix}_norm": norm, f"{prefix}_tokens": _map_unique(norm, str.split)},
        index=series.index,
    )


def normalize_iban(series: pd.Series) -> pd.Series:
    """IBAN sans espaces ni séparateurs, en majuscules ; vide → NA."""
    out = normalize_series(series).str.replace(" ", "", regex=False)
    return out.where(out != "", pd.NA).astype("string")


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/load/loader.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Chargement des tables sources vers le modèle canonique (brief §3).

Tout le mapping vers les noms réels vient de `config/schema.yaml`. Le loader
ne devine rien : un mapping requis manquant lève `SchemaConfigError` avec la
liste complète de ce qui reste à renseigner.

Conventions de sortie :
- identifiants en chaînes (`string`) ;
- montants en entiers de centimes (`Int64`) ;
- dates et horodatages en `datetime64[us]` naïfs exprimés en UTC.
"""





DATETIME_DTYPE = "datetime64[us]"


class SchemaConfigError(ValueError):
    """La configuration de schéma est incomplète ou incohérente."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("config/schema.yaml incomplet :\n  - " + "\n  - ".join(problems))


@dataclass
class Issue:
    table: str
    field: str | None
    kind: str
    count: int
    detail: str = ""


@dataclass
class LoadedData:
    tables: dict[str, pd.DataFrame]
    # Champs canoniques effectivement mappés, par table.
    mapped_fields: dict[str, list[str]]
    issues: list[Issue] = field(default_factory=list)
    # Données chargées pour contrôle uniquement, jamais exposées à la pipeline.
    audit: dict[str, pd.DataFrame] = field(default_factory=dict)

    def has_table(self, name: str) -> bool:
        return name in self.tables


# --- Conversion de types ----------------------------------------------------


def _to_id(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.where(out != "", pd.NA)


def _to_text(s: pd.Series) -> pd.Series:
    return _to_id(s)


def _parse_datetime(s: pd.Series, fmt: str | None) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, format=fmt, errors="coerce")


def _to_date(s: pd.Series, fmt: str | None) -> pd.Series:
    parsed = _parse_datetime(s, fmt)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize().astype(DATETIME_DTYPE)


def _to_timestamp(s: pd.Series, fmt: str | None, source_tz: str | None) -> pd.Series:
    parsed = _parse_datetime(s, fmt)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    elif source_tz:
        parsed = (
            parsed.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
    return parsed.astype(DATETIME_DTYPE)


_AMOUNT_PATTERN = r"^(?P<sign>[+-]?)(?P<integer>[0-9]*)(?:\.(?P<frac>[0-9]*))?$"


def _to_cents(s: pd.Series, unit: str, decimal_sep: str) -> tuple[pd.Series, int]:
    """Convertit en entiers de centimes, sans jamais passer par un flottant pour le texte.

    `unit` : "cents" (valeurs déjà en centimes) ou "units" (ex. 1234.56).
    Retourne (série Int64, nombre de valeurs non vides invalides). Une valeur
    avec plus de décimales que l'unité n'en permet (hors zéros) est invalide.
    """
    scale_digits = 2 if unit == "units" else 0
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if pd.api.types.is_integer_dtype(s):
            return s.astype("Int64") * 10**scale_digits, 0
        scaled = s.astype("Float64") * 10**scale_digits
        rounded = scaled.round()
        ok = (scaled - rounded).abs() < 1e-6
        invalid = int((scaled.notna() & ~ok.fillna(False)).sum())
        return rounded.where(ok).astype("Int64"), invalid

    text = s.astype("str").str.replace(r"[\s ]+", "", regex=True)
    text = text.where(text != "")
    if decimal_sep != ".":
        text = text.str.replace(".", "", regex=False).str.replace(decimal_sep, ".", regex=False)
    arr = to_arrow(text).cast(pa.string())
    parts = pc.extract_regex(arr, _AMOUNT_PATTERN)
    integer, frac = parts.field("integer"), parts.field("frac")
    extra = pc.utf8_slice_codeunits(frac, scale_digits, 1 << 30)
    ok = pc.and_kleene(
        pc.is_valid(parts),
        pc.and_kleene(
            pc.greater(pc.add(pc.utf8_length(integer), pc.utf8_length(frac)), 0),
            pc.match_substring_regex(extra, "^0*$"),
        ),
    ).fill_null(False)
    head = pc.utf8_rpad(pc.utf8_slice_codeunits(frac, 0, scale_digits), scale_digits, "0")
    digits = pc.binary_join_element_wise(integer, head, "")
    digits = pc.if_else(pc.and_(ok, pc.greater(pc.utf8_length(digits), 0)), digits, "0")
    cents = pc.cast(digits, pa.int64())
    cents = pc.if_else(pc.equal(parts.field("sign"), "-"), pc.negate(cents), cents)
    ok_np = ok.to_numpy(zero_copy_only=False)
    out = pd.Series(cents.to_numpy(zero_copy_only=False), index=s.index).astype("Int64").where(ok_np)
    invalid = int((text.notna().to_numpy() & ~ok_np).sum())
    return out.astype("Int64"), invalid


# --- Lecture ----------------------------------------------------------------


def _read_source(path: Path, read_options: dict[str, Any]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **read_options)
    if suffix in (".csv", ".txt", ".tsv"):
        opts = {"dtype": str, "keep_default_na": False, "na_values": [""], **read_options}
        return pd.read_csv(path, **opts)
    raise SchemaConfigError([f"{path}: format de fichier non supporté ({suffix})"])


def _check_config(schema_cfg: dict[str, Any], base_dir: Path) -> list[str]:
    problems: list[str] = []
    if schema_cfg.get("amount_unit") not in ("cents", "units"):
        problems.append("amount_unit : renseigner 'cents' ou 'units'")
    tables_cfg = schema_cfg.get("tables") or {}
    for name, table in TABLES.items():
        tcfg = tables_cfg.get(name) or {}
        source = tcfg.get("source")
        if not source:
            if not table.required:
                continue
            problems.append(f"tables.{name}.source : fichier source non renseigné")
        elif not (base_dir / source).exists():
            problems.append(f"tables.{name}.source : fichier introuvable ({base_dir / source})")
        columns = tcfg.get("columns") or {}
        unknown = sorted(set(columns) - {f.name for f in table.fields})
        if unknown:
            problems.append(f"tables.{name}.columns : champs canoniques inconnus {unknown}")
        for f in table.fields:
            if f.required and not columns.get(f.name):
                problems.append(f"tables.{name}.columns.{f.name} : colonne réelle non renseignée")
    return problems


def _load_table(
    table: Table,
    tcfg: dict[str, Any],
    schema_cfg: dict[str, Any],
    base_dir: Path,
    issues: list[Issue],
) -> tuple[pd.DataFrame, list[str]]:
    raw = _read_source(base_dir / tcfg["source"], tcfg.get("read_options") or {})
    columns: dict[str, str] = {k: v for k, v in (tcfg.get("columns") or {}).items() if v}

    missing_cols = sorted(v for v in columns.values() if v not in raw.columns)
    if missing_cols:
        raise SchemaConfigError(
            [f"tables.{table.name} : colonnes absentes du fichier source {missing_cols}"]
        )

    unit = schema_cfg["amount_unit"]
    decimal_sep = schema_cfg.get("decimal_separator") or "."
    date_fmt = tcfg.get("date_format") or schema_cfg.get("date_format")
    ts_fmt = tcfg.get("timestamp_format") or schema_cfg.get("timestamp_format")
    source_tz = schema_cfg.get("source_timezone")
    value_maps: dict[str, dict[str, str]] = tcfg.get("value_maps") or {}

    out = pd.DataFrame(index=raw.index)
    for f in table.fields:
        if f.name not in columns:
            out[f.name] = _empty(f.type, len(raw), raw.index)
            continue
        col = raw[columns[f.name]]
        non_null_in = int(col.notna().sum())
        if f.type is FieldType.ID:
            conv = _to_id(col)
        elif f.type is FieldType.TEXT:
            conv = _to_text(col)
        elif f.type is FieldType.DATE:
            conv = _to_date(col, date_fmt)
        elif f.type is FieldType.TIMESTAMP:
            conv = _to_timestamp(col, ts_fmt, source_tz)
        elif f.type is FieldType.AMOUNT:
            conv, _ = _to_cents(col, unit, decimal_sep)
        else:  # pragma: no cover
            raise AssertionError(f.type)

        if f.name in value_maps:
            # value_maps : {valeur_canonique: [valeurs réelles]} ou {canonique: réelle}
            reverse: dict[str, str] = {}
            for canonical_value, real in value_maps[f.name].items():
                for r in real if isinstance(real, list) else [real]:
                    reverse[str(r)] = canonical_value
            conv = conv.map(lambda v: reverse.get(v, v) if not pd.isna(v) else v).astype("string")

        lost = non_null_in - int(conv.notna().sum())
        if lost > 0:
            issues.append(Issue(table.name, f.name, "valeur_invalide", lost,
                                "valeurs non vides non convertibles, mises à NA"))
        out[f.name] = conv

    return out.reset_index(drop=True), sorted(columns)


def _empty(ftype: FieldType, n: int, index: pd.Index) -> pd.Series:
    if ftype in (FieldType.DATE, FieldType.TIMESTAMP):
        return pd.Series(pd.NaT, index=index, dtype=DATETIME_DTYPE)
    if ftype is FieldType.AMOUNT:
        return pd.Series(pd.NA, index=index, dtype="Int64")
    return pd.Series(pd.NA, index=index, dtype="string")


def _enrich(tables: dict[str, pd.DataFrame], executor: Executor | None = None) -> None:
    """Applique la normalisation (brief §3.3) aux libellés, références et noms."""
    pay = tables["payment"]
    for col in ("iban_debtor", "iban_creditor"):
        pay[col] = normalize_iban(pay[col])
    tables["payment"] = pd.concat([pay, enrich_label(pay["label"], executor=executor)], axis=1)

    inv = tables["invoice"]
    tables["invoice"] = pd.concat(
        [
            inv,
            enrich_reference(inv["client_reference"], "client_reference", executor),
            enrich_reference(inv["internal_reference"], "internal_reference", executor),
        ],
        axis=1,
    )

    for name in ("assignor", "debtor"):
        party = tables[name]
        party["iban"] = normalize_iban(party["iban"])
        tables[name] = pd.concat([party, enrich_name(party["name"])], axis=1)

    if "technical_account" in tables:
        tables["technical_account"]["iban"] = normalize_iban(
            tables["technical_account"]["iban"]
        )

    if "client_file" in tables:
        cf = tables["client_file"]
        cf["iban"] = normalize_iban(cf["iban"])
        tables["client_file"] = pd.concat(
            [cf, enrich_label(cf["payment_reference"], "payment_reference", executor)], axis=1
        )

    if "client_file_line" in tables:
        lines = tables["client_file_line"]
        tables["client_file_line"] = pd.concat(
            [lines, enrich_reference(lines["invoice_reference"], "invoice_reference", executor)],
            axis=1,
        )


PARTY_ROLES = ("assignor", "debtor")


def consolidate_parties(tables: dict[str, pd.DataFrame], issues: list[Issue]) -> None:
    """Une ligne par partie ; tous les couples (IBAN, bankroll) dans la table `party_iban`.

    Les sources peuvent contenir plusieurs lignes pour une même partie (plusieurs comptes, plusieurs
    portefeuilles, historique). On garde : le premier nom non vide (toutes les variantes dans
    `name_variants`, pour l'index des noms), le premier bankroll_code non vide, la date d'ouverture la
    plus ancienne, et une fermeture seulement si toutes les lignes sont fermées (la plus récente).
    Le rapport de chargement indique quelles colonnes varient entre les lignes d'une même partie.
    """
    ibans = []
    for role in PARTY_ROLES:
        df = tables[role]
        missing_id = df["party_id"].isna()
        if missing_id.any():
            issues.append(Issue(role, "party_id", "identifiant_manquant", int(missing_id.sum()), "lignes ignorées"))
            df = df[~missing_id]
        dup = df["party_id"].duplicated(keep=False)
        if dup.any():
            d = df[dup]
            varying = []
            for col in ("iban", "bankroll_code", "name_norm", "opened_at", "closed_at"):
                n = int((d.groupby("party_id")[col].nunique(dropna=False) > 1).sum())
                if n:
                    varying.append(f"{col} ({n})")
            issues.append(Issue(role, "party_id", "lignes_multiples_par_partie", int(d["party_id"].nunique()),
                                f"{int(dup.sum())} lignes consolidées ; colonnes qui varient entre les lignes "
                                f"d'une même partie : {', '.join(varying) or 'aucune'}"))
        ibans.append(df.loc[df["iban"].notna(), ["party_id", "iban", "bankroll_code"]]
                     .drop_duplicates().assign(role=role))
        g = df.groupby("party_id", sort=False)
        out = g.first()
        out["opened_at"] = g["opened_at"].min()
        still_open = g["closed_at"].count() < g.size()
        out["closed_at"] = g["closed_at"].max().where(~still_open)
        variants = df[["party_id", "name_norm"]].dropna().drop_duplicates()
        out["name_variants"] = variants.groupby("party_id", sort=False)["name_norm"].agg(list).reindex(out.index)
        out["name_variants"] = [v if isinstance(v, list) else [] for v in out["name_variants"]]
        tables[role] = out.reset_index()[[*df.columns, "name_variants"]]
    tables["party_iban"] = pd.concat(ibans, ignore_index=True)[["role", "party_id", "iban", "bankroll_code"]]


def load_all(
    schema_cfg: dict[str, Any], base_dir: str | Path | None = None, workers: int = 1
) -> LoadedData:
    """Charge toutes les tables configurées et applique la normalisation.

    `base_dir` : répertoire des fichiers sources. Par défaut `schema_cfg["base_dir"]`.
    `workers`  : processus pour la normalisation des gros volumes (1 = séquentiel).
    Le résultat ne dépend pas de `workers`.
    """
    base = Path(base_dir or schema_cfg.get("base_dir") or ".")
    problems = _check_config(schema_cfg, base)
    if problems:
        raise SchemaConfigError(problems)

    tables_cfg = schema_cfg.get("tables") or {}
    tables: dict[str, pd.DataFrame] = {}
    mapped: dict[str, list[str]] = {}
    issues: list[Issue] = []
    for name, table in TABLES.items():
        tcfg = tables_cfg.get(name) or {}
        if not tcfg.get("source"):
            issues.append(Issue(name, None, "table_non_configuree", 0, "table optionnelle absente"))
            continue
        tables[name], mapped[name] = _load_table(table, tcfg, schema_cfg, base, issues)

    audit: dict[str, pd.DataFrame] = {}
    inv = tables["invoice"]
    if "current_amount" in mapped["invoice"]:
        audit["invoice_current_amount"] = inv[["invoice_id", "current_amount"]].copy()
    # Jamais de current_amount brut dans la pipeline (brief §4.1).
    tables["invoice"] = inv.drop(columns=["current_amount"])
    mapped["invoice"] = [f for f in mapped["invoice"] if f != "current_amount"]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            _enrich(tables, executor)
    else:
        _enrich(tables)
    consolidate_parties(tables, issues)
    return LoadedData(tables=tables, mapped_fields=mapped, issues=issues, audit=audit)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/load/events.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Journal d'événements ordonné (brief §3.4, spec §8.1).

Le journal est un DataFrame léger : chaque ligne référence une entité par son
identifiant, le détail reste dans les tables chargées. Colonnes :

- `seq`         : position dans le journal (0..n-1)
- `ts`          : horodatage UTC naïf
- `event_type`  : voir `EventType`
- `entity_id`   : payment_id, invoice_id, party_id, agreement_id ou file_id
- `related_id`  : invoice_id pour une imputation, rôle pour une partie, sinon ""
- `amount`      : montant imputé (centimes) pour IMPUTATION_APPLIED, sinon NA

Ordre : `ts`, puis rang du type d'événement, puis `entity_id`, puis `related_id`.
Le rang place, à horodatage égal, ce qui ouvre avant ce qui l'utilise, et ce qui
ferme en dernier.
"""






class EventType(str, Enum):
    PARTY_OPENED = "PARTY_OPENED"
    AGREEMENT_CREATED = "AGREEMENT_CREATED"
    INVOICE_CREATED = "INVOICE_CREATED"
    CLIENT_FILE_RECEIVED = "CLIENT_FILE_RECEIVED"
    PAYMENT_RECEIVED = "PAYMENT_RECEIVED"
    IMPUTATION_APPLIED = "IMPUTATION_APPLIED"
    AGREEMENT_DISABLED = "AGREEMENT_DISABLED"
    PARTY_CLOSED = "PARTY_CLOSED"


EVENT_RANK = {e.value: i for i, e in enumerate(EventType)}

JOURNAL_COLUMNS = ["seq", "ts", "event_type", "entity_id", "related_id", "amount"]


def derive_imputed_amounts(imputation: pd.DataFrame, invoice: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `balance_before` et `imputed_amount` à chaque ligne d'imputation.

    `residual_amount` est le solde de la facture après la ligne (brief §3.5) :
    montant imputé = solde avant − solde après, le solde avant étant le résidu
    de la ligne précédente sur la même facture, ou `initial_amount` pour la
    première. Chaque ligne n'utilise que les lignes qui la précèdent.

    Ordre de chaînage par facture : `updated_at`, puis résidu décroissant (le
    solde ne fait que baisser dans le cas nominal), puis `payment_id`.
    """
    imp = imputation.copy()
    imp["_neg_residual"] = -imp["residual_amount"].astype("Float64")
    imp = imp.sort_values(
        ["invoice_id", "updated_at", "_neg_residual", "payment_id"], kind="mergesort"
    )
    prev = imp.groupby("invoice_id", sort=False)["residual_amount"].shift(1)
    first = lookup(imp["invoice_id"], invoice["invoice_id"], invoice["initial_amount"]).astype("Int64")
    imp["balance_before"] = prev.astype("Int64").fillna(first)
    imp["imputed_amount"] = imp["balance_before"] - imp["residual_amount"]
    return imp.drop(columns="_neg_residual").sort_index()


def _events(ts: pd.Series, event_type: EventType, entity_id: pd.Series,
            related_id: pd.Series | str = "", amount: pd.Series | None = None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "ts": ts.astype(DATETIME_DTYPE),
            "event_type": event_type.value,
            "entity_id": entity_id.astype("string"),
            "related_id": related_id if isinstance(related_id, str)
            else related_id.astype("string").fillna(""),
            "amount": amount.astype("Int64") if amount is not None
            else pd.Series(pd.NA, index=ts.index, dtype="Int64"),
        }
    )
    return df[df["ts"].notna()]


def payment_event_time(payment: pd.DataFrame) -> pd.Series:
    """Date de connaissance du paiement : `booking_date` si connue, sinon `value_date`."""
    return payment["booking_date"].fillna(payment["value_date"])


def build_journal(data: LoadedData) -> tuple[pd.DataFrame, list[Issue]]:
    t = data.tables
    issues: list[Issue] = []
    parts: list[pd.DataFrame] = []

    for role in ("assignor", "debtor"):
        party = t[role]
        role_tag = role.upper()
        parts.append(_events(party["opened_at"], EventType.PARTY_OPENED, party["party_id"], role_tag))
        parts.append(_events(party["closed_at"], EventType.PARTY_CLOSED, party["party_id"], role_tag))
        n_no_open = int(party["opened_at"].isna().sum())
        if n_no_open:
            issues.append(Issue(role, "opened_at", "sans_evenement_ouverture", n_no_open,
                                "partie considérée active depuis toujours"))

    agr = t["agreement"]
    parts.append(_events(agr["created_at"], EventType.AGREEMENT_CREATED, agr["agreement_id"]))
    parts.append(_events(agr["disabled_at"], EventType.AGREEMENT_DISABLED, agr["agreement_id"]))

    inv = t["invoice"]
    parts.append(_events(inv["creation_date"], EventType.INVOICE_CREATED, inv["invoice_id"]))

    if "client_file" in t:
        cf = t["client_file"]
        parts.append(_events(cf["received_at"], EventType.CLIENT_FILE_RECEIVED, cf["file_id"]))

    pay = t["payment"]
    parts.append(_events(payment_event_time(pay), EventType.PAYMENT_RECEIVED, pay["payment_id"]))
    if "booking_date" in data.mapped_fields["payment"]:
        n_fallback = int(pay["booking_date"].isna().sum())
        if n_fallback:
            issues.append(Issue("payment", "booking_date", "repli_sur_value_date", n_fallback))

    imp = derive_imputed_amounts(t["imputation"], inv)
    parts.append(_events(imp["updated_at"], EventType.IMPUTATION_APPLIED, imp["payment_id"],
                         imp["invoice_id"], imp["imputed_amount"]))

    journal = pd.concat(parts, ignore_index=True)
    journal["_rank"] = journal["event_type"].map(EVENT_RANK)
    journal = journal.sort_values(
        ["ts", "_rank", "entity_id", "related_id"], kind="mergesort"
    ).drop(columns="_rank").reset_index(drop=True)
    journal.insert(0, "seq", range(len(journal)))
    return journal[JOURNAL_COLUMNS], issues


def journal_hash(journal: pd.DataFrame) -> str:
    """Empreinte SHA-256 du journal, stable entre exécutions et versions de pandas.

    Calculée colonne par colonne sur une représentation binaire explicite
    (horodatages en µs int64, chaînes UTF-8 séparées par le caractère US (0x1F), montants int64
    avec sentinelle pour NA), plutôt que sur une sérialisation CSV complète.
    """
    h = hashlib.sha256()
    h.update(f"{len(journal)}|{','.join(JOURNAL_COLUMNS)}".encode())
    h.update(journal["seq"].to_numpy(dtype="<i8").tobytes())
    h.update(journal["ts"].astype(DATETIME_DTYPE).to_numpy().view("<i8").tobytes())
    for col in ("event_type", "entity_id", "related_id"):
        h.update("\x1f".join(journal[col].astype("str").fillna("").tolist()).encode("utf-8"))
    h.update(journal["amount"].astype("Int64").fillna(-(2**63)).to_numpy(dtype="<i8").tobytes())
    return h.hexdigest()


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/load/quality.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Contrôles de qualité des données chargées (brief §3.5).

Rien n'est corrigé ni supprimé ici : chaque anomalie est comptée et remontée
dans le rapport de chargement pour décision.
"""





def _count(mask: pd.Series) -> int:
    return int(mask.fillna(False).sum())


def _orphans(child: pd.Series, parent: pd.Series) -> int:
    return int((~isin(child.dropna(), parent)).sum())


def check_quality(data: LoadedData, imputation_derived: pd.DataFrame | None = None) -> list[Issue]:
    """`imputation_derived` : sortie de `derive_imputed_amounts`, recalculée si absente."""
    t = data.tables
    issues: list[Issue] = []

    def add(table: str, fld: str | None, kind: str, n: int, detail: str = "") -> None:
        if n:
            issues.append(Issue(table, fld, kind, n, detail))

    # Unicité des clés primaires.
    for name, df in t.items():
        pk = list(TABLES[name].primary_key) if name in TABLES else []
        if pk:
            add(name, ",".join(pk), "cle_dupliquee", _count(df.duplicated(pk, keep=False)))

    # Intégrité référentielle.
    inv, pay, imp, agr = t["invoice"], t["payment"], t["imputation"], t["agreement"]
    add("invoice", "debtor_id", "orphelin", _orphans(inv["debtor_id"], t["debtor"]["party_id"]))
    add("invoice", "agreement_id", "orphelin", _orphans(inv["agreement_id"], agr["agreement_id"]))
    add("agreement", "debtor_id", "orphelin", _orphans(agr["debtor_id"], t["debtor"]["party_id"]))
    add("agreement", "client_id", "orphelin", _orphans(agr["client_id"], t["assignor"]["party_id"]))
    add("imputation", "payment_id", "orphelin", _orphans(imp["payment_id"], pay["payment_id"]))
    add("imputation", "invoice_id", "orphelin", _orphans(imp["invoice_id"], inv["invoice_id"]))
    if "client_file_line" in t and "client_file" in t:
        add("client_file_line", "file_id", "orphelin",
            _orphans(t["client_file_line"]["file_id"], t["client_file"]["file_id"]))

    # Cohérence de la facture.
    agr_debtor = lookup(inv["agreement_id"], agr["agreement_id"], agr["debtor_id"])
    add("invoice", "debtor_id", "debiteur_different_du_contrat",
        _count(agr_debtor.notna() & (inv["debtor_id"] != agr_debtor)))
    add("invoice", "due_date", "echeance_avant_creation", _count(inv["due_date"] < inv["creation_date"]))
    add("invoice", "initial_amount", "montant_non_positif", _count(inv["initial_amount"] <= 0))

    # Paiements hors périmètre v1.
    add("payment", "amount", "montant_negatif_hors_perimetre", _count(pay["amount"] < 0))
    add("payment", "amount", "montant_nul", _count(pay["amount"] == 0))
    main_ccy = pay["currency"].mode()
    if len(main_ccy):
        add("payment", "currency", "devise_minoritaire", _count(pay["currency"] != main_ccy.iloc[0]),
            f"devise principale {main_ccy.iloc[0]}")

    # Imputations.
    add("imputation", "status", "statut_inconnu",
        _count(~imp["status"].isin(IMPUTATION_STATUSES) & imp["status"].notna()),
        "renseigner value_maps.status dans config/schema.yaml")
    add("imputation", "residual_amount", "full_avec_residu_non_nul",
        _count((imp["status"] == "FULL") & (imp["residual_amount"] != 0)))
    add("imputation", "residual_amount", "partial_avec_residu_nul",
        _count((imp["status"] == "PARTIAL") & (imp["residual_amount"] == 0)))
    add("imputation", "payment_id,invoice_id", "lignes_multiples_meme_paire",
        _count(imp.duplicated(["payment_id", "invoice_id"], keep=False)))

    derived = imputation_derived if imputation_derived is not None else derive_imputed_amounts(imp, inv)
    add("imputation", "imputed_amount", "montant_impute_negatif", _count(derived["imputed_amount"] < 0),
        "le solde de la facture remonte : vérifier l'interprétation de residual_amount")
    add("imputation", "imputed_amount", "montant_impute_nul", _count(derived["imputed_amount"] == 0))

    imp_value = lookup(imp["payment_id"], pay["payment_id"], pay["value_date"])
    imp_known = lookup(imp["payment_id"], pay["payment_id"], payment_event_time(pay))
    add("imputation", "updated_at", "avant_date_de_valeur_du_paiement",
        _count(imp["updated_at"] < imp_value))
    add("imputation", "updated_at", "avant_connaissance_du_paiement",
        _count(imp["updated_at"] < imp_known),
        "imputation antérieure à l'événement PAYMENT_RECEIVED")
    inv_created = lookup(imp["invoice_id"], inv["invoice_id"], inv["creation_date"])
    add("imputation", "updated_at", "avant_creation_facture", _count(imp["updated_at"] < inv_created))

    # Audit : le solde final reconstruit doit égaler invoice.current_amount.
    if "invoice_current_amount" in data.audit:
        imputed = derived.groupby("invoice_id", sort=False)["imputed_amount"].sum()
        imputed_total = lookup(inv["invoice_id"], pd.Series(imputed.index), imputed.reset_index(drop=True))
        rebuilt = inv["initial_amount"] - imputed_total.fillna(0).astype("Int64")
        audit_df = data.audit["invoice_current_amount"]
        final = lookup(inv["invoice_id"], audit_df["invoice_id"], audit_df["current_amount"])
        add("invoice", "current_amount", "solde_reconstruit_different",
            _count(final.notna() & (rebuilt != final)),
            "initial_amount − Σ imputations ≠ current_amount final")

    # Parties.
    for role in ("assignor", "debtor"):
        party = t[role]
        add(role, "closed_at", "fermeture_avant_ouverture", _count(party["closed_at"] < party["opened_at"]))
    add("agreement", "disabled_at", "desactivation_avant_creation",
        _count(agr["disabled_at"] < agr["created_at"]))

    return issues


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/load/profile.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Profil des données chargées : volumes, plages de dates, champs manquants,
anomalies et résumé du journal (critère « fini quand » de l'étape 1)."""






def build_profile(data: LoadedData, journal: pd.DataFrame, issues: list[Issue]) -> dict[str, pd.DataFrame]:
    volumes, missing, dates = [], [], []
    for name, df in data.tables.items():
        volumes.append({"table": name, "rows": len(df)})
        if name not in TABLES:
            continue
        mapped = set(data.mapped_fields[name])
        for f in TABLES[name].fields:
            if f.name not in df.columns:
                continue
            n_null = int(df[f.name].isna().sum())
            missing.append({
                "table": name,
                "field": f.name,
                "required": f.required,
                "mapped": f.name in mapped,
                "rows": len(df),
                "null": n_null,
                "null_pct": round(100 * n_null / len(df), 2) if len(df) else 0.0,
            })
            if f.type in (FieldType.DATE, FieldType.TIMESTAMP) and f.name in mapped:
                col = df[f.name]
                dates.append({"table": name, "field": f.name, "min": col.min(), "max": col.max()})

    events = (
        journal.groupby("event_type", sort=False)
        .agg(count=("seq", "size"), first_ts=("ts", "min"), last_ts=("ts", "max"))
        .reset_index()
    )
    issue_df = pd.DataFrame([asdict(i) for i in issues],
                            columns=["table", "field", "kind", "count", "detail"])
    return {
        "volumes": pd.DataFrame(volumes),
        "missing_fields": pd.DataFrame(missing),
        "date_ranges": pd.DataFrame(dates),
        "issues": issue_df,
        "events": events,
    }


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(vide)_\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = ["" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_profile_markdown(profile: dict[str, pd.DataFrame], meta: dict[str, Any]) -> str:
    missing = profile["missing_fields"]
    notable = missing[(~missing["mapped"]) | (missing["null"] > 0)]
    parts = [
        "# Profil de chargement — étape 1\n",
        "\n".join(f"- **{k}** : `{v}`" for k, v in meta.items()) + "\n",
        "## Volumes\n", _md_table(profile["volumes"]),
        "## Plages de dates\n", _md_table(profile["date_ranges"]),
        "## Champs non mappés ou incomplets\n", _md_table(notable),
        "## Anomalies\n", _md_table(profile["issues"]),
        "## Journal d'événements\n", _md_table(profile["events"]),
    ]
    return "\n".join(parts)


def write_reports(profile: dict[str, pd.DataFrame], meta: dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name, df in profile.items():
        df.to_csv(reports_dir / f"load_{name}.csv", index=False)
    md_path = reports_dir / "load_profile.md"
    md_path.write_text(render_profile_markdown(profile, meta), encoding="utf-8")
    return md_path


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/load/interim.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Relecture des sorties de l'étape 1 (tables normalisées + journal) pour les étapes suivantes."""






class InterimError(RuntimeError):
    pass


def load_interim(directory: str | Path, verify: bool = True) -> tuple[LoadedData, pd.DataFrame, dict]:
    """Tables, journal et métadonnées de l'étape 1. Vérifie l'empreinte du journal si `verify`."""
    d = Path(directory)
    meta_path = d / "journal_meta.json"
    if not meta_path.exists():
        raise InterimError(f"aucune sortie de l'étape 1 dans {d} : lancer d'abord le chargement")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    names = [*TABLES, "party_iban"]
    tables = {name: pd.read_parquet(d / f"{name}.parquet") for name in names if (d / f"{name}.parquet").exists()}
    journal = pd.read_parquet(d / "journal.parquet")
    if verify and journal_hash(journal) != meta["journal_sha256"]:
        raise InterimError("le journal ne correspond pas à son empreinte : relancer l'étape 1")
    mapped = meta.get("mapped_fields") or {
        name: [f.name for f in TABLES[name].fields if f.name in df.columns and df[f.name].notna().any()]
        for name, df in tables.items() if name in TABLES
    }
    return LoadedData(tables=tables, mapped_fields=mapped), journal, meta


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/load/schema_io.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Lecture / écriture commentée de `config/schema.yaml` (mapping vers les noms réels).

Le fichier est régénéré à partir du modèle canonique : les commentaires
(champs requis, rôle de chaque clé) survivent à une sauvegarde depuis l'UI.
"""




GLOBAL_KEYS: dict[str, str] = {
    "amount_unit": "Unité des montants dans les sources : \"cents\" (entiers de centimes) ou \"units\" (ex. 1234.56).",
    "decimal_separator": "Séparateur décimal des montants lus en texte (\".\" ou \",\").",
    "source_timezone": "Fuseau des horodatages sans fuseau explicite (ex. \"Europe/Paris\"). null → UTC.",
    "date_format": "Format strptime des dates (null → inférence pandas).",
    "timestamp_format": "Format strptime des horodatages (null → inférence pandas).",
    "base_dir": "Répertoire des fichiers sources (relatif à la racine du dépôt ou absolu).",
}

TABLE_NOTES: dict[str, str] = {
    "technical_account": "Optionnel — référentiel des IBAN de comptes techniques.",
    "client_file": "Optionnel — client files. FORMAT RÉEL À CONFIRMER (brief §3.2). Voie tabulaire uniquement.",
    "client_file_line": "Optionnel — lignes des client files.",
}

FIELD_NOTES: dict[tuple[str, str], str] = {
    ("payment", "booking_date"): "date de comptabilisation si elle existe (§3.5) — ordonne le journal",
    ("payment", "amount"): "signé",
    ("payment", "bankroll_code"): "souvent absent (§3.5)",
    ("invoice", "current_amount"): "audit uniquement, jamais utilisé par la pipeline",
    ("imputation", "status"): "FULL / PARTIAL après value_maps",
    ("imputation", "residual_amount"): "solde de la facture APRÈS la ligne",
    ("debtor", "closed_at"): "peut ne pas exister (§3.5)",
    ("agreement", "client_id"): "→ assignor.party_id",
    ("client_file", "received_at"): "pivot temporel",
}

VALUE_MAP_FIELDS: dict[str, list[str]] = {"imputation": ["status"]}

_HEADER = """\
# Mapping modèle canonique → noms réels des tables et colonnes.
# À REMPLIR PAR L'ÉQUIPE. Aucune valeur ne doit être devinée.
# Éditable à la main ou depuis l'UI (page Configuration).
#
# - Chaque `null` sous `columns` est une colonne réelle à renseigner.
#   Les champs marqués (requis) bloquent le chargement tant qu'ils sont vides.
#   Les autres peuvent rester à null : ils seront signalés dans le rapport.
# - `source` : fichier (csv ou parquet) relatif à `base_dir`.
# - `read_options` : options passées telles quelles à pandas.read_csv / read_parquet
#   (ex. sep: ";", encoding: "latin-1").
# - `value_maps` : traduction des valeurs réelles vers les valeurs canoniques,
#   ex. status: {FULL: [TOTAL, SOLDE], PARTIAL: PARTIEL}.
"""


def empty_schema() -> dict[str, Any]:
    cfg: dict[str, Any] = {k: None for k in GLOBAL_KEYS}
    cfg["tables"] = {}
    for name, table in TABLES.items():
        tcfg: dict[str, Any] = {"source": None, "read_options": {},
                                "columns": {f.name: None for f in table.fields}}
        if name in VALUE_MAP_FIELDS:
            tcfg["value_maps"] = {f: {} for f in VALUE_MAP_FIELDS[name]}
        cfg["tables"][name] = tcfg
    return cfg


def normalize_schema(cfg: dict[str, Any]) -> dict[str, Any]:
    """Complète un schéma partiel avec toutes les clés attendues (valeurs vides)."""
    out = empty_schema()
    for k in GLOBAL_KEYS:
        if k in cfg:
            out[k] = cfg[k]
    for name, tcfg in (cfg.get("tables") or {}).items():
        if name not in out["tables"] or not tcfg:
            continue
        target = out["tables"][name]
        target["source"] = tcfg.get("source")
        target["read_options"] = dict(tcfg.get("read_options") or {})
        for fld, col in (tcfg.get("columns") or {}).items():
            target["columns"][fld] = col
        for fld, mapping in (tcfg.get("value_maps") or {}).items():
            target.setdefault("value_maps", {})[fld] = copy.deepcopy(mapping) or {}
        for opt in ("date_format", "timestamp_format"):
            if tcfg.get(opt):
                target[opt] = tcfg[opt]
    return out


def dump_schema(cfg: dict[str, Any]) -> str:
    cfg = normalize_schema(cfg)
    lines = [_HEADER]
    for key, note in GLOBAL_KEYS.items():
        lines += [f"# {note}", f"{key}: {scalar(cfg[key])}"]
    lines += ["", "tables:"]
    for name, table in TABLES.items():
        tcfg = cfg["tables"][name]
        if name in TABLE_NOTES:
            lines.append(f"  # {TABLE_NOTES[name]}")
        lines += [f"  {name}:", f"    source: {scalar(tcfg['source'])}",
                  f"    read_options: {scalar(tcfg['read_options'])}"]
        for opt in ("date_format", "timestamp_format"):
            if tcfg.get(opt):
                lines.append(f"    {opt}: {scalar(tcfg[opt])}")
        lines.append("    columns:")
        for f in table.fields:
            notes = [n for n in ("(requis)" if f.required else None, FIELD_NOTES.get((name, f.name))) if n]
            comment = f"  # {' '.join(notes)}" if notes else ""
            lines.append(f"      {f.name}: {scalar(tcfg['columns'].get(f.name))}{comment}")
        if "value_maps" in tcfg:
            lines.append("    value_maps:")
            for fld, mapping in tcfg["value_maps"].items():
                lines.append(f"      {fld}: {scalar(mapping or {})}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    p = Path(path)
    return normalize_schema(read_yaml(p) if p.exists() else {})


def save_schema(cfg: dict[str, Any], path: str | Path = DEFAULT_SCHEMA_PATH) -> None:
    Path(path).write_text(dump_schema(cfg), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/synthetic/generate.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Jeu de données synthétique à seed fixe, au format « source » (avant mapping).

Sert aux tests et à la démonstration de la pipeline, jamais à mesurer une
performance réelle. Génération vectorisée (numpy / pandas) pour atteindre le
volume réel : ~2 millions de paiements sur un an.

Situations métier reproduites :
- débiteurs de tailles très inégales (distribution à queue lourde) ;
- références numérotées par cédant, donc réutilisées d'un cédant à l'autre ;
- références citées de façon bruitée (sans préfixe, sans zéros, inversées) ;
- paiements groupés 1↔n, partiels n↔1, factures impayées ;
- flux via comptes techniques, IBAN inconnus ;
- paiements orphelins sans facture (remboursements…), jamais imputés ; doublons ;
- écarts de montant : escompte, frais SWIFT, retenue de garantie BTP ;
- groupes n↔n : lot de factures payé en deux virements de montants arbitraires ;
- références erronées : faute de frappe, référence d'une autre facture ;
- client files pour une partie des paiements groupés.

Les montants sont écrits en euros (texte « 1234.56 »), les horodatages en
heure de Paris sans fuseau, les statuts d'imputation en TOTAL / PARTIEL — pour
exercer les conversions du loader.
"""




_WORDS = np.array([
    "ALPHA", "Béton", "NORD", "Sud", "TRANS", "Logistique", "ÉLEC", "Métal", "AGRI", "Bâti",
    "PLAST", "Conseil", "Services", "Distrib", "Atlantique", "Rhône", "Alpes", "Ouest", "Génie",
    "Froid", "Hôtellerie", "Imprimerie", "Menuiserie", "Câblage", "Industrie", "Transports",
    "Boulangerie", "Chimie", "Énergie", "Maçonnerie", "Peinture", "Verrerie", "Textile",
    "Mécanique", "Emballage", "Papeterie", "Nettoyage", "Sécurité", "Informatique", "Médical",
])
_CITIES = np.array([
    "", "", "", "PARIS", "LYON", "Marseille", "LILLE", "Nantes", "Bordeaux", "Toulouse", "RENNES",
    "Strasbourg", "Nice", "Grenoble", "Dijon", "Angers", "Reims", "Le Havre", "Saint-Étienne",
])
_FORMS = np.array(["SARL", "SAS", "SA", "EURL", "S.A.S."])
_MARKETS = np.array(["BTP", "INDUSTRIE", "SERVICES", "DISTRIBUTION"])
_HEADS = np.array(["VIR SEPA", "VIREMENT", "VIR RECU", "PRLV", ""])
_CITE_TEMPLATES = np.array(["FACT ", "REGLT ", "", "Facture n° "])
_NO_CITE_BODIES = np.array(["REGLEMENT FACTURES", "PAIEMENT", "", "ECHEANCE"])
_N_REF_STYLES = 5
_FACTOR_IBAN = "FR76 3000 4000 0500 0000 0000 001"


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    start: date = date(2024, 1, 1)
    n_days: int = 365
    # Volume cible de paiements (approximatif : ± quelques %).
    n_payments: int = 2_000_000
    debtors_per_payment: float = 0.025       # 2 M paiements → 50 000 débiteurs
    assignors_per_payment: float = 0.001     # 2 M paiements → 2 000 cédants
    n_technical_accounts: int = 20
    # Factures générées par paiement visé (compense groupés, impayés, partiels).
    invoices_per_payment: float = 1.47
    # Part de paiements orphelins, sans facture correspondante, jamais imputés.
    orphan_share: float = 0.03
    # Part de paiements payés deux fois (le doublon n'est jamais imputé).
    duplicate_share: float = 0.003


# --- Utilitaires vectorisés ---------------------------------------------------


def _ibans(rng: np.random.Generator, n: int) -> np.ndarray:
    digits = (rng.integers(0, 10, size=(n, 23), dtype=np.uint8) + ord("0")).view("S23").ravel()
    s = pd.Series(digits.astype(str))
    return ("FR76 " + s.str[0:4] + " " + s.str[4:8] + " " + s.str[8:12] + " " + s.str[12:16]
            + " " + s.str[16:20] + " " + s.str[20:23]).to_numpy()


_SYLLABLES_1 = np.array(["Ber", "Mor", "Dal", "Lef", "Gau", "Ro", "Mar", "Char", "Duv", "Fon", "Gir", "Lam",
                         "Pel", "Ren", "Tes", "Vau", "Bou", "Cha", "Del", "Fer", "Gui", "Mal", "Pic", "Sau",
                         "Tho", "Val", "Bris", "Clé", "Gré", "Lou", "Mé", "Pou", "Ri", "Ség", "Tou", "Ver"])
_SYLLABLES_2 = np.array(["nard", "van", "lac", "èvre", "tier", "ssel", "chand", "pentier", "al", "taine", "ard",
                         "bert", "letier", "aud", "sier", "tron", "chet", "ron", "mas", "rand", "llot", "zon",
                         "card", "nier", "mier", "ry", "gnon", "quet", "lin", "vet", "tel", "reau", "din", "mont"])


def _surnames(rng: np.random.Generator, n: int) -> pd.Series:
    """Patronymes synthétiques (≈ 30 000 combinaisons) : un nom commence souvent par un mot distinctif."""
    third = np.where(rng.random(n) < 0.3, rng.choice(["et", "in", "ier", "ot", "eau", "ac"], n), "")
    return pd.Series(rng.choice(_SYLLABLES_1, n)) + pd.Series(rng.choice(_SYLLABLES_2, n)) + pd.Series(third)


def _companies(rng: np.random.Generator, n: int) -> np.ndarray:
    """Raisons sociales : patronyme + activité (60 %), activité + patronyme (25 %), deux activités (15 %)."""
    surname = _surnames(rng, n)
    a, b = pd.Series(rng.choice(_WORDS, n)), pd.Series(rng.choice(_WORDS, n))
    u = rng.random(n)
    head = np.select([u < 0.60, u < 0.85], [surname + " " + a, a + " " + surname], default=a + " " + b)
    name = pd.Series(head) + " " + pd.Series(rng.choice(_CITIES, n)) + " " + pd.Series(rng.choice(_FORMS, n))
    return name.str.replace(r"\s+", " ", regex=True).to_numpy()


def _typo(ref: str, u: float) -> str:
    """Remplace un chiffre de la référence citée (déterministe pour un tirage `u` donné)."""
    digits = [i for i, c in enumerate(ref) if c.isdigit()]
    if not digits:
        return ref
    i = digits[int(u * len(digits)) % len(digits)]
    return ref[:i] + str((int(ref[i]) + 1 + int(u * 1000) % 9) % 10) + ref[i + 1:]


def _ids(prefix: str, n: int, width: int) -> np.ndarray:
    return (prefix + pd.Series(np.arange(n)).astype(str).str.zfill(width)).to_numpy()


def _day_offsets(start: np.datetime64, offsets: np.ndarray) -> np.ndarray:
    return start + offsets.astype("timedelta64[D]")


# --- Génération ---------------------------------------------------------------


def generate(cfg: SyntheticConfig = SyntheticConfig()) -> dict[str, pd.DataFrame]:
    """Retourne une DataFrame par table, montants en centimes, dates en datetime64."""
    rng = np.random.default_rng(cfg.seed)
    start = np.datetime64(cfg.start, "D")
    end = start + np.timedelta64(cfg.n_days - 1, "D")

    # Cédants.
    n_a = max(2, round(cfg.n_payments * cfg.assignors_per_payment))
    assignor = pd.DataFrame({
        "party_id": _ids("A", n_a, 5),
        "bankroll_code": rng.choice(["BR_STD", "BR_SP"], n_a),
        "iban": _ibans(rng, n_a),
        "name": _companies(rng, n_a),
        "opened_at": _day_offsets(start, -rng.integers(30, 900, n_a)),
        "closed_at": np.full(n_a, np.datetime64("NaT"), dtype="datetime64[D]"),
    })
    a_weight = rng.pareto(1.5, n_a) + 1
    a_style = np.arange(n_a) % _N_REF_STYLES

    # Débiteurs et comportements.
    n_d = max(5, round(cfg.n_payments * cfg.debtors_per_payment))
    d_iban = _ibans(rng, n_d)
    d_iban[rng.random(n_d) >= 0.9] = ""
    d_opened = _day_offsets(start, -rng.integers(-120, 900, n_d))
    debtor = pd.DataFrame({
        "party_id": _ids("D", n_d, 6),
        "bankroll_code": rng.choice(["BR_STD", "BR_SP"], n_d),
        "iban": d_iban,
        "name": _companies(rng, n_d),
        "opened_at": d_opened,
    })
    d_weight = np.minimum(rng.pareto(1.2, n_d) + 1, 2000)
    d_cite_rate = rng.choice([0.05, 0.5, 0.9, 0.95], n_d)
    d_delay = rng.normal(5, 15, n_d)
    d_break = rng.choice([1.0, 0.85, 0.5], n_d)          # 1.0 = ne groupe jamais

    technical = pd.DataFrame({
        "iban": _ibans(rng, cfg.n_technical_accounts),
        "bankroll_code": "BR_SP",
        "description": [f"Compte de liaison {i}" for i in range(cfg.n_technical_accounts)],
    })

    # Contrats : 1 à 3 par débiteur.
    per_debtor = 1 + (rng.random(n_d) < 0.3) + (rng.random(n_d) < 0.05)
    ag_debtor = np.repeat(np.arange(n_d), per_debtor)
    n_ag = len(ag_debtor)
    ag_assignor = rng.choice(n_a, n_ag, p=a_weight / a_weight.sum())
    ag_created = np.maximum(d_opened[ag_debtor],
                            _day_offsets(start, rng.integers(-400, 90, n_ag)))
    ag_disabled = np.full(n_ag, np.datetime64("NaT"), dtype="datetime64[D]")
    dis = rng.random(n_ag) < 0.05
    room = np.maximum((end - ag_created).astype(int) - 120, 1)
    ag_disabled[dis] = ag_created[dis] + (120 + rng.integers(0, room[dis])).astype("timedelta64[D]")
    ag_disabled[ag_disabled > end] = np.datetime64("NaT")
    ag_market = rng.choice(_MARKETS, n_ag)
    agreement = pd.DataFrame({
        "agreement_id": _ids("AG", n_ag, 6),
        "debtor_id": debtor["party_id"].to_numpy()[ag_debtor],
        "client_id": assignor["party_id"].to_numpy()[ag_assignor],
        "contract_number": "CTR-" + pd.Series(rng.integers(100000, 999999, n_ag)).astype(str),
        "created_at": ag_created,
        "disabled_at": ag_disabled,
        "market": ag_market,
        "product": rng.choice(["CLASSIQUE", "CONFIDENTIEL"], n_ag),
        "recourse": rng.choice(["AVEC", "SANS"], n_ag),
    })

    # Factures, réparties selon le poids du débiteur.
    n_inv = round(cfg.n_payments * cfg.invoices_per_payment)
    ag_w = d_weight[ag_debtor] * rng.uniform(0.5, 1.5, n_ag)
    inv_ag = rng.choice(n_ag, n_inv, p=ag_w / ag_w.sum())
    lo = np.maximum(ag_created[inv_ag], start - np.timedelta64(60, "D"))
    hi = np.where(np.isnat(ag_disabled[inv_ag]), end, ag_disabled[inv_ag])
    span = np.maximum((hi - lo).astype(int), 0) + 1
    inv_created = lo + np.floor(rng.random(n_inv) * span).astype("timedelta64[D]")
    order = np.lexsort((inv_ag, inv_created))
    inv_ag, inv_created = inv_ag[order], inv_created[order]
    inv_debtor = ag_debtor[inv_ag]
    inv_assignor = ag_assignor[inv_ag]
    inv_amount = np.maximum(np.round(rng.lognormal(7.5, 1.0, n_inv) * 100), 100).astype(np.int64)
    inv_due = inv_created + rng.choice([30, 45, 60], n_inv).astype("timedelta64[D]")

    # Références : numérotation propre à chaque cédant, style propre à chaque cédant.
    seq = pd.Series(inv_assignor).groupby(inv_assignor).cumcount().to_numpy()
    num = pd.Series(seq + rng.integers(1, 50_000, n_a)[inv_assignor]).astype(str)
    year = pd.Series(inv_created.astype("datetime64[Y]").astype(int) + 1970).astype(str)
    style = a_style[inv_assignor]
    styles = [
        "FA" + num.str.zfill(7),
        "F-" + year + "-" + num.str.zfill(5),
        num.str.zfill(6),
        "INV" + num,
        "FACT" + num.str.zfill(5),
    ]
    inv_ref = np.select([style == k for k in range(_N_REF_STYLES)], [s.to_numpy() for s in styles])
    inv_ids = _ids("I", n_inv, 8)

    # --- Paiements -----------------------------------------------------------
    # Factures payées, triées par débiteur puis échéance, regroupées en lots de 1 à 4.
    paid = np.flatnonzero(rng.random(n_inv) >= 0.08)
    paid = paid[np.lexsort((paid, inv_due[paid], inv_debtor[paid]))]
    deb = inv_debtor[paid]
    new_debtor = np.r_[True, deb[1:] != deb[:-1]]
    brk = new_debtor | (rng.random(len(paid)) < d_break[deb])
    gid = np.cumsum(brk) - 1
    pos = np.arange(len(paid)) - np.flatnonzero(brk)[gid]
    gid = np.cumsum(brk | (pos % 4 == 0)) - 1

    g = pd.DataFrame({"gid": gid, "inv": paid, "due": inv_due[paid], "created": inv_created[paid],
                      "amount": inv_amount[paid], "debtor": deb})
    grp = g.groupby("gid", sort=True).agg(
        due=("due", "max"), created=("created", "max"), amount=("amount", "sum"),
        size=("inv", "size"), debtor=("debtor", "first"))
    n_g = len(grp)
    g_debtor = grp["debtor"].to_numpy()
    delay = np.round(rng.normal(d_delay[g_debtor], 8)).astype(int)
    pay_day = grp["due"].to_numpy().astype("datetime64[D]") + delay.astype("timedelta64[D]")
    pay_day = np.maximum(pay_day, grp["created"].to_numpy().astype("datetime64[D]") + np.timedelta64(1, "D"))
    in_period = pay_day <= end
    size = grp["size"].to_numpy()
    g_amount = grp["amount"].to_numpy()
    g_market = ag_market[inv_ag[g.drop_duplicates("gid")["inv"].to_numpy()]]
    # Scénarios par lot : partiel en deux fois (n↔1), retenue de garantie BTP (5 % payés des mois
    # plus tard, ou jamais), n↔n (lot de 3-4 factures payé en deux virements de montants arbitraires).
    partial = (size == 1) & (rng.random(n_g) < 0.08)
    retention = (size == 1) & ~partial & (g_market == "BTP") & (rng.random(n_g) < 0.15)
    nn = (size >= 3) & (rng.random(n_g) < 0.15)

    first_amt = np.select(
        [partial, retention, nn],
        [g_amount * rng.choice([30, 50, 70], n_g) // 100, g_amount * 95 // 100,
         np.round(g_amount * rng.uniform(0.4, 0.6, n_g)).astype(np.int64)],
        default=g_amount)
    second_day = pay_day + np.select(
        [retention, nn], [rng.integers(180, 366, n_g), rng.integers(1, 6, n_g)],
        default=rng.integers(10, 41, n_g)).astype("timedelta64[D]")
    has_second = (partial | nn | (retention & (rng.random(n_g) < 0.6))) & in_period & (second_day <= end)

    p1_g = np.flatnonzero(in_period)
    p2_g = np.flatnonzero(has_second)
    pay = pd.DataFrame({
        "gid": np.r_[p1_g, p2_g],
        "value_date": np.r_[pay_day[p1_g], second_day[p2_g]],
        "amount": np.r_[first_amt[p1_g], (g_amount - first_amt)[p2_g]],
        "second": np.r_[np.zeros(len(p1_g), bool), np.ones(len(p2_g), bool)],
    })
    pay = pay.sort_values(["value_date", "gid", "second"], kind="mergesort").reset_index(drop=True)
    n_p = len(pay)
    pay["payment_id"] = _ids("P", n_p, 8)
    pay["prow"] = np.arange(n_p)
    p_debtor = g_debtor[pay["gid"].to_numpy()]
    pay["booking_date"] = pay["value_date"].to_numpy() + rng.choice([0, 0, 1, 2], n_p).astype("timedelta64[D]")

    route = rng.random(n_p)
    iban = np.where(route < 0.90, technical["iban"].to_numpy()[rng.integers(0, len(technical), n_p)], "")
    own = (route < 0.75) & (d_iban[p_debtor] != "")
    iban = np.where(own, d_iban[p_debtor], iban)
    unknown = route >= 0.90
    iban[unknown] = _ibans(rng, int(unknown.sum()))

    # Allocation paiement → factures (vérité terrain des imputations).
    alloc = g.merge(pay[["gid", "payment_id", "prow", "amount", "second", "booking_date"]], on="gid",
                    suffixes=("_inv", ""))
    alloc_gid = alloc["gid"].to_numpy()
    alloc["imputed"] = np.where((partial | retention)[alloc_gid], alloc["amount"], alloc["amount_inv"])
    # n↔n : le premier virement solde les factures dans l'ordre des échéances jusqu'à épuisement,
    # le second continue ; une facture peut être partagée entre les deux.
    is_nn = nn[alloc_gid]
    if is_nn.any():
        part = alloc[is_nn].sort_values(["gid", "second", "due", "inv"], kind="mergesort")
        c_hi = part.groupby(["gid", "second"])["amount_inv"].cumsum().to_numpy()
        c_lo = c_hi - part["amount_inv"].to_numpy()
        x1 = first_amt[part["gid"].to_numpy()]
        first_share = np.clip(np.minimum(c_hi, x1) - c_lo, 0, None)
        second_share = np.clip(c_hi - np.maximum(c_lo, x1), 0, None)
        alloc.loc[part.index, "imputed"] = np.where(part["second"].to_numpy(), second_share, first_share)
    alloc = alloc[alloc["imputed"] > 0]
    alloc = alloc.sort_values(["payment_id", "due", "inv"], kind="mergesort").reset_index(drop=True)

    # Libellés : références citées de façon bruitée, au plus 3 par paiement.
    ref = pd.Series(inv_ref[alloc["inv"].to_numpy()])
    digits = ref.str.replace(r"\D", "", regex=True)
    prefix = ref.str.replace(r"[^A-Za-z]", "", regex=True)
    u = rng.random(len(alloc))
    cited = np.select(
        [u < 0.45, u < 0.60, (u < 0.70) & (prefix != ""), u < 0.80],
        [ref, digits.str.lstrip("0").where(lambda s: s != "", digits),
         digits + " " + prefix, ref.str.lower().str.replace("-", " ")],
        default=digits,
    )
    # Bruit : faute de frappe sur un chiffre (2 %), référence d'une autre facture (1,5 %).
    v = rng.random(len(alloc))
    typo = np.flatnonzero(v < 0.02)
    cited[typo] = [_typo(c, r) for c, r in zip(cited[typo], rng.random(len(typo)))]
    wrong = np.flatnonzero((v >= 0.02) & (v < 0.035))
    cited[wrong] = inv_ref[rng.integers(0, n_inv, len(wrong))]
    rank = alloc.groupby("payment_id", sort=False).cumcount().to_numpy()
    prow = alloc["prow"].to_numpy()
    refs = pd.Series("", index=range(n_p), dtype=object)
    for k in range(3):                     # au plus 3 références citées par libellé
        slot = np.full(n_p, "", dtype=object)
        slot[prow[rank == k]] = cited[rank == k]
        refs = refs + " " + slot

    names = debtor["name"].to_numpy()[p_debtor]
    short = pd.Series(names).str.replace(r" .*", "", regex=True).to_numpy()
    name = np.where(rng.random(n_p) < 0.7, names, short)
    cites = rng.random(n_p) < d_cite_rate[p_debtor]
    body = np.where(cites, rng.choice(_CITE_TEMPLATES, n_p) + refs.to_numpy(),
                    rng.choice(_NO_CITE_BODIES, n_p))
    label = pd.Series(rng.choice(_HEADS, n_p) + " " + name + " " + body)
    label = label.str.replace(r"\s+", " ", regex=True).str.strip()

    # Écarts de montant sur les paiements soldant leurs factures : escompte (0,5-3 %) ou frais
    # SWIFT (5-40 €). L'imputation solde quand même la facture (écart passé en perte).
    channel = rng.choice(["SEPA", "SEPA", "SEPA", "SWIFT", "LCR"], n_p)
    p_gid = pay["gid"].to_numpy()
    settles = ~(partial | retention | nn)[p_gid]
    amount = pay["amount"].to_numpy().copy()
    discount = settles & (rng.random(n_p) < 0.04)
    amount[discount] -= np.round(amount[discount] * rng.uniform(0.005, 0.03, int(discount.sum()))).astype(np.int64)
    fee = settles & ~discount & (channel == "SWIFT") & (rng.random(n_p) < 0.3)
    amount[fee] -= rng.integers(500, 4001, int(fee.sum()))
    amount = np.maximum(amount, 100)
    pay["amount"] = amount

    payment = pd.DataFrame({
        "payment_id": pay["payment_id"],
        "value_date": pay["value_date"],
        "booking_date": pay["booking_date"],
        "amount": pay["amount"],
        "currency": "EUR",
        "iban_debtor": iban,
        "iban_creditor": _FACTOR_IBAN,
        "label": label.to_numpy(),
        "channel": channel,
        "payment_type": "VIREMENT",
    })

    # Paiements orphelins (3 %) : aucun lien avec une facture (remboursements, flux hors périmètre).
    # Ils ne sont jamais imputés ; tous les paiements liés à des factures le sont.
    n_o = round(n_p * cfg.orphan_share)
    o_day = _day_offsets(start, rng.integers(0, cfg.n_days, n_o))
    o_iban = np.where(rng.random(n_o) < 0.5, technical["iban"].to_numpy()[rng.integers(0, len(technical), n_o)], "")
    o_unknown = o_iban == ""
    o_iban[o_unknown] = _ibans(rng, int(o_unknown.sum()))
    o_label = pd.Series(rng.choice(_HEADS, n_o) + " " + _companies(rng, n_o) + " "
                        + rng.choice(["REMBOURSEMENT", "AVOIR", "REGUL", "VIREMENT", ""], n_o))
    orphans = pd.DataFrame({
        "payment_id": ("P" + pd.Series(np.arange(n_p, n_p + n_o)).astype(str).str.zfill(8)).to_numpy(),
        "value_date": o_day,
        "booking_date": o_day + rng.choice([0, 0, 1, 2], n_o).astype("timedelta64[D]"),
        "amount": np.maximum(np.round(rng.lognormal(7.0, 1.2, n_o) * 100), 100).astype(np.int64),
        "currency": "EUR", "iban_debtor": o_iban, "iban_creditor": _FACTOR_IBAN,
        "label": o_label.str.replace(r"\s+", " ", regex=True).str.strip().to_numpy(),
        "channel": rng.choice(["SEPA", "SEPA", "SWIFT"], n_o), "payment_type": "VIREMENT",
    })
    # Doublons (0,3 %) : le débiteur paie deux fois ; le second virement n'est jamais imputé.
    dup = payment.iloc[np.flatnonzero(rng.random(n_p) < cfg.duplicate_share)].copy()
    shift = rng.integers(1, 4, len(dup)).astype("timedelta64[D]")
    dup["value_date"] = dup["value_date"].to_numpy().astype("datetime64[D]") + shift
    dup["booking_date"] = dup["booking_date"].to_numpy().astype("datetime64[D]") + shift
    dup["payment_id"] = ("P" + pd.Series(np.arange(n_p + n_o, n_p + n_o + len(dup))).astype(str).str.zfill(8)).to_numpy()
    payment = pd.concat([payment, orphans, dup], ignore_index=True)

    imp = alloc.copy()
    pay_when = pd.Series(
        pay["booking_date"].to_numpy().astype("datetime64[s]")
        + (rng.choice([0, 0, 1, 3], n_p) * 86400 + rng.integers(8, 19, n_p) * 3600
           + rng.integers(0, 60, n_p) * 60).astype("timedelta64[s]"),
        index=pay["payment_id"],
    )
    imp["updated_at"] = pay_when.reindex(imp["payment_id"]).to_numpy()
    imp = imp.sort_values(["inv", "updated_at"], kind="mergesort")
    imp["residual"] = inv_amount[imp["inv"].to_numpy()] - imp.groupby("inv")["imputed"].cumsum()
    imputation = pd.DataFrame({
        "payment_id": imp["payment_id"].to_numpy(),
        "invoice_id": inv_ids[imp["inv"].to_numpy()],
        "status": np.where(imp["residual"].to_numpy() == 0, "TOTAL", "PARTIEL"),
        "updated_at": imp["updated_at"].to_numpy(),
        "residual_amount": imp["residual"].to_numpy(),
    })
    current = inv_amount.copy()
    np.subtract.at(current, imp["inv"].to_numpy(), imp["imputed"].to_numpy())

    invoice = pd.DataFrame({
        "invoice_id": inv_ids,
        "client_reference": inv_ref,
        "internal_reference": "INT" + pd.Series(inv_ids).str[1:],
        "creation_date": inv_created,
        "due_date": inv_due,
        "initial_amount": inv_amount,
        "current_amount": current,
        "currency": "EUR",
        "debtor_id": debtor["party_id"].to_numpy()[inv_debtor],
        "agreement_id": agreement["agreement_id"].to_numpy()[inv_ag],
    })

    # Client files : 40 % des paiements groupés (hors partiels).
    grouped = pay[(grp["size"].to_numpy()[pay["gid"].to_numpy()] > 1)]
    cf_pay = grouped[rng.random(len(grouped)) < 0.4].reset_index(drop=True)
    n_cf = len(cf_pay)
    cf_ids = _ids("CF", n_cf, 7)
    pay_idx = payment.set_index("payment_id")
    cf_label = pay_idx["label"].reindex(cf_pay["payment_id"]).to_numpy()
    client_file = pd.DataFrame({
        "file_id": cf_ids,
        "received_at": cf_pay["value_date"].to_numpy().astype("datetime64[s]")
        + (rng.integers(-2, 4, n_cf) * 86400 + rng.integers(7, 21, n_cf) * 3600).astype("timedelta64[s]"),
        "source_format": "CSV",
        "payment_reference": pd.Series(cf_label).str[:30].to_numpy(),
        "total_amount": cf_pay["amount"].to_numpy(),
        "payment_date": cf_pay["value_date"].to_numpy(),
        "iban": pay_idx["iban_debtor"].reindex(cf_pay["payment_id"]).to_numpy(),
        "issuer_name": debtor["name"].to_numpy()[g_debtor[cf_pay["gid"].to_numpy()]],
    })
    cf_of_pay = pd.Series(cf_ids, index=cf_pay["payment_id"])
    lines = alloc[alloc["payment_id"].isin(cf_of_pay.index)]
    client_file_line = pd.DataFrame({
        "file_id": cf_of_pay.reindex(lines["payment_id"]).to_numpy(),
        "line_no": lines.groupby("payment_id").cumcount().to_numpy() + 1,
        "invoice_reference": inv_ref[lines["inv"].to_numpy()],
        "amount": lines["amount_inv"].to_numpy(),
        "gap_reason": "",
    })

    return {
        "assignor": assignor, "debtor": debtor, "agreement": agreement,
        "technical_account": technical, "invoice": invoice, "payment": payment,
        "imputation": imputation, "client_file": client_file, "client_file_line": client_file_line,
    }


# --- Écriture au format source ---------------------------------------------------

_AMOUNT_COLUMNS = {"amount", "initial_amount", "current_amount", "residual_amount", "total_amount"}
_TIMESTAMP_COLUMNS = {"updated_at", "received_at"}


def _format_amount(cents: pd.Series) -> pd.Series:
    c = cents.astype(np.int64)
    a = c.abs()
    sign = np.where(c < 0, "-", "")
    return sign + (a // 100).astype(str) + "." + (a % 100).astype(str).str.zfill(2)


def _format_datetime(values: pd.Series, with_time: bool) -> pd.Series:
    arr = values.to_numpy()
    if with_time:
        out = np.char.replace(np.datetime_as_string(arr.astype("datetime64[s]"), unit="s"), "T", " ")
    else:
        out = np.datetime_as_string(arr.astype("datetime64[D]"), unit="D")
    return pd.Series(np.where(np.isnat(arr), "", out), index=values.index)


def to_source_format(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in _AMOUNT_COLUMNS:
            out[col] = _format_amount(df[col])
        elif df[col].to_numpy().dtype.kind == "M":
            out[col] = _format_datetime(df[col], with_time=col in _TIMESTAMP_COLUMNS)
        else:
            out[col] = df[col]
    return out


def write_synthetic(cfg: SyntheticConfig, out_dir: str | Path) -> dict[str, int]:
    """Écrit un CSV par table dans `out_dir`. Retourne le nombre de lignes par table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for table, df in generate(cfg).items():
        to_source_format(df).to_csv(out / f"{table}.csv", index=False, lineterminator="\n")
        counts[table] = len(df)
    return counts


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/timeline/state.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""État du grand livre à une date (brief §4.1, spec §8.2).

`LedgerState` rejoue le journal d'événements et répond aux questions du moteur
pour l'instant `as_of` auquel il a été avancé : il reflète exactement les
événements d'horodatage **strictement antérieur** à `as_of`.

Garanties anti-fuite, mécaniques :
- toute lecture exige un `as_of` égal à l'instant de l'état, sinon `TemporalError` ;
- l'état ne recule jamais (`advance_to` refuse un instant passé) ;
- une facture n'est visible qu'après son `INVOICE_CREATED`, un client file après
  son `CLIENT_FILE_RECEIVED` ;
- le restant dû est reconstruit par les imputations (`invoice.current_amount`
  n'est jamais chargé dans les tables).

Volumétrie : l'état est en tableaux numpy indexés par position ; le journal est
converti une fois en entiers et appliqué par blocs.
"""





DAY_US = 86_400_000_000
_CODE = {e: EVENT_RANK[e.value] for e in EventType}
_ROLES = ("assignor", "debtor")

# Agrégats comportementaux en fenêtre glissante (colonnes de `_win`).
_W_PAYMENTS, _W_LINES, _W_PARTIAL, _W_GROUPED, _W_CITED, _W_DELAY_N, _W_DELAY_SUM, _W_DELAY_SQ = range(8)
_N_WIN = 8

INVOICE_COLUMNS = [
    "invoice_id", "client_reference", "internal_reference", "creation_date", "due_date", "initial_amount",
    "currency", "debtor_id", "agreement_id", "client_reference_keys", "internal_reference_keys",
]


class TemporalError(RuntimeError):
    """Lecture ou avance de l'état incompatible avec l'instant qu'il représente."""


def _to_us(t: pd.Timestamp | np.datetime64 | str) -> int:
    return int(pd.Timestamp(t).as_unit("us").asm8.view("i8"))


def _days(values: pd.Series) -> np.ndarray:
    """Dates → numéro de jour (int64), NaT → valeur sentinelle très basse."""
    arr = values.to_numpy(dtype="datetime64[us]").astype("datetime64[D]")
    out = arr.astype(np.int64)
    out[np.isnat(arr)] = np.iinfo(np.int64).min // 2
    return out


class _Positions:
    """Identifiant → position dans une table (−1 si inconnu). Table de hachage construite une fois."""

    def __init__(self, ids: pd.Series):
        self.index = pd.Index(ids.astype(object).to_numpy())

    def __call__(self, ids) -> np.ndarray:
        values = ids.astype(object).to_numpy() if isinstance(ids, pd.Series) else np.asarray(ids, dtype=object)
        return self.index.get_indexer(values).astype(np.int64)

    def __len__(self) -> int:
        return len(self.index)


@dataclass(frozen=True)
class Journal:
    """Journal converti en tableaux : un événement par ligne, dans l'ordre du journal."""

    ts: np.ndarray       # int64, µs
    code: np.ndarray     # int8, rang du type d'événement
    pos: np.ndarray      # int64, position de l'entité dans sa table
    pos2: np.ndarray     # int64, facture (imputation) ou rôle (partie), sinon −1
    amount: np.ndarray   # int64, montant imputé (imputation), sinon 0

    def __len__(self) -> int:
        return len(self.ts)


class LedgerState:
    def __init__(self, data: LoadedData, journal: pd.DataFrame, window_days: int = 180):
        t = data.tables
        self.window_days = window_days
        self._inv = t["invoice"].reset_index(drop=True)
        self._pay = t["payment"].reset_index(drop=True)
        self._agr = t["agreement"].reset_index(drop=True)
        self._parties = {r: t[r].reset_index(drop=True) for r in _ROLES}
        self._cf = t["client_file"].reset_index(drop=True) if "client_file" in t else None
        self._cfl = t["client_file_line"].reset_index(drop=True) if "client_file_line" in t else None
        self._tech = t["technical_account"].reset_index(drop=True) if "technical_account" in t else None
        self._party_iban = t["party_iban"].reset_index(drop=True) if "party_iban" in t else None

        self.inv_pos = _Positions(self._inv["invoice_id"])
        self.pay_pos = _Positions(self._pay["payment_id"])
        self.agr_pos = _Positions(self._agr["agreement_id"])
        self.party_pos = {r: _Positions(p["party_id"]) for r, p in self._parties.items()}
        self.cf_pos = _Positions(self._cf["file_id"]) if self._cf is not None else None
        debtor_pos = self.party_pos["debtor"]

        n_inv, n_pay, n_deb = len(self._inv), len(self._pay), len(self._parties["debtor"])
        # Attributs statiques utilisés par l'état.
        self._inv_initial = self._inv["initial_amount"].fillna(0).to_numpy(dtype=np.int64)
        self._inv_debtor = debtor_pos(self._inv["debtor_id"])
        self._inv_due_day = _days(self._inv["due_date"])
        self._pay_value_day = _days(self._pay["value_date"])
        # Débiteur → factures (CSR) pour les lectures par débiteur.
        order = np.argsort(self._inv_debtor, kind="stable")
        self._by_debtor = order
        self._debtor_offsets = np.searchsorted(self._inv_debtor[order], np.arange(n_deb + 1))

        # État mutable.
        self._cursor = 0                       # prochain événement à appliquer
        self._as_of: int | None = None         # µs ; événements < as_of appliqués
        self._inv_created = np.zeros(n_inv, dtype=bool)
        self._balance = np.zeros(n_inv, dtype=np.int64)
        self._pay_received = np.zeros(n_pay, dtype=bool)
        self._pay_imputed = np.zeros(n_pay, dtype=np.int64)
        self._pay_lines = np.zeros(n_pay, dtype=np.int64)
        self._agr_active = np.zeros(len(self._agr), dtype=bool)
        # Partie sans date d'ouverture : active depuis toujours.
        self._party_active = {r: p["opened_at"].isna().to_numpy().copy() for r, p in self._parties.items()}
        self._party_known = {r: a.copy() for r, a in self._party_active.items()}
        self._cf_received = np.zeros(len(self._cf) if self._cf is not None else 0, dtype=bool)
        self._open_count = np.zeros(n_deb, dtype=np.int64)
        self._open_amount = np.zeros(n_deb, dtype=np.int64)
        self._win = np.zeros((n_deb, _N_WIN), dtype=np.float64)
        self._contrib: deque[tuple[int, np.ndarray, np.ndarray]] = deque()   # (jour, débiteurs, valeurs)

        self.journal = self._encode(journal)

    # --- Construction ----------------------------------------------------------------------

    def _encode(self, journal: pd.DataFrame) -> Journal:
        j = journal.reset_index(drop=True)
        n = len(j)
        code = j["event_type"].map(EVENT_RANK).to_numpy(dtype=np.int8)
        pos = np.full(n, -1, dtype=np.int64)
        pos2 = np.full(n, -1, dtype=np.int64)
        ent = j["entity_id"]

        def fill(types: tuple[EventType, ...], positions: _Positions | None) -> np.ndarray:
            mask = np.isin(code, [_CODE[t] for t in types])
            if positions is not None and mask.any():
                pos[mask] = positions(ent[mask])
            return mask

        fill((EventType.INVOICE_CREATED,), self.inv_pos)
        fill((EventType.PAYMENT_RECEIVED,), self.pay_pos)
        fill((EventType.AGREEMENT_CREATED, EventType.AGREEMENT_DISABLED), self.agr_pos)
        fill((EventType.CLIENT_FILE_RECEIVED,), self.cf_pos)
        imp = fill((EventType.IMPUTATION_APPLIED,), self.pay_pos)
        pos2[imp] = self.inv_pos(j["related_id"][imp])
        party = np.isin(code, [_CODE[EventType.PARTY_OPENED], _CODE[EventType.PARTY_CLOSED]])
        for r, role in enumerate(_ROLES):
            mask = party & (j["related_id"] == role.upper()).to_numpy()
            pos[mask] = self.party_pos[role](ent[mask])
            pos2[mask] = r
        ts = j["ts"].to_numpy(dtype="datetime64[us]").astype(np.int64)
        if np.any(np.diff(ts) < 0):
            raise ValueError("journal non trié par horodatage")
        return Journal(ts, code, pos, pos2, j["amount"].fillna(0).to_numpy(dtype=np.int64))

    # --- Avance -------------------------------------------------------------------------------

    @property
    def as_of(self) -> pd.Timestamp | None:
        return None if self._as_of is None else pd.Timestamp(self._as_of, unit="us")

    def advance_to(self, as_of) -> None:
        """Applique tous les événements d'horodatage strictement antérieur à `as_of`."""
        target = _to_us(as_of)
        if self._as_of is not None and target < self._as_of:
            raise TemporalError(f"l'état est au {self.as_of}, il ne peut pas revenir au {pd.Timestamp(as_of)}")
        ts = self.journal.ts
        hi = int(np.searchsorted(ts, target, side="left"))
        # Application jour par jour : le résultat ne dépend pas de la taille des sauts.
        while self._cursor < hi:
            day_end = (int(ts[self._cursor]) // DAY_US + 1) * DAY_US
            nxt = min(hi, int(np.searchsorted(ts, day_end, side="left")))
            self._apply(self._cursor, nxt)
            self._cursor = nxt
        self._as_of = target
        self._expire(target // DAY_US - self.window_days)

    def _apply(self, lo: int, hi: int) -> None:
        """Applique les événements [lo, hi) d'un même jour."""
        j = self.journal
        code, pos, pos2, amount = j.code[lo:hi], j.pos[lo:hi], j.pos2[lo:hi], j.amount[lo:hi]
        day = j.ts[lo:hi] // DAY_US

        def sel(event: EventType) -> np.ndarray:
            return (code == _CODE[event]) & (pos >= 0)

        for event, active in ((EventType.PARTY_OPENED, True), (EventType.PARTY_CLOSED, False)):
            m = sel(event)
            for r, role in enumerate(_ROLES):
                self._party_active[role][pos[m & (pos2 == r)]] = active
                if active:
                    self._party_known[role][pos[m & (pos2 == r)]] = True
        self._agr_active[pos[sel(EventType.AGREEMENT_CREATED)]] = True
        self._agr_active[pos[sel(EventType.AGREEMENT_DISABLED)]] = False
        self._cf_received[pos[sel(EventType.CLIENT_FILE_RECEIVED)]] = True
        self._pay_received[pos[sel(EventType.PAYMENT_RECEIVED)]] = True

        created = pos[sel(EventType.INVOICE_CREATED)]
        imp = sel(EventType.IMPUTATION_APPLIED) & (pos2 >= 0)
        imp_inv = pos2[imp]
        touched = np.unique(np.concatenate([created, imp_inv]))
        before = np.where(self._inv_created[touched], np.maximum(self._balance[touched], 0), 0)
        was_open = self._inv_created[touched] & (self._balance[touched] > 0)

        self._inv_created[created] = True
        self._balance[created] = self._inv_initial[created]
        np.subtract.at(self._balance, imp_inv, amount[imp])
        np.add.at(self._pay_imputed, pos[imp], amount[imp])
        np.add.at(self._pay_lines, pos[imp], 1)

        after = np.where(self._inv_created[touched], np.maximum(self._balance[touched], 0), 0)
        is_open = self._inv_created[touched] & (self._balance[touched] > 0)
        deb = self._inv_debtor[touched]
        ok = deb >= 0
        np.add.at(self._open_count, deb[ok], is_open[ok].astype(np.int64) - was_open[ok])
        np.add.at(self._open_amount, deb[ok], after[ok] - before[ok])

        if imp.any():
            self._add_window(day[imp], pos[imp], imp_inv)

    def _add_window(self, day: np.ndarray, pay: np.ndarray, inv: np.ndarray) -> None:
        """Contributions des imputations aux agrégats glissants, par jour d'imputation."""
        deb = self._inv_debtor[inv]
        keep = deb >= 0
        day, pay, inv, deb = day[keep], pay[keep], inv[keep], deb[keep]
        lines = pd.DataFrame({"day": day, "pay": pay, "inv": inv, "deb": deb})
        # Ligne groupée : le paiement impute plusieurs factures le même jour.
        per_pay = lines.groupby(["day", "pay"])["inv"].transform("nunique").to_numpy()
        first_of_pay = ~lines.duplicated(["day", "pay", "deb"]).to_numpy()
        delay = (self._pay_value_day[pay] - self._inv_due_day[inv]).astype(np.float64)
        has_delay = np.abs(delay) < 1e6
        delay = np.where(has_delay, delay, 0.0)
        pay_numbers = self._pay["label_numbers"].to_numpy()
        inv_keys = self._inv["client_reference_keys"].to_numpy()
        cited = np.fromiter((bool(set(pay_numbers[p]) & set(inv_keys[i])) for p, i in zip(pay, inv)),
                            dtype=bool, count=len(pay))
        values = np.zeros((len(lines), _N_WIN))
        values[:, _W_PAYMENTS] = first_of_pay
        values[:, _W_LINES] = 1
        values[:, _W_PARTIAL] = self._balance[inv] > 0
        values[:, _W_GROUPED] = per_pay > 1
        values[:, _W_CITED] = cited
        values[:, _W_DELAY_N] = has_delay
        values[:, _W_DELAY_SUM] = delay
        values[:, _W_DELAY_SQ] = delay * delay
        np.add.at(self._win, deb, values)
        for d in np.unique(day):
            m = day == d
            self._contrib.append((int(d), deb[m], values[m]))

    def _expire(self, first_kept_day: int) -> None:
        while self._contrib and self._contrib[0][0] < first_kept_day:
            _, deb, values = self._contrib.popleft()
            np.subtract.at(self._win, deb, values)

    # --- Lectures (toutes exigent as_of) ------------------------------------------------------

    def _check(self, as_of) -> None:
        if self._as_of is None:
            raise TemporalError("l'état n'a pas encore été avancé")
        if _to_us(as_of) != self._as_of:
            raise TemporalError(f"lecture au {pd.Timestamp(as_of)} sur un état au {self.as_of}")

    def _invoice_frame(self, positions: np.ndarray) -> pd.DataFrame:
        df = self._inv.iloc[positions][INVOICE_COLUMNS].reset_index(drop=True)
        df["open_amount"] = pd.array(self._balance[positions], dtype="Int64")
        return df

    def open_invoices(self, debtor_ids, as_of) -> pd.DataFrame:
        """Factures créées et non soldées à `as_of` des débiteurs donnés, avec leur restant dû."""
        self._check(as_of)
        deb = self.party_pos["debtor"](pd.Series(debtor_ids, dtype=object))
        deb = np.unique(deb[deb >= 0])
        if len(deb):
            starts, ends = self._debtor_offsets[deb], self._debtor_offsets[deb + 1]
            lengths = ends - starts
            idx = np.repeat(ends - lengths.cumsum(), lengths) + np.arange(lengths.sum())
            cand = self._by_debtor[idx]
        else:
            cand = np.array([], dtype=np.int64)
        cand = cand[self._inv_created[cand] & (self._balance[cand] > 0)]
        return self._invoice_frame(np.sort(cand))

    def invoices(self, invoice_ids, as_of) -> pd.DataFrame:
        """Factures demandées déjà créées à `as_of` (les autres sont ignorées), avec leur restant dû."""
        self._check(as_of)
        p = self.inv_pos(pd.Series(invoice_ids, dtype=object))
        p = p[p >= 0]
        return self._invoice_frame(p[self._inv_created[p]])

    def open_amount(self, invoice_ids, as_of) -> pd.Series:
        """Restant dû à `as_of` (`current_amount_as_of`), NA si la facture n'existe pas encore."""
        self._check(as_of)
        p = self.inv_pos(pd.Series(invoice_ids, dtype=object))
        known = (p >= 0) & self._inv_created[np.maximum(p, 0)]
        out = pd.array(np.where(known, self._balance[np.maximum(p, 0)], 0), dtype="Int64")
        out[~known] = pd.NA
        return pd.Series(out, index=invoice_ids.index if isinstance(invoice_ids, pd.Series) else None)

    def party_active(self, role: str, party_ids, as_of) -> np.ndarray:
        self._check(as_of)
        p = self.party_pos[role](pd.Series(party_ids, dtype=object))
        return np.where(p >= 0, self._party_active[role][np.maximum(p, 0)], False)

    def agreement_active(self, agreement_ids, as_of) -> np.ndarray:
        self._check(as_of)
        p = self.agr_pos(pd.Series(agreement_ids, dtype=object))
        return np.where(p >= 0, self._agr_active[np.maximum(p, 0)], False)

    def client_files(self, as_of) -> pd.DataFrame:
        """Client files reçus avant `as_of`."""
        self._check(as_of)
        if self._cf is None:
            return pd.DataFrame()
        return self._cf[self._cf_received].reset_index(drop=True)

    def client_file_lines(self, file_ids, as_of) -> pd.DataFrame:
        """Lignes des client files demandés, uniquement ceux déjà reçus."""
        self._check(as_of)
        if self._cf is None or self._cfl is None:
            return pd.DataFrame()
        p = self.cf_pos(pd.Series(file_ids, dtype=object))
        received = set(self._cf["file_id"].iloc[p[(p >= 0)][self._cf_received[p[p >= 0]]]])
        return self._cfl[self._cfl["file_id"].isin(received)].reset_index(drop=True)

    def payment_imputed(self, payment_ids, as_of) -> np.ndarray:
        """Le paiement a-t-il déjà au moins une imputation prononcée avant `as_of` ?"""
        self._check(as_of)
        p = self.pay_pos(pd.Series(payment_ids, dtype=object))
        return np.where(p >= 0, self._pay_lines[np.maximum(p, 0)] > 0, False)

    def debtor_stats(self, debtor_ids, as_of) -> pd.DataFrame:
        """Agrégats comportementaux à `as_of` : encours (instantané) et historique sur la fenêtre
        glissante `[as_of − window_days, as_of)`, strictement antérieure."""
        self._check(as_of)
        ids = pd.Series(debtor_ids, dtype=object).reset_index(drop=True)
        p = self.party_pos["debtor"](ids)
        ok = p >= 0
        q = np.maximum(p, 0)
        w = np.where(ok[:, None], self._win[q], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lines = w[:, _W_LINES]
            n_delay = w[:, _W_DELAY_N]
            mean = w[:, _W_DELAY_SUM] / n_delay
            var = np.maximum(w[:, _W_DELAY_SQ] / n_delay - mean * mean, 0.0)
            out = pd.DataFrame({
                "debtor_id": ids,
                "open_invoice_count": np.where(ok, self._open_count[q], 0),
                "open_invoice_amount": np.where(ok, self._open_amount[q], 0),
                "payment_count": np.rint(w[:, _W_PAYMENTS]).astype(np.int64),
                "imputation_count": np.rint(lines).astype(np.int64),
                "mean_payment_delay": np.where(n_delay > 0, mean, np.nan),
                "std_payment_delay": np.where(n_delay > 0, np.sqrt(var), np.nan),
                "partial_payment_rate": np.where(lines > 0, w[:, _W_PARTIAL] / lines, np.nan),
                "grouping_rate": np.where(lines > 0, w[:, _W_GROUPED] / lines, np.nan),
                "ref_citation_rate": np.where(lines > 0, w[:, _W_CITED] / lines, np.nan),
            })
        return out

    # --- Accès par positions (index de l'allocation, itérateur) -----------------------------------------
    # Les attributs statiques (débiteur d'une facture, nom d'un débiteur) ne dépendent pas du temps ;
    # ce qui en dépend (existence à D) passe par des lectures contrôlées.

    @property
    def invoice_debtor_positions(self) -> np.ndarray:
        return self._inv_debtor

    def invoice_created_at(self, positions: np.ndarray, as_of) -> np.ndarray:
        self._check(as_of)
        return self._inv_created[positions]

    def party_known_at(self, role: str, positions: np.ndarray, as_of) -> np.ndarray:
        """La partie existe-t-elle à `as_of` (ouverte, même fermée depuis) ?"""
        self._check(as_of)
        return self._party_known[role][positions]

    def known_party_count(self, role: str, as_of) -> int:
        self._check(as_of)
        return int(self._party_known[role].sum())

    def open_balance_at(self, positions: np.ndarray, as_of) -> np.ndarray:
        """Restant dû à `as_of` par positions ; 0 pour une facture pas encore créée."""
        self._check(as_of)
        return np.where(self._inv_created[positions], self._balance[positions], 0)

    def debtor_open_invoices_at(self, debtor_positions: np.ndarray, as_of) -> tuple[np.ndarray, np.ndarray,
                                                                                    np.ndarray]:
        """Factures ouvertes à `as_of` des débiteurs donnés : (index du débiteur demandé, facture, restant dû)."""
        self._check(as_of)
        deb = np.asarray(debtor_positions, dtype=np.int64)
        starts, ends = self._debtor_offsets[deb], self._debtor_offsets[deb + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            return (np.array([], dtype=np.int64),) * 3
        owner = np.repeat(np.arange(len(deb)), lengths)
        inv = self._by_debtor[np.repeat(ends - lengths.cumsum(), lengths) + np.arange(total)]
        ok = self._inv_created[inv] & (self._balance[inv] > 0)
        return owner[ok], inv[ok], self._balance[inv[ok]]

    def agreement_active_at(self, positions: np.ndarray, as_of) -> np.ndarray:
        self._check(as_of)
        return self._agr_active[positions]

    def open_invoice_positions(self, as_of) -> tuple[np.ndarray, np.ndarray]:
        """Positions et restant dû des factures ouvertes à `as_of`."""
        self._check(as_of)
        pos = np.flatnonzero(self._inv_created & (self._balance > 0))
        return pos, self._balance[pos]

    def client_files_received_at(self, as_of) -> np.ndarray:
        """Positions des client files reçus avant `as_of`."""
        self._check(as_of)
        return np.flatnonzero(self._cf_received)

    def table(self, name: str) -> pd.DataFrame:
        """Table statique (attributs non temporels). Toute lecture d'existence passe par les accesseurs datés."""
        return {"invoice": self._inv, "payment": self._pay, "agreement": self._agr, "client_file": self._cf,
                "client_file_line": self._cfl, "technical_account": self._tech,
                "party_iban": self._party_iban, **self._parties}[name]

    # --- Accès pour l'itérateur ---------------------------------------------------------------------

    def payments_frame(self, positions: np.ndarray) -> pd.DataFrame:
        return self._pay.iloc[positions].reset_index(drop=True)

    def payments_arriving(self, start, end) -> tuple[np.ndarray, np.ndarray]:
        """Paiements dont l'événement PAYMENT_RECEIVED tombe dans [start, end) : (positions, numéros de jour).

        Lu dans le journal, pas dans la table : c'est l'arrivée du lot, pas une lecture d'état.
        """
        j = self.journal
        lo, hi = np.searchsorted(j.ts, [_to_us(start), _to_us(end)], side="left")
        m = (j.code[lo:hi] == _CODE[EventType.PAYMENT_RECEIVED]) & (j.pos[lo:hi] >= 0)
        return j.pos[lo:hi][m], j.ts[lo:hi][m] // DAY_US

    def payment_day_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Premier et dernier jour d'arrivée de paiement dans le journal."""
        j = self.journal
        ts = j.ts[j.code == _CODE[EventType.PAYMENT_RECEIVED]]
        return pd.Timestamp(int(ts.min() // DAY_US), unit="D"), pd.Timestamp(int(ts.max() // DAY_US), unit="D")

    def payment_imputed_at(self, positions: np.ndarray, as_of) -> np.ndarray:
        """Comme `payment_imputed`, par positions (usage interne de l'itérateur)."""
        self._check(as_of)
        return self._pay_lines[positions] > 0


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/timeline/split.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Découpage temporel en périodes train / validation / test (brief §4.2).

Découpage sur les jours, jamais aléatoire, ancré sur la fin de l'historique :
le test couvre les derniers mois, la validation les mois précédents, et
l'entraînement le reste. `purge_days` jours sont exclus entre deux blocs.
Toutes les bornes sont inclusives.
"""





PERIODS = ("train", "validation", "test")
PURGE = "purge"
OUTSIDE = "hors période"


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class Period:
    name: str
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass(frozen=True)
class Split:
    periods: tuple[Period, ...]
    warnings: tuple[str, ...] = field(default=())

    def period(self, name: str) -> Period:
        return next(p for p in self.periods if p.name == name)


def _months_before(end: date, months: int) -> date:
    """Premier jour d'une période de `months` mois se terminant le jour `end`."""
    return (pd.Timestamp(end) + pd.Timedelta(days=1) - pd.DateOffset(months=months)).date()


def compute_split(cfg: SplitSettings, data_start: date, data_end: date) -> Split:
    if data_start > data_end:
        raise SplitError("plage de données vide")
    end = cfg.anchor_end or data_end
    if not data_start <= end <= data_end:
        raise SplitError(f"anchor_end {end} hors de la plage des données [{data_start}, {data_end}]")
    gap = timedelta(days=cfg.purge_days + 1)

    test = Period("test", _months_before(end, cfg.test_months), end)
    val_end = test.start - gap
    validation = Period("validation", _months_before(val_end, cfg.validation_months), val_end)
    train_end = validation.start - gap
    train_start = data_start
    if cfg.train_months is not None:
        train_start = max(data_start, _months_before(train_end, cfg.train_months))
    if train_start > train_end:
        raise SplitError("période d'entraînement vide : historique trop court pour ces durées")
    train = Period("train", train_start, train_end)

    warnings: list[str] = []
    if cfg.train_months is not None and train_start == data_start and \
            _months_before(train_end, cfg.train_months) < data_start:
        warnings.append(f"historique insuffisant : l'entraînement couvre {train.days} jours "
                        f"au lieu de {cfg.train_months} mois")
    if validation.start < data_start:
        raise SplitError("période de validation hors des données : historique trop court")
    return Split((train, validation, test), tuple(warnings))


def assign_period(days: pd.Series, split: Split) -> pd.Series:
    """Nom de la période de chaque date : train / validation / test / purge / hors période."""
    d = pd.to_datetime(days).dt.normalize()
    out = np.full(len(d), OUTSIDE, dtype=object)
    first, last = split.periods[0].start, split.periods[-1].end
    inside = (d >= pd.Timestamp(first)) & (d <= pd.Timestamp(last))
    out[inside.to_numpy()] = PURGE
    for p in split.periods:
        mask = (d >= pd.Timestamp(p.start)) & (d <= pd.Timestamp(p.end))
        out[mask.to_numpy()] = p.name
    return pd.Series(out, index=days.index)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/timeline/loop.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Boucle quotidienne (brief §4.1) : un seul code pour le rejeu de l'historique et la production.

Pour chaque jour D, `DailyIterator` avance l'état à D (événements < D, soit
l'état figé à la veille) et fournit le lot à traiter : paiements arrivés le
jour D + reliquat non résolu des jours précédents, dans la limite de
`retention_days`. Un paiement quitte le reliquat quand :
- une imputation réelle a été prononcée avant D (traité par ailleurs) ;
- le rapprocheur l'a auto-validé ;
- sa rétention est dépassée (il sort du cycle automatique).

Tout paiement en reliquat est retraité chaque jour : il l'est donc notamment
le jour où son client file devient visible.

Un rapprocheur (`Matcher`) reçoit un `DayContext` et renvoie ses décisions ;
`run_replay` les contrôle (paiement du lot, facture déjà connue à D) — une
décision qui cite une facture future est une fuite et lève `LeakError`.

En rejeu, l'état suit le journal réel : les décisions du moteur ne modifient
pas les soldes, elles sont comparées aux imputations réelles à l'évaluation.
"""





AUTO, REVIEW, REJECT = "auto", "review", "reject"
ACTIONS = (AUTO, REVIEW, REJECT)
DECISION_COLUMNS = ["payment_id", "invoice_id", "amount", "action", "step", "rule_id", "rule_version", "score"]


class LeakError(RuntimeError):
    """Une décision utilise une information non disponible au jour D."""


@dataclass
class DayContext:
    day: pd.Timestamp
    batch: pd.DataFrame          # paiements à traiter + is_new, first_day, days_pending
    state: LedgerState

    @property
    def as_of(self) -> pd.Timestamp:
        """Instant de l'état : minuit du jour D (événements strictement antérieurs)."""
        return self.day


class Matcher(Protocol):
    name: str

    def process(self, ctx: DayContext) -> pd.DataFrame:
        """Décisions du jour, colonnes `DECISION_COLUMNS` (une ligne par paiement × facture)."""
        ...


class NullMatcher:
    """Rapprocheur vide : ne décide rien. Sert de référence et de test du harnais."""

    name = "null"

    def process(self, ctx: DayContext) -> pd.DataFrame:
        return empty_decisions()


def empty_decisions() -> pd.DataFrame:
    return pd.DataFrame({
        "payment_id": pd.Series(dtype="string"), "invoice_id": pd.Series(dtype="string"),
        "amount": pd.Series(dtype="Int64"), "action": pd.Series(dtype="string"),
        "step": pd.Series(dtype="string"), "rule_id": pd.Series(dtype="string"),
        "rule_version": pd.Series(dtype="Int64"), "score": pd.Series(dtype="float64"),
    })


class DailyIterator:
    def __init__(self, state: LedgerState, start: date, end: date, retention_days: int = 60):
        if start > end:
            raise ValueError("start postérieur à end")
        self.state = state
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.retention_days = retention_days
        self._pending = np.array([], dtype=np.int64)      # positions des paiements en reliquat
        self._first_day = np.array([], dtype=np.int64)    # jour d'arrivée (numéro de jour)
        self._resolved: set[int] = set()
        self.last_counts: dict[str, int] = {}

    def resolve(self, payment_ids) -> None:
        """Retire du reliquat des paiements auto-validés par le rapprocheur."""
        p = self.state.pay_pos(pd.Series(list(payment_ids), dtype=object))
        self._resolved.update(int(x) for x in p[p >= 0])

    def __iter__(self) -> Iterator[DayContext]:
        state = self.state
        # Reliquat initial : paiements arrivés dans la fenêtre de rétention avant `start`.
        warm = self.start - pd.Timedelta(days=self.retention_days)
        state.advance_to(warm)
        self._pending, self._first_day = state.payments_arriving(warm, self.start)

        day = self.start
        while day <= self.end:
            state.advance_to(day)
            today = _to_us(day) // DAY_US
            new, _ = state.payments_arriving(day, day + pd.Timedelta(days=1))

            pend, first = self._pending, self._first_day
            imputed = state.payment_imputed_at(pend, day)
            engine = np.isin(pend, np.fromiter(self._resolved, dtype=np.int64, count=len(self._resolved)))
            expired = (today - first) > self.retention_days
            keep = ~(imputed | engine | expired)
            self.last_counts = {"new": len(new), "carried": int(keep.sum()), "left_imputed": int(imputed.sum()),
                                "left_engine": int((engine & ~imputed).sum()),
                                "expired": int((expired & ~imputed & ~engine).sum())}

            positions = np.concatenate([pend[keep], new])
            first_days = np.concatenate([first[keep], np.full(len(new), today, dtype=np.int64)])
            batch = state.payments_frame(positions)
            batch["is_new"] = np.concatenate([np.zeros(keep.sum(), bool), np.ones(len(new), bool)])
            batch["first_day"] = pd.to_datetime(first_days, unit="D")
            batch["days_pending"] = today - first_days
            self._pending, self._first_day = positions, first_days

            yield DayContext(day=day, batch=batch, state=state)
            day += pd.Timedelta(days=1)



@dataclass
class ReplayResult:
    decisions: pd.DataFrame
    daily: pd.DataFrame
    matcher: str
    start: date
    end: date
    seconds: float
    extra: dict = field(default_factory=dict)


def _validate(decisions: pd.DataFrame, ctx: DayContext) -> pd.DataFrame:
    missing = [c for c in DECISION_COLUMNS if c not in decisions.columns]
    if missing:
        raise ValueError(f"décisions sans les colonnes {missing}")
    dec = decisions[DECISION_COLUMNS].copy()
    if len(dec) == 0:
        return dec
    bad_action = ~dec["action"].isin(ACTIONS)
    if bad_action.any():
        raise ValueError(f"actions inconnues : {sorted(dec.loc[bad_action, 'action'].unique())}")
    outside = ~dec["payment_id"].isin(set(ctx.batch["payment_id"]))
    if outside.any():
        raise LeakError(f"{int(outside.sum())} décision(s) sur des paiements hors du lot du {ctx.day.date()}")
    cited = dec["invoice_id"].dropna().unique()
    known = set(ctx.state.invoices(cited, ctx.as_of)["invoice_id"])
    unknown = [i for i in cited if i not in known]
    if unknown:
        raise LeakError(f"{len(unknown)} facture(s) inconnue(s) au {ctx.day.date()} citées par le rapprocheur, "
                        f"ex. {unknown[:3]}")
    return dec


def run_replay(state: LedgerState, matcher: Matcher, start: date, end: date, retention_days: int = 60,
               on_day: Callable[[DayContext, dict], None] | None = None) -> ReplayResult:
    """Rejoue les jours [start, end] avec `matcher` ; décisions contrôlées et journalisées."""
    t0 = time.perf_counter()
    it = DailyIterator(state, start, end, retention_days)
    frames, daily = [], []
    for ctx in it:
        t = time.perf_counter()
        memory.mark_day(ctx.day, len(ctx.batch))
        dec = _validate(matcher.process(ctx), ctx)
        it.resolve(dec.loc[dec["action"] == AUTO, "payment_id"].unique())
        if len(dec):
            frames.append(dec.assign(day=ctx.day))
        row = {"day": ctx.day, "batch": len(ctx.batch), **it.last_counts,
               "auto": int(dec.loc[dec["action"] == AUTO, "payment_id"].nunique()),
               "review": int(dec.loc[dec["action"] == REVIEW, "payment_id"].nunique()),
               "seconds": round(time.perf_counter() - t, 4)}
        daily.append(row)
        memory.end_day()
        if on_day is not None:
            on_day(ctx, row)
    decisions = pd.concat(frames, ignore_index=True) if frames else empty_decisions().assign(
        day=pd.Series(dtype="datetime64[us]"))
    return ReplayResult(decisions, pd.DataFrame(daily), getattr(matcher, "name", type(matcher).__name__),
                        pd.Timestamp(start).date(), pd.Timestamp(end).date(), round(time.perf_counter() - t0, 1))


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/allocation/indexes.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Index de l'allocation, construits une fois sur les attributs statiques.

Aucun index ne porte d'information temporelle : ils listent toutes les factures,
tous les débiteurs, tous les IBAN connus du référentiel. Ce qui dépend du temps
— la facture existe-t-elle à D, le débiteur est-il connu à D, combien de
débiteurs partagent une clé à D — est filtré à la requête par l'état daté.
Aucune recherche ne parcourt tous les débiteurs : on part toujours du paiement.
"""





def _flatten(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Colonne de listes → (longueurs, valeurs aplaties en objets)."""
    arr = values.to_numpy() if isinstance(values, pd.Series) else values
    lengths = np.fromiter((0 if v is None or (isinstance(v, float)) else len(v) for v in arr),
                          dtype=np.int64, count=len(arr))
    flat = np.fromiter(chain.from_iterable(v for v in arr if v is not None and not isinstance(v, float)),
                       dtype=object, count=int(lengths.sum()))
    return lengths, flat


@dataclass
class Postings:
    """Listes d'occurrences compactes (CSR) : clé entière → valeurs entières."""

    offsets: np.ndarray
    values: np.ndarray

    @classmethod
    def build(cls, keys: np.ndarray, values: np.ndarray, n_keys: int) -> Postings:
        ok = (keys >= 0) & (values >= 0)
        keys, values = keys[ok], values[ok]
        order = np.lexsort((values, keys))
        keys, values = keys[order], values[order]
        # Déduplication (clé, valeur).
        keep = np.ones(len(keys), dtype=bool)
        keep[1:] = (keys[1:] != keys[:-1]) | (values[1:] != values[:-1])
        keys, values = keys[keep], values[keep]
        offsets = np.searchsorted(keys, np.arange(n_keys + 1))
        return cls(offsets, values)

    @classmethod
    def from_lists(cls, lengths: np.ndarray, flat_ids: np.ndarray) -> Postings:
        """Propriétaire i → ses ids (issus d'une colonne de listes aplatie)."""
        owners = np.repeat(np.arange(len(lengths)), lengths)
        return cls.build(owners, flat_ids, len(lengths))

    def lengths(self, keys: np.ndarray) -> np.ndarray:
        return self.offsets[keys + 1] - self.offsets[keys]

    def gather(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pour chaque clé demandée, ses valeurs : (index de la clé dans `keys`, valeur)."""
        keys = np.asarray(keys, dtype=np.int64)
        lengths = self.lengths(keys)
        total = int(lengths.sum())
        if total == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        owner = np.repeat(np.arange(len(keys)), lengths)
        starts = np.repeat(self.offsets[keys] - (np.cumsum(lengths) - lengths), lengths)
        return owner, self.values[starts + np.arange(total)]


def _hash(values: np.ndarray) -> np.ndarray:
    """Empreintes 64 bits déterministes de chaînes (valeurs nulles : 0, jamais dans un vocabulaire)."""
    values = np.asarray(values, dtype=object)
    if len(values) == 0:
        return np.array([], dtype=np.uint64)
    return pd.util.hash_array(values, categorize=False)


class Vocabulary:
    """Chaînes → identifiants entiers (−1 si inconnue).

    Stocke les empreintes 64 bits triées des chaînes, pas les chaînes : quelques octets par entrée
    au lieu d'un objet Python et d'une table de hachage pandas. Une collision entre deux clés
    distinctes (probabilité ~ n² / 2⁶⁵, soit 10⁻⁵ pour 30 millions de clés) les confondrait.
    """

    def __init__(self, values: np.ndarray):
        values = np.asarray(values, dtype=object)
        valid = ~pd.isna(values)
        hashes = _hash(values[valid])
        self.hashes, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
        self.codes = np.full(len(values), -1, dtype=np.int64)
        self.codes[valid] = inverse
        kept = values[valid][first]
        self.lengths = np.fromiter((len(v) for v in kept), dtype=np.int64, count=len(kept))

    def __len__(self) -> int:
        return len(self.hashes)

    def lookup(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=object)
        out = np.full(len(values), -1, dtype=np.int64)
        if len(values) == 0 or len(self.hashes) == 0:
            return out
        valid = ~pd.isna(values)
        h = _hash(values[valid])
        # Requêtes triées : la recherche dichotomique parcourt alors le vocabulaire dans l'ordre (cache).
        order = np.argsort(h, kind="stable")
        i = np.empty(len(h), dtype=np.int64)
        i[order] = np.searchsorted(self.hashes, h[order])
        i = np.minimum(i, len(self.hashes) - 1)
        out[valid] = np.where(self.hashes[i] == h, i, -1)
        return out


# Les listes de clés / termes des libellés sont traduites par blocs : pas de copie en objets Python
# de tous les libellés de l'historique à la fois.
_PAYMENT_CHUNK = 200_000


def _payment_postings(n: int, ids_of_chunk) -> Postings:
    """Paiement → identifiants, construits par blocs de `_PAYMENT_CHUNK` paiements.

    `ids_of_chunk(start, end)` rend (propriétaire relatif au bloc, identifiant) ; −1 = hors vocabulaire.
    """
    owners, ids = [np.array([], dtype=np.int64)], [np.array([], dtype=np.int64)]
    for start in range(0, n, _PAYMENT_CHUNK):
        o, i = ids_of_chunk(start, min(start + _PAYMENT_CHUNK, n))
        known = i >= 0
        owners.append(o[known] + start)
        ids.append(i[known])
    return Postings.build(np.concatenate(owners), np.concatenate(ids), n)


class ReferenceIndex:
    """Clé de référence → factures ; paiement → clés de son libellé."""

    def __init__(self, invoices: pd.DataFrame, payments: pd.DataFrame):
        parts = []
        for col in ("client_reference_keys", "internal_reference_keys"):
            lengths, flat = _flatten(invoices[col])
            parts.append((np.repeat(np.arange(len(invoices)), lengths), flat))
        inv_pos = np.concatenate([p[0] for p in parts])
        flat = np.concatenate([p[1] for p in parts])
        self.vocab = Vocabulary(flat)
        self.key_invoices = Postings.build(self.vocab.codes, inv_pos, len(self.vocab))
        numbers = payments["label_numbers"]

        def keys_of(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
            lengths, flat = _flatten(numbers.iloc[start:end])
            return np.repeat(np.arange(end - start), lengths), self.vocab.lookup(flat)

        self.payment_keys = _payment_postings(len(payments), keys_of)

    def key_ids(self, keys) -> np.ndarray:
        return self.vocab.lookup(np.asarray(list(keys), dtype=object))


def name_terms(tokens: list[str] | np.ndarray, min_length: int) -> list[str]:
    """Termes indexés d'un nom ou d'un libellé : mots assez longs + bigrammes de mots adjacents."""
    words = [t for t in tokens if len(t) >= min_length and t.isalpha()]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def terms_by_owner(texts: pd.Series, min_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Version vectorisée de `name_terms` sur une colonne de textes normalisés : (propriétaire, terme)."""
    tokens = pd.Series(texts.to_numpy(), dtype=object).fillna("").astype(str).str.split().explode().dropna()
    words = tokens.astype(str)
    keep = (words.str.len() >= min_length).to_numpy() & words.str.isalpha().to_numpy()
    owner = tokens.index.to_numpy(dtype=np.int64)[keep]
    word = words.to_numpy(dtype=object)[keep]
    same = owner[1:] == owner[:-1]
    bigram = (word[:-1] + " " + word[1:])[same] if len(word) > 1 else np.array([], dtype=object)
    return np.concatenate([owner, owner[:-1][same]]), np.concatenate([word, bigram])


class NameIndex:
    """Terme → débiteurs ; débiteur → termes ; paiement → termes de son libellé."""

    def __init__(self, debtors: pd.DataFrame, payments: pd.DataFrame, min_length: int):
        # Toutes les variantes de nom d'un débiteur (lignes multiples en source), sinon son nom.
        variants = debtors["name_variants"] if "name_variants" in debtors.columns \
            else debtors["name_norm"].map(lambda n: [n])
        exploded = pd.Series([v if len(v) else [""] for v in variants]).explode()
        owners, terms = terms_by_owner(exploded.reset_index(drop=True), min_length)
        owners = exploded.index.to_numpy(dtype=np.int64)[owners]
        self.vocab = Vocabulary(terms)
        self.term_debtors = Postings.build(self.vocab.codes, owners, len(self.vocab))
        self.debtor_terms = Postings.build(owners, self.vocab.codes, len(debtors))
        # Termes du libellé, restreints au vocabulaire des noms.
        labels = payments["label_norm"]

        def terms_of(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
            o, t = terms_by_owner(labels.iloc[start:end].reset_index(drop=True), min_length)
            return o, self.vocab.lookup(t)

        self.payment_terms = _payment_postings(len(payments), terms_of)


def party_ibans(debtors: pd.DataFrame, assignors: pd.DataFrame, party_iban: pd.DataFrame | None) -> pd.DataFrame:
    """Couples (rôle, partie, IBAN, bankroll) : table `party_iban` du chargement, sinon colonnes des parties."""
    if party_iban is not None:
        return party_iban
    parts = [df.loc[df["iban"].notna(), ["party_id", "iban", "bankroll_code"]].assign(role=role)
             for role, df in (("assignor", assignors), ("debtor", debtors))]
    return pd.concat(parts, ignore_index=True)


class IbanIndex:
    """IBAN → débiteurs, cédants (tous leurs comptes) ; bankroll par (IBAN, partie) ; comptes techniques."""

    def __init__(self, debtors: pd.DataFrame, assignors: pd.DataFrame, technical: pd.DataFrame | None,
                 party_iban: pd.DataFrame | None = None):
        links = party_ibans(debtors, assignors, party_iban)
        tech = technical["iban"].dropna() if technical is not None else pd.Series([], dtype=object)
        self.vocab = Vocabulary(pd.concat([links["iban"], tech], ignore_index=True).astype(object).to_numpy())
        self.bankroll: dict[str, pd.DataFrame] = {}
        for role, parties, attr in (("debtor", debtors, "debtors"), ("assignor", assignors, "assignors")):
            rows = links[links["role"] == role]
            pos = pd.Index(parties["party_id"].astype(object)).get_indexer(rows["party_id"].astype(object))
            ids = self.vocab.lookup(rows["iban"].astype(object).to_numpy())
            setattr(self, attr, Postings.build(ids, pos.astype(np.int64), len(self.vocab)))
            self.bankroll[role] = pd.DataFrame({"ib": ids, "pos": pos, "br": rows["bankroll_code"].to_numpy()}) \
                .query("ib >= 0 and pos >= 0").drop_duplicates()
        self.technical = np.zeros(len(self.vocab), dtype=bool)
        self.technical[self.vocab.lookup(tech.astype(object).to_numpy())] = True


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/allocation/allocator.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Allocation des paiements aux débiteurs (brief §5).

Pour chaque paiement du lot, une liste classée de débiteurs candidats avec leur
score et les signaux qui les ont produits, et le client file rattaché.

Signaux, du plus fort au plus faible :
1. client file rattaché sans ambiguïté → débiteurs des factures qu'il cite ;
2. référence de facture trouvée dans le libellé → débiteur de la facture ;
3. IBAN, routé DEBTOR_DIRECT / ASSIGNOR / TECHNICAL_ACCOUNT / UNKNOWN ;
4. nom du débiteur retrouvé dans le libellé (index inversé, mots rares pondérés) ;
5. montant : facture ouverte à D de restant dû exactement égal au paiement
   (extension du brief, clé K3 de la spec ; désactivable).

Les scores des signaux se combinent en « ou » probabiliste : 1 − Π(1 − sᵢ).
L'allocation est **ferme** si un seul débiteur porte un signal fort (client
file univoque, référence désignant un seul débiteur, IBAN direct unique).

L'allocateur garde un état propre au moteur : les client files déjà rattachés.
Il se crée au début d'un rejeu et ne se partage pas entre deux rejeux.
"""





CLIENT_FILE, REFERENCE, IBAN, NAME, AMOUNT = "client_file", "reference", "iban", "name", "amount"
SIGNALS = (CLIENT_FILE, REFERENCE, IBAN, NAME, AMOUNT)
DEBTOR_DIRECT, ASSIGNOR, TECHNICAL_ACCOUNT, UNKNOWN = "DEBTOR_DIRECT", "ASSIGNOR", "TECHNICAL_ACCOUNT", "UNKNOWN"
FIRM, MULTIPLE, NONE = "ferme", "multiple", "aucun"

# Poids des signaux (score maximal qu'un signal seul peut donner). Valeurs initiales, à calibrer.
WEIGHTS = {CLIENT_FILE: 1.0, REFERENCE: 0.95, IBAN: 0.9, NAME: 0.7, AMOUNT: 0.5}

# Taille des blocs de paiements traités ensemble : borne la mémoire des jointures d'une journée.
CHUNK_ROWS = 20_000

_SIGNAL_COLUMNS = ["row", "debtor", "signal", "score", "strong"]
_SIGNAL_CODE = {s: i for i, s in enumerate(SIGNALS)}
_SIGNAL_BIT = {s: 1 << i for i, s in enumerate(SIGNALS)}
_SIGNAL_LABEL = {m: "+".join(s for s in SIGNALS if m & _SIGNAL_BIT[s]) for m in range(1 << len(SIGNALS))}


@dataclass
class Allocation:
    candidates: pd.DataFrame   # payment_id, rank, debtor_id, score, signal, signals, strong
    payments: pd.DataFrame     # payment_id, status, firm_debtor_id, n_candidates, iban_route, client_file_id


def _empty_signal() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                         zip(_SIGNAL_COLUMNS, ["int64", "int64", "object", "float64", "bool"])})


class Allocator:
    def __init__(self, state: LedgerState, settings: AllocationSettings):
        self.state = state
        self.cfg = settings
        sig = settings.signals
        inv, pay = state.table("invoice"), state.table("payment")
        deb, asg = state.table("debtor"), state.table("assignor")
        self._debtor_ids = deb["party_id"].astype(object).to_numpy()
        self._inv_debtor = state.invoice_debtor_positions
        self._pay_value_day = _days(pay["value_date"])
        self._pay_amount = pay["amount"].fillna(-1).to_numpy(dtype=np.int64)
        self._pay_label = pay["label_norm"].fillna("").astype(object).to_numpy()

        self.ref = ReferenceIndex(inv, pay) if (sig.reference.enabled or sig.client_file.enabled) else None
        self.names = NameIndex(deb, pay, sig.name.min_token_length) if sig.name.enabled else None
        self.iban = None
        if sig.iban.enabled or sig.client_file.enabled:
            self.iban = IbanIndex(deb, asg, state.table("technical_account"), state.table("party_iban"))
            self._pay_iban = self.iban.vocab.lookup(pay["iban_debtor"].astype(object).to_numpy())
            self._pay_bankroll = pay["bankroll_code"].astype(object).to_numpy()

        self._cf = state.table("client_file")
        if sig.client_file.enabled and self._cf is not None and self.ref is not None:
            cf = self._cf
            self._cf_ids = cf["file_id"].astype(object).to_numpy()
            self._cf_amount = cf["total_amount"].fillna(-1).to_numpy(dtype=np.int64)
            self._cf_day = _days(cf["payment_date"])
            self._cf_iban = self.iban.vocab.lookup(cf["iban"].astype(object).to_numpy()) if self.iban else None
            self._cf_ref = cf["payment_reference_norm"].fillna("").astype(object).to_numpy()
            lines = state.table("client_file_line")
            file_of_line = pd.Index(cf["file_id"].astype(object)).get_indexer(
                lines["file_id"].astype(object)).astype(np.int64)
            lengths, flat = _flatten(lines["invoice_reference_keys"])
            owners = np.repeat(file_of_line, lengths)
            self._file_keys = Postings.build(owners, self.ref.vocab.lookup(flat), len(cf))
        else:
            self._cf = None
        self._attached: dict[int, int] = {}       # paiement → client file (positions)
        self._consumed: set[int] = set()
        self._amounts_as_of = None                 # (as_of, montants, débiteurs) du jour courant

    # --- Signaux -------------------------------------------------------------------------------

    def _reference(self, pos: np.ndarray, as_of) -> pd.DataFrame:
        cfg = self.cfg.signals.reference
        row, keys = self.ref.payment_keys.gather(pos)
        keep = self.ref.vocab.lengths[keys] >= cfg.min_key_length
        row, keys = row[keep], keys[keep]
        return self._debtors_of_keys(row, keys, as_of, REFERENCE, WEIGHTS[REFERENCE], cfg.max_debtors_per_key,
                                     cfg.strong_min_key_length, self._pay_amount[pos])

    def _debtors_of_keys(self, row: np.ndarray, keys: np.ndarray, as_of, signal: str, weight: float,
                         max_debtors: int, strong_min_length: int,
                         row_amount: np.ndarray | None = None) -> pd.DataFrame:
        """(ligne, clé) → débiteurs des factures **ouvertes** à D portant la clé.

        Si `row_amount` est fourni et qu'une de ces factures a un restant dû égal au montant du
        paiement, seules celles-là sont retenues. Spécificité = 1 / nombre de débiteurs restants,
        calculée par (paiement, clé) : une référence partagée par plusieurs cédants reste
        exploitable quand une seule facture ouverte correspond.
        """
        if len(keys) == 0:
            return _empty_signal()
        ukeys, k_of_row = np.unique(keys, return_inverse=True)
        k_owner, inv = self.ref.key_invoices.gather(ukeys)
        balance = self.state.open_balance_at(inv, as_of)
        deb = self._inv_debtor[inv]
        ok = (balance > 0) & (deb >= 0)
        k_owner, deb, balance = k_owner[ok], deb[ok], balance[ok]
        asked = pd.DataFrame({"row": row, "k": k_of_row}).drop_duplicates()
        # Jointures sur des tables dédoublonnées, jamais (ligne × toutes les factures de la clé) : une clé
        # fréquente (numéro de commande générique, année) chez un gros débiteur porte des milliers de
        # factures. Clé → débiteurs distincts ; clés partagées par trop de débiteurs écartées d'emblée.
        key_debtors = pd.DataFrame({"k": k_owner, "debtor": deb}).drop_duplicates()
        n_key = np.bincount(key_debtors["k"].to_numpy(), minlength=len(ukeys))
        exact = pd.DataFrame({c: pd.Series(dtype="int64") for c in ("row", "k", "debtor")})
        if row_amount is not None:
            # Factures de la clé dont le restant dû égale le montant du paiement : elles seules comptent.
            by_amount = pd.DataFrame({"k": k_owner, "amount": balance, "debtor": deb}).drop_duplicates()
            exact = (asked.assign(amount=row_amount[asked["row"].to_numpy()])
                     .merge(by_amount, on=["k", "amount"])[["row", "k", "debtor"]])
            if len(exact):
                code = asked["row"].to_numpy() * len(ukeys) + asked["k"].to_numpy()
                matched = np.unique(exact["row"].to_numpy() * len(ukeys) + exact["k"].to_numpy())
                asked = asked[~np.isin(code, matched)]
        light = key_debtors[n_key[key_debtors["k"].to_numpy()] <= max_debtors]
        pairs = pd.concat([exact, asked.merge(light, on="k")], ignore_index=True)
        if pairs.empty:
            return _empty_signal()
        n_deb = pairs.groupby(["row", "k"])["debtor"].transform("size").to_numpy()
        pairs = pairs[n_deb <= max_debtors]
        n_deb = n_deb[n_deb <= max_debtors]
        k = pairs["k"].to_numpy()
        pairs = pairs.assign(signal=signal, score=weight / n_deb,
                             strong=(n_deb == 1) & (self.ref.vocab.lengths[ukeys[k]] >= strong_min_length))
        return (pairs.groupby(["row", "debtor", "signal"], as_index=False)
                .agg(score=("score", "max"), strong=("strong", "any")))

    def _iban(self, pos: np.ndarray, as_of) -> tuple[pd.DataFrame, np.ndarray]:
        n = len(pos)
        ib = self._pay_iban[pos]
        route = np.full(n, UNKNOWN, dtype=object)
        rows = np.flatnonzero(ib >= 0)
        if len(rows) == 0:
            return _empty_signal(), route
        tech = self.iban.technical[ib[rows]]
        d_owner, d_pos = self.iban.debtors.gather(ib[rows])
        known = self.state.party_known_at("debtor", d_pos, as_of)
        d_owner, d_pos = d_owner[known], d_pos[known]
        a_owner, a_pos = self.iban.assignors.gather(ib[rows])
        known = self.state.party_known_at("assignor", a_pos, as_of)
        a_owner, a_pos = a_owner[known], a_pos[known]
        n_d = np.bincount(d_owner, minlength=len(rows))
        n_a = np.bincount(a_owner, minlength=len(rows))

        # IBAN à la fois débiteur et cédant : arbitrage par bankroll_code du paiement s'il est connu.
        conflict = (n_d > 0) & (n_a > 0)
        to_assignor = np.zeros(len(rows), dtype=bool)
        if conflict.any():
            pay_br = self._pay_bankroll[pos[rows]]
            ib_rows = ib[rows]
            a_match = pd.DataFrame({"o": a_owner, "ib": ib_rows[a_owner], "pos": a_pos}).merge(
                self.iban.bankroll["assignor"], on=["ib", "pos"])
            d_match = pd.DataFrame({"o": d_owner, "ib": ib_rows[d_owner], "pos": d_pos}).merge(
                self.iban.bankroll["debtor"], on=["ib", "pos"])
            a_hit = np.zeros(len(rows), dtype=bool)
            d_hit = np.zeros(len(rows), dtype=bool)
            for frame, hit in ((a_match, a_hit), (d_match, d_hit)):
                eq = frame["br"].to_numpy() == pay_br[frame["o"].to_numpy()]
                hit[frame["o"].to_numpy()[eq & pd.notna(frame["br"]).to_numpy()]] = True
            to_assignor = conflict & a_hit & ~d_hit
        sub = np.select([tech, to_assignor, n_d > 0, n_a > 0], [TECHNICAL_ACCOUNT, ASSIGNOR, DEBTOR_DIRECT, ASSIGNOR],
                        default=UNKNOWN)
        route[rows] = sub
        direct = (sub == DEBTOR_DIRECT)[d_owner]
        o, d = d_owner[direct], d_pos[direct]
        signal = pd.DataFrame({"row": rows[o], "debtor": d, "signal": IBAN,
                               "score": WEIGHTS[IBAN] / n_d[o], "strong": (n_d[o] == 1) & ~conflict[o]})
        return signal, route

    def _name(self, pos: np.ndarray, as_of) -> pd.DataFrame:
        cfg = self.cfg.signals.name
        idx = self.names
        row, terms = idx.payment_terms.gather(pos)
        if len(terms) == 0:
            return _empty_signal()
        n_known = max(self.state.known_party_count("debtor", as_of), 1)
        limit = min(cfg.max_token_share * n_known, cfg.max_debtors_per_term)

        def df_of(term_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Débiteurs connus à D par terme et fréquence documentaire à D."""
            owner, deb = idx.term_debtors.gather(term_ids)
            known = self.state.party_known_at("debtor", deb, as_of)
            owner, deb = owner[known], deb[known]
            return owner, deb, np.bincount(owner, minlength=len(term_ids))

        uterms, t_of_row = np.unique(terms, return_inverse=True)
        t_owner, t_deb, df = df_of(uterms)
        usable = (df > 0) & (df <= limit)
        idf = np.where(usable, np.log1p(n_known / np.maximum(df, 1)), 0.0)
        keep = usable[t_owner]
        postings = pd.DataFrame({"t": t_owner[keep], "debtor": t_deb[keep]})
        matched = (pd.DataFrame({"row": row, "t": t_of_row}).drop_duplicates()
                   .merge(postings, on="t"))
        if matched.empty:
            return _empty_signal()
        matched["idf"] = idf[matched["t"].to_numpy()]
        matched["spec"] = 1.0 / np.maximum(df[matched["t"].to_numpy()], 1)
        score = matched.groupby(["row", "debtor"], as_index=False).agg(idf=("idf", "sum"), spec=("spec", "max"))

        # Dénominateur : poids de tous les termes (utilisables à D) du nom de chaque débiteur candidat.
        cand = np.unique(score["debtor"].to_numpy())
        c_owner, c_terms = idx.debtor_terms.gather(cand)
        ut, t_inv = np.unique(c_terms, return_inverse=True)
        _, _, df_c = df_of(ut)
        w = np.where((df_c > 0) & (df_c <= limit), np.log1p(n_known / np.maximum(df_c, 1)), 0.0)
        total = np.bincount(c_owner, weights=w[t_inv], minlength=len(cand))
        score["total"] = total[np.searchsorted(cand, score["debtor"].to_numpy())]
        score = score[score["total"] > 0]
        # Couverture du nom, ou spécificité du meilleur mot retrouvé : un mot porté par un seul débiteur
        # connu à D (patronyme, marque) désigne ce débiteur même si le libellé tronque le nom.
        coverage = (score["idf"] / score["total"]).clip(upper=1.0).to_numpy()
        similarity = np.maximum(coverage, score["spec"].to_numpy())
        keep = similarity >= cfg.min_similarity
        strong_names = pd.DataFrame({"row": score["row"].to_numpy()[keep], "debtor": score["debtor"].to_numpy()[keep],
                                     "signal": NAME, "score": WEIGHTS[NAME] * similarity[keep], "strong": False})
        weak = score[~keep][["row", "debtor"]]
        if self.cfg.signals.amount.enabled and len(weak):
            return pd.concat([strong_names, self._weak_name_near_amount(pos, weak, as_of)], ignore_index=True)
        return strong_names

    def _weak_name_near_amount(self, pos: np.ndarray, weak: pd.DataFrame, as_of) -> pd.DataFrame:
        """Nom peu spécifique (homonymes) corroboré par une facture ouverte de restant dû proche du paiement.

        Tolérance d'escompte ou de frais : 5 € ou 3 %. Seuls les débiteurs retenus par le nom sont
        examinés (pas de recherche sur tous les débiteurs).
        """
        debtors, d_of = np.unique(weak["debtor"].to_numpy(), return_inverse=True)
        owner, _, balance = self.state.debtor_open_invoices_at(debtors, as_of)
        if len(owner) == 0:
            return _empty_signal()
        # Restants dus triés par (débiteur, montant) : pour chaque couple (paiement, débiteur), existe-t-il
        # un restant dû dans [montant − tolérance, montant + tolérance] ? Deux recherches dichotomiques.
        top = np.int64(1) << 40
        sorted_keys = np.sort(owner.astype(np.int64) * top + np.clip(balance, 0, top - 1))
        amount = self._pay_amount[pos[weak["row"].to_numpy()]]
        tol = np.floor(np.maximum(500, 0.03 * amount)).astype(np.int64)   # |restant − montant| ≤ tol, en entiers
        base = d_of.astype(np.int64) * top
        lo = np.searchsorted(sorted_keys, base + np.clip(amount - tol, 0, top - 1), side="left")
        hi = np.searchsorted(sorted_keys, base + np.clip(amount + tol, 0, top - 1), side="right")
        pairs = weak[hi > lo]
        if pairs.empty:
            return _empty_signal()
        n = pairs.groupby("row")["debtor"].transform("size").to_numpy()
        return pd.DataFrame({"row": pairs["row"].to_numpy(), "debtor": pairs["debtor"].to_numpy(), "signal": NAME,
                             "score": WEIGHTS[NAME] * 0.8 / n, "strong": False})

    def _amount(self, pos: np.ndarray, as_of) -> pd.DataFrame:
        """Débiteurs ayant une facture ouverte de restant dû égal au paiement.

        Montant partagé par peu de débiteurs : tous retenus. Par davantage (jusqu'à
        `max_debtors_with_name_hint`) : seuls ceux dont un mot du nom figure dans le libellé.
        """
        cfg = self.cfg.signals.amount
        amounts, debtors = self._open_amounts(as_of)
        if len(amounts) == 0:
            return _empty_signal()
        # Couples (montant, débiteur) distincts triés par montant : les débiteurs d'un montant forment une
        # tranche ; un montant rond partagé par trop de débiteurs est écarté sans être développé.
        pay_amount = self._pay_amount[pos]
        lo = np.searchsorted(amounts, pay_amount, side="left")
        n_per_row = np.searchsorted(amounts, pay_amount, side="right") - lo
        limit = max(cfg.max_debtors_per_amount, cfg.max_debtors_with_name_hint if self.names is not None else 0)
        n_per_row[(pay_amount <= 0) | (n_per_row > limit)] = 0
        total = int(n_per_row.sum())
        if total == 0:
            return _empty_signal()
        rows = np.repeat(np.arange(len(pos)), n_per_row)
        starts = np.repeat(lo - (np.cumsum(n_per_row) - n_per_row), n_per_row)
        pairs = pd.DataFrame({"row": rows, "debtor": debtors[starts + np.arange(total)]})
        n_all = n_per_row[rows]
        hint = self._name_hint(pos, pairs["row"].to_numpy(), pairs["debtor"].to_numpy())             if self.names is not None else np.zeros(len(pairs), dtype=bool)
        keep = (n_all <= cfg.max_debtors_per_amount) | hint
        pairs, hint, n_all = pairs[keep], hint[keep], n_all[keep]
        n_hint = pd.Series(hint).groupby(pairs["row"].to_numpy()).transform("sum").to_numpy()
        n = np.where(hint & (n_all > cfg.max_debtors_per_amount), n_hint, n_all)
        return pd.DataFrame({"row": pairs["row"].to_numpy(), "debtor": pairs["debtor"].to_numpy(), "signal": AMOUNT,
                             "score": WEIGHTS[AMOUNT] / np.maximum(n, 1), "strong": False})

    def _open_amounts(self, as_of) -> tuple[np.ndarray, np.ndarray]:
        """Couples (restant dû, débiteur) distincts des factures ouvertes à D, triés ; calculés une fois par jour."""
        if self._amounts_as_of is None or self._amounts_as_of[0] != as_of:
            inv, balance = self.state.open_invoice_positions(as_of)
            deb = self._inv_debtor[inv]
            ok = deb >= 0
            amount, deb = balance[ok].astype(np.int64), deb[ok].astype(np.int64)
            order = np.lexsort((deb, amount))
            amount, deb = amount[order], deb[order]
            keep = np.ones(len(amount), dtype=bool)
            keep[1:] = (amount[1:] != amount[:-1]) | (deb[1:] != deb[:-1])
            self._amounts_as_of = (as_of, amount[keep], deb[keep])
        return self._amounts_as_of[1], self._amounts_as_of[2]

    def _name_hint(self, pos: np.ndarray, rows: np.ndarray, debtors: np.ndarray) -> np.ndarray:
        """Pour chaque couple (ligne, débiteur) : un terme du nom du débiteur figure-t-il dans le libellé ?"""
        if len(rows) == 0:
            return np.zeros(0, dtype=bool)
        pairs = pd.DataFrame({"i": np.arange(len(rows)), "row": rows, "debtor": debtors})
        p_row, p_term = self.names.payment_terms.gather(pos)
        pay_terms = pd.DataFrame({"row": p_row, "term": p_term})
        udeb, d_of = np.unique(debtors, return_inverse=True)
        d_owner, d_term = self.names.debtor_terms.gather(udeb)
        deb_terms = pd.DataFrame({"debtor": udeb[d_owner], "term": d_term})
        hits = pairs.merge(pay_terms, on="row").merge(deb_terms, on=["debtor", "term"])
        out = np.zeros(len(rows), dtype=bool)
        out[hits["i"].to_numpy()] = True
        return out

    def _client_file(self, pos: np.ndarray, as_of) -> tuple[pd.DataFrame, np.ndarray]:
        """Rattache les client files reçus aux paiements, puis en déduit les débiteurs cités."""
        cfg = self.cfg.signals.client_file
        attached = np.array([self._attached.get(int(p), -1) for p in pos], dtype=np.int64)
        free = np.flatnonzero(attached < 0)
        received = self.state.client_files_received_at(as_of)
        available = received[~np.isin(received, np.fromiter(self._consumed, np.int64, len(self._consumed)))]
        if len(free) and len(available):
            files = pd.DataFrame({"f": available, "amount": self._cf_amount[available]})
            pays = pd.DataFrame({"r": free, "amount": self._pay_amount[pos[free]]})
            pairs = pays.merge(files[files["amount"] >= 0], on="amount")
            if len(pairs):
                r, f = pairs["r"].to_numpy(), pairs["f"].to_numpy()
                p = pos[r]
                date_ok = np.abs(self._cf_day[f] - self._pay_value_day[p]) <= cfg.date_tolerance_days
                iban_ok = (self._cf_iban[f] >= 0) & (self._cf_iban[f] == self._pay_iban[p]) \
                    if self._cf_iban is not None else np.zeros(len(f), bool)
                ref_ok = np.fromiter((bool(c) and c in lbl for c, lbl in zip(self._cf_ref[f], self._pay_label[p])),
                                     dtype=bool, count=len(f))
                pairs = pairs[date_ok | iban_ok | ref_ok]
                # Sans ambiguïté : un seul fichier pour le paiement, un seul paiement pour le fichier.
                pairs = pairs[~pairs.duplicated("r", keep=False) & ~pairs.duplicated("f", keep=False)]
                for r_i, f_i in zip(pairs["r"].to_numpy(), pairs["f"].to_numpy()):
                    self._attached[int(pos[r_i])] = int(f_i)
                    self._consumed.add(int(f_i))
                    attached[r_i] = f_i
        rows = np.flatnonzero(attached >= 0)
        if len(rows) == 0:
            return _empty_signal(), attached
        f_owner, keys = self._file_keys.gather(attached[rows])
        cfg_ref = self.cfg.signals.reference
        keep = self.ref.vocab.lengths[keys] >= cfg_ref.min_key_length
        by_key = self._debtors_of_keys(rows[f_owner[keep]], keys[keep], as_of, CLIENT_FILE, 1.0,
                                       cfg_ref.max_debtors_per_key, 0)
        if by_key.empty:
            return by_key, attached
        # Univoque si toutes les lignes du fichier désignent le même débiteur.
        n = by_key.groupby("row")["debtor"].transform("nunique").to_numpy()
        by_key = by_key.assign(score=WEIGHTS[CLIENT_FILE] / n, strong=n == 1)
        return by_key, attached

    # --- Allocation du lot -------------------------------------------------------------------------

    def allocate(self, ctx: DayContext) -> Allocation:
        """Allocation du lot. Le rattachement des client files se fait sur tout le lot (unicité paiement ↔
        fichier) ; les autres signaux, propres à chaque paiement, par blocs de `CHUNK_ROWS` lignes, ce qui
        borne la mémoire quelle que soit la taille du reliquat."""
        as_of = ctx.as_of
        batch_ids = ctx.batch["payment_id"].astype(object).to_numpy()
        pos = self.state.pay_pos(pd.Series(batch_ids, dtype=object))
        attached = np.full(len(pos), -1, dtype=np.int64)
        cf_signal = _empty_signal()
        if self.cfg.signals.client_file.enabled and self._cf is not None:
            memory.mark("allocation · client files")
            cf_signal, attached = self._client_file(pos, as_of)
        cf_row = cf_signal["row"].to_numpy()
        chunks = []
        n_blocks = max(-(-len(pos) // CHUNK_ROWS), 1)
        for start in range(0, max(len(pos), 1), CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, len(pos))
            self._block = f"bloc {start // CHUNK_ROWS + 1}/{n_blocks}"
            in_chunk = (cf_row >= start) & (cf_row < end)
            cf_part = cf_signal[in_chunk].assign(row=cf_row[in_chunk] - start)
            chunks.append(self._allocate_rows(pos[start:end], batch_ids[start:end], attached[start:end],
                                              cf_part, as_of))
        if len(chunks) == 1:
            return chunks[0]
        # infer_objects : mêmes types qu'en un seul bloc (un bloc sans aucun débiteur ferme reste en `object`).
        return Allocation(pd.concat([c.candidates for c in chunks], ignore_index=True).infer_objects(),
                          pd.concat([c.payments for c in chunks], ignore_index=True).infer_objects())

    def _allocate_rows(self, pos: np.ndarray, batch_ids: np.ndarray, attached: np.ndarray,
                       cf_signal: pd.DataFrame, as_of) -> Allocation:
        sig = self.cfg.signals
        parts = [cf_signal]
        route = np.full(len(pos), UNKNOWN, dtype=object)
        block = getattr(self, "_block", "")
        if len(pos):
            if sig.reference.enabled:
                memory.mark(f"allocation · référence · {block}")
                parts.append(self._reference(pos, as_of))
            if sig.iban.enabled:
                memory.mark(f"allocation · IBAN · {block}")
                iban_signal, route = self._iban(pos, as_of)
                parts.append(iban_signal)
            if sig.name.enabled:
                memory.mark(f"allocation · nom · {block}")
                parts.append(self._name(pos, as_of))
            if sig.amount.enabled:
                memory.mark(f"allocation · montant · {block}")
                parts.append(self._amount(pos, as_of))
        memory.mark(f"allocation · combinaison · {block}")
        signals = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) \
            else _empty_signal()
        return self._combine(signals, batch_ids, route, attached)

    def _combine(self, s: pd.DataFrame, batch_ids: np.ndarray, route: np.ndarray, attached: np.ndarray) -> Allocation:
        n = len(batch_ids)
        cf_ids = np.full(n, None, dtype=object)
        if self._cf is not None:
            has = attached >= 0
            cf_ids[has] = self._cf_ids[attached[has]]
        if s.empty:
            cand = pd.DataFrame(columns=["payment_id", "rank", "debtor_id", "score", "signal", "signals", "strong"])
            status = np.full(n, NONE, dtype=object)
            firm = np.full(n, None, dtype=object)
            n_cand = np.zeros(n, dtype=np.int64)
        else:
            # Réductions par segments (row, débiteur) en numpy : pas de groupby pandas par jour.
            row = s["row"].to_numpy(dtype=np.int64)
            deb = s["debtor"].to_numpy(dtype=np.int64)
            score = s["score"].to_numpy(dtype=np.float64)
            code = s["signal"].map(_SIGNAL_CODE).to_numpy(dtype=np.int64)
            strong = s["strong"].to_numpy(dtype=bool)
            order = np.lexsort((-score, deb, row))
            row, deb, score, code, strong = row[order], deb[order], score[order], code[order], strong[order]
            starts = np.flatnonzero(np.r_[True, (row[1:] != row[:-1]) | (deb[1:] != deb[:-1])])
            c_row, c_deb = row[starts], deb[starts]
            c_score = 1.0 - np.multiply.reduceat(1.0 - score, starts)
            c_strong = np.logical_or.reduceat(strong, starts)
            c_bits = np.bitwise_or.reduceat(1 << code, starts)
            c_signal = code[starts]                          # signal au score le plus élevé
            # Classement : score décroissant, puis position du débiteur (déterministe). Score arrondi pour le
            # tri : deux scores égaux au bruit de sommation flottante près sont départagés par le débiteur.
            order = np.lexsort((c_deb, -np.round(c_score, 9), c_row))
            c_row, c_deb, c_score, c_strong, c_bits, c_signal = (
                a[order] for a in (c_row, c_deb, c_score, c_strong, c_bits, c_signal))
            first = np.flatnonzero(np.r_[True, c_row[1:] != c_row[:-1]])
            rank = np.arange(len(c_row)) - np.repeat(first, np.diff(np.r_[first, len(c_row)])) + 1
            n_strong = np.bincount(c_row[c_strong], minlength=n)
            firm_mask = c_strong & (n_strong[c_row] == 1)
            firm = np.full(n, None, dtype=object)
            firm[c_row[firm_mask]] = self._debtor_ids[c_deb[firm_mask]]
            keep = rank <= self.cfg.max_candidates
            n_cand = np.bincount(c_row[keep], minlength=n)
            status = np.where(pd.notna(firm), FIRM, np.where(n_cand > 0, MULTIPLE, NONE))
            cand = pd.DataFrame({
                "payment_id": batch_ids[c_row[keep]], "rank": rank[keep], "debtor_id": self._debtor_ids[c_deb[keep]],
                "score": c_score[keep], "signal": np.asarray(SIGNALS, dtype=object)[c_signal[keep]],
                "signals": pd.Series(c_bits[keep]).map(_SIGNAL_LABEL).to_numpy(), "strong": c_strong[keep],
            })
        payments = pd.DataFrame({"payment_id": batch_ids, "status": status, "firm_debtor_id": firm,
                                 "n_candidates": n_cand, "iban_route": route, "client_file_id": cf_ids})
        return Allocation(cand, payments)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/allocation/evaluate.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Mesure de l'allocation (brief §5, §8) : rappel, précision de l'allocation ferme.

La sonde `AllocationProbe` se branche dans la boucle quotidienne comme un
rapprocheur qui ne décide rien : elle enregistre l'allocation de chaque paiement
à son premier passage (jour d'arrivée) et à son dernier passage dans le lot.

Vérité : le débiteur des factures réellement imputées au paiement.
"""




# Le dernier passage de chaque paiement est dédoublonné tous les N jours : sans cela, les lots successifs
# (reliquat compris) s'accumulent en mémoire.
_COMPACT_EVERY = 5


class AllocationProbe:
    name = "allocation"

    def __init__(self, allocator: Allocator):
        self.allocator = allocator
        self._first: list[pd.DataFrame] = []
        self._last: list[pd.DataFrame] = []

    def process(self, ctx: DayContext) -> pd.DataFrame:
        alloc = self.allocator.allocate(ctx)
        summary = alloc.payments.copy()
        cand = alloc.candidates
        keys, debtors = _tuples_by_key(cand["payment_id"].to_numpy(dtype=object),
                                       cand["debtor_id"].to_numpy(dtype=object))
        _, signals = _tuples_by_key(cand["payment_id"].to_numpy(dtype=object), cand["signals"].to_numpy(dtype=object))
        lists = pd.DataFrame({"payment_id": keys, "candidates": debtors, "candidate_signals": signals})
        summary = summary.merge(lists, on="payment_id", how="left")
        empty = summary["candidates"].isna()
        summary.loc[empty, "candidates"] = pd.Series([()] * int(empty.sum()), index=summary.index[empty], dtype=object)
        summary.loc[empty, "candidate_signals"] = pd.Series([()] * int(empty.sum()), index=summary.index[empty],
                                                            dtype=object)
        summary["day"] = ctx.day
        self._first.append(summary[ctx.batch["is_new"].to_numpy()])
        self._last.append(summary)
        if len(self._last) >= _COMPACT_EVERY:
            self._last = [pd.concat(self._last, ignore_index=True).drop_duplicates("payment_id", keep="last")]
        return empty_decisions()

    def first_pass(self) -> pd.DataFrame:
        return pd.concat(self._first, ignore_index=True) if self._first else pd.DataFrame()

    def last_pass(self) -> pd.DataFrame:
        if not self._last:
            return pd.DataFrame()
        return pd.concat(self._last, ignore_index=True).drop_duplicates("payment_id", keep="last")


def truth_debtors(imputation: pd.DataFrame, invoice: pd.DataFrame) -> pd.DataFrame:
    """Par paiement imputé : débiteurs des factures imputées (tuple trié)."""
    debtor_of = invoice[["invoice_id", "debtor_id"]].drop_duplicates("invoice_id")
    pairs = (imputation[["payment_id", "invoice_id"]].merge(debtor_of, on="invoice_id")
             [["payment_id", "debtor_id"]].dropna().drop_duplicates().sort_values(["payment_id", "debtor_id"]))
    keys, tuples = _tuples_by_key(pairs["payment_id"].to_numpy(dtype=object), pairs["debtor_id"].to_numpy(dtype=object))
    return pd.DataFrame({"payment_id": keys, "truth_debtors": tuples})


def _found(candidates, truth) -> bool:
    return isinstance(truth, tuple) and bool(truth) and set(truth) <= set(candidates)


def allocation_metrics(first: pd.DataFrame, last: pd.DataFrame, truth: pd.DataFrame, target_recall: float,
                       payments: pd.DataFrame | None = None) -> dict:
    """Métriques sur les paiements arrivés dans la période (premier passage) ayant une imputation réelle."""
    p = first.merge(truth, on="payment_id", how="left")
    p = p.merge(last[["payment_id", "candidates", "status", "client_file_id"]]
                .rename(columns={"candidates": "candidates_last", "status": "status_last",
                                 "client_file_id": "client_file_last"}), on="payment_id", how="left")
    has_truth = p["truth_debtors"].map(lambda t: isinstance(t, tuple))
    p["found_first"] = [_found(c, t) for c, t in zip(p["candidates"], p["truth_debtors"])]
    p["found_last"] = [_found(c, t) for c, t in zip(p["candidates_last"], p["truth_debtors"])]
    single = p["truth_debtors"].map(lambda t: isinstance(t, tuple) and len(t) == 1)
    p["top1"] = [s and len(c) > 0 and c[0] == t[0] for s, c, t in zip(single, p["candidates"], p["truth_debtors"])]
    p["firm"] = p["status"] == FIRM
    p["firm_correct"] = [f and s and fd == t[0] for f, s, fd, t in
                         zip(p["firm"], single, p["firm_debtor_id"], p["truth_debtors"])]

    def found_by(c, sigs, t) -> str:
        if not _found(c, t) or len(t) != 1:
            return "non trouvé" if not _found(c, t) else "plusieurs débiteurs"
        return sigs[list(c).index(t[0])]

    p["found_by"] = [found_by(c, s, t) for c, s, t in zip(p["candidates"], p["candidate_signals"], p["truth_debtors"])]
    q = p[has_truth]
    n = len(q)
    firm = q["firm"].sum()
    recall = q["found_first"].mean() if n else float("nan")
    summary = {
        "paiements_périmètre": len(p),
        "avec_imputation_réelle": n,
        "rappel_premier_passage": recall,
        "rappel_dernier_passage": q["found_last"].mean() if n else float("nan"),
        "rappel_cible": target_recall,
        "cible_atteinte": bool(n and recall >= target_recall),
        "top1": q["top1"].mean() if n else float("nan"),
        "taux_ferme": firm / n if n else float("nan"),
        "précision_ferme": q["firm_correct"].sum() / firm if firm else float("nan"),
        "sans_candidat": (q["status"] == NONE).mean() if n else float("nan"),
        "candidats_moyens": q["candidates"].map(len).mean() if n else float("nan"),
        "avec_client_file": q["client_file_last"].notna().mean() if n else float("nan"),
    }

    def rate_table(key: str) -> pd.DataFrame:
        g = q.groupby(key, dropna=False)
        out = pd.DataFrame({"paiements": g.size(), "rappel": g["found_first"].mean(),
                            "rappel_dernier_passage": g["found_last"].mean(), "top1": g["top1"].mean(),
                            "taux_ferme": g["firm"].mean(),
                            "précision_ferme": g["firm_correct"].sum() / g["firm"].sum().where(g["firm"].sum() > 0)})
        out = out.reset_index()
        out["part"] = out["paiements"] / n if n else 0.0
        return out

    by_route = rate_table("iban_route")
    by_status = rate_table("status")
    found = q["found_by"].value_counts().rename_axis("signal").reset_index(name="paiements")
    found["part"] = found["paiements"] / n if n else 0.0
    misses = q[~q["found_first"]].head(200)
    misses = misses[["payment_id", "iban_route", "status", "candidates", "truth_debtors"]].copy()
    if payments is not None:
        misses = misses.merge(payments[["payment_id", "label"]], on="payment_id", how="left")
    for col in ("candidates", "truth_debtors"):
        misses[col] = misses[col].map(lambda v: " ".join(v[:5]) if isinstance(v, tuple) else "")
    return {"summary": summary, "by_route": by_route, "by_status": by_status, "found_by": found,
            "misses": misses, "detail": p}


def signal_order() -> list[str]:
    return list(SIGNALS)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/evaluation/metrics.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Évaluation des décisions d'un rejeu contre les imputations réellement prononcées (brief §8).

Unité de mesure : le paiement. Un paiement auto-validé est **correct** si
l'ensemble des factures décidées est exactement celui des factures réellement
imputées à ce paiement (toutes dates confondues). Un paiement sans imputation
réelle qui est auto-validé compte comme une erreur.

Périmètre : les paiements arrivés pendant la période évaluée. Les décisions sur
des paiements arrivés avant (reliquat de démarrage) sont ignorées.
"""





GROUP_TYPES = ("1↔1", "1↔n", "n↔1", "n↔n", "sans imputation")
_CURVE_POINTS = 200


@dataclass
class Evaluation:
    summary: dict
    by_step: pd.DataFrame
    by_rule: pd.DataFrame
    by_group: pd.DataFrame
    by_month: pd.DataFrame
    curve: pd.DataFrame
    cascade: pd.DataFrame
    payments: pd.DataFrame = field(repr=False)   # détail par paiement du périmètre

    def tables(self) -> dict[str, pd.DataFrame]:
        return {"by_step": self.by_step, "by_rule": self.by_rule, "by_group": self.by_group,
                "by_month": self.by_month, "curve": self.curve, "cascade": self.cascade}


def _tuples_by_key(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
    """Regroupe `values` par `keys` (déjà triés par clé) en tuples, sans groupby Python par groupe."""
    if len(keys) == 0:
        return keys, []
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    return keys[starts], [tuple(values[a:b]) for a, b in zip(starts, ends)]


def ground_truth(imputation: pd.DataFrame) -> pd.DataFrame:
    """Par paiement imputé : factures réellement imputées (tuple trié) et type de groupe.

    Le type se lit sur la composante connexe du graphe paiements–factures :
    1↔1, 1↔n (un paiement, plusieurs factures), n↔1, n↔n.
    """
    pairs = imputation[["payment_id", "invoice_id"]].dropna().drop_duplicates()
    pay_codes, pay_ids = pd.factorize(pairs["payment_id"])
    inv_codes, _ = pd.factorize(pairs["invoice_id"])
    n_pay, n_inv = len(pay_ids), inv_codes.max() + 1 if len(inv_codes) else 0
    graph = coo_matrix((np.ones(len(pairs)), (pay_codes, n_pay + inv_codes)), shape=(n_pay + n_inv,) * 2)
    _, component = connected_components(graph, directed=False)
    comp_pay = np.bincount(component[:n_pay], minlength=component.max() + 1 if len(component) else 0)
    comp_inv = np.bincount(component[n_pay:], minlength=len(comp_pay))
    k_pay, k_inv = comp_pay[component[:n_pay]], comp_inv[component[:n_pay]]
    group_type = np.select([(k_pay == 1) & (k_inv == 1), k_pay == 1, k_inv == 1], ["1↔1", "1↔n", "n↔1"],
                           default="n↔n")
    ordered = pairs.sort_values(["payment_id", "invoice_id"])
    keys, tuples = _tuples_by_key(ordered["payment_id"].to_numpy(dtype=object),
                                  ordered["invoice_id"].to_numpy(dtype=object))
    truth = pd.DataFrame({"payment_id": keys, "truth_invoices": tuples})
    truth["n_invoices"] = [len(t) for t in tuples]
    types = pd.Series(group_type, index=pd.Index(pay_ids.astype(object)))
    truth["group_type"] = types.reindex(truth["payment_id"]).to_numpy()
    return truth


def _proposals(decisions: pd.DataFrame) -> pd.DataFrame:
    """Une proposition par paiement : la première auto-validation, sinon la première mise en revue."""
    cols = ["payment_id", "action", "day", "invoices", "score", "step", "rule_id", "rule_version"]
    d = decisions[decisions["action"].isin([AUTO, REVIEW])]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d.assign(auto=d["action"] == AUTO).dropna(subset=["invoice_id"])
    d = d.drop_duplicates(["payment_id", "day", "auto", "invoice_id"])
    d = d.sort_values(["payment_id", "day", "auto", "invoice_id"]).reset_index(drop=True)
    group = d.groupby(["payment_id", "day", "auto"], sort=False)
    grouped = group.agg(score=("score", "max"), step=("step", "first"), rule_id=("rule_id", "first"),
                        rule_version=("rule_version", "first")).reset_index()
    _, grouped["invoices"] = _tuples_by_key(group.ngroup().to_numpy(), d["invoice_id"].to_numpy(dtype=object))
    # Priorité à l'auto-validation (la plus précoce), sinon la revue la plus précoce.
    grouped = grouped.sort_values(["payment_id", "auto", "day"], ascending=[True, False, True])
    first = grouped.drop_duplicates("payment_id").copy()
    first["action"] = np.where(first["auto"], AUTO, REVIEW)
    return first[cols].reset_index(drop=True)


def _rate_table(df: pd.DataFrame, key: str, in_scope: int | None = None) -> pd.DataFrame:
    """Par valeur de `key` : paiements, auto-validés, corrects, précision, taux d'automatisation."""
    g = df.groupby(key, dropna=False, sort=True)
    out = pd.DataFrame({
        "paiements": g.size(),
        "auto": g["is_auto"].sum(),
        "corrects": g["correct"].sum(),
    }).reset_index()
    out["précision"] = out["corrects"] / out["auto"].where(out["auto"] > 0)
    denominator = in_scope if in_scope is not None else out["paiements"]
    out["taux_automatisation"] = out["auto"] / denominator
    return out


def precision_curve(scores: np.ndarray, correct: np.ndarray, in_scope: int, base_n: int = 0,
                    base_correct: int = 0) -> pd.DataFrame:
    """Courbe automatisation / précision en auto-validant les propositions par score décroissant.

    `base_n` / `base_correct` : décisions acquises quel que soit le seuil (règles de l'étape 4).
    """
    if in_scope == 0 or (len(scores) == 0 and base_n == 0):
        return pd.DataFrame(columns=["seuil", "taux_automatisation", "précision"])
    order = np.argsort(-scores, kind="stable")
    s, c = scores[order], correct[order].astype(np.float64)
    n = base_n + np.arange(1, len(s) + 1)
    prec = (base_correct + np.cumsum(c)) / n
    # Un seuil ne peut couper qu'entre deux scores différents.
    last_of_score = np.r_[s[1:] != s[:-1], True] if len(s) else np.array([], dtype=bool)
    curve = pd.DataFrame({"seuil": s, "taux_automatisation": n / in_scope, "précision": prec})[last_of_score]
    if base_n:
        start = pd.DataFrame({"seuil": [np.inf], "taux_automatisation": [base_n / in_scope],
                              "précision": [base_correct / base_n]})
        curve = pd.concat([start, curve], ignore_index=True)
    if len(curve) > _CURVE_POINTS:
        idx = np.unique(np.linspace(0, len(curve) - 1, _CURVE_POINTS).round().astype(int))
        curve = curve.iloc[idx]
    return curve.reset_index(drop=True)


def automation_at_precision(curve: pd.DataFrame, target: float) -> float:
    ok = curve[curve["précision"] >= target]
    return float(ok["taux_automatisation"].max()) if len(ok) else 0.0


def evaluate(decisions: pd.DataFrame, truth: pd.DataFrame, scope: pd.DataFrame, target_precision: float,
             current_automation_rate: float | None = None) -> Evaluation:
    """`scope` : paiements du périmètre (colonnes payment_id, arrival_day)."""
    props = _proposals(decisions)
    p = (scope[["payment_id", "arrival_day"]]
         .merge(truth, on="payment_id", how="left")
         .merge(props, on="payment_id", how="left"))
    p["group_type"] = p["group_type"].fillna("sans imputation")
    p["is_auto"] = p["action"] == AUTO
    p["is_review"] = p["action"] == REVIEW
    has_truth = p["truth_invoices"].notna()
    same = [isinstance(a, tuple) and isinstance(b, tuple) and a == b
            for a, b in zip(p["invoices"], p["truth_invoices"])]
    p["correct_proposal"] = has_truth & p["action"].notna() & pd.Series(same, index=p.index)
    p["correct"] = p["is_auto"] & p["correct_proposal"]
    p["month"] = pd.to_datetime(p["arrival_day"]).dt.strftime("%Y-%m")
    n = len(p)
    auto = int(p["is_auto"].sum())
    correct = int(p["correct"].sum())

    # Précision / rappel au niveau des paires (paiement, facture) auto-validées.
    decided = {(pid, i) for pid, inv, a in zip(p["payment_id"], p["invoices"], p["is_auto"]) if a for i in inv}
    real = {(pid, i) for pid, inv in zip(p["payment_id"], p["truth_invoices"]) if isinstance(inv, tuple) for i in inv}
    hits = len(decided & real)

    # Les décisions des règles sont acquises ; le seuil ne porte que sur les autres propositions.
    base = p["is_auto"] & (p["step"] == "rules")
    proposed = p[p["action"].notna() & ~base]
    curve = precision_curve(proposed["score"].fillna(0).to_numpy(dtype=np.float64),
                            proposed["correct_proposal"].to_numpy(), n, int(base.sum()),
                            int(p.loc[base, "correct"].sum()))
    at_target = automation_at_precision(curve, target_precision)
    rate = auto / n if n else 0.0
    precision = correct / auto if auto else float("nan")

    summary = {
        "paiements_périmètre": n,
        "avec_imputation_réelle": int(has_truth.sum()),
        "sans_imputation_réelle": int((~has_truth).sum()),
        "auto": auto, "auto_corrects": correct, "revue": int(p["is_review"].sum()),
        "taux_automatisation": rate,
        "précision": precision,
        "taux_automatisation_correct": correct / n if n else 0.0,
        "précision_cible": target_precision,
        "cible_atteinte": bool(auto == 0 or precision >= target_precision),
        "taux_automatisation_à_précision_cible": at_target,
        "précision_paires": hits / len(decided) if decided else float("nan"),
        "rappel_paires": hits / len(real) if real else float("nan"),
        "taux_actuel_référence": current_automation_rate,
    }

    rules_auto = p["is_auto"] & (p["step"] == "rules")
    rules_correct = rules_auto & p["correct"]
    cascade = pd.DataFrame([
        {"niveau": "Algorithme actuel", "taux_automatisation": current_automation_rate, "précision": None},
        {"niveau": "Étape 4 — règles seules", "taux_automatisation": rules_auto.sum() / n if n else 0.0,
         "précision": rules_correct.sum() / rules_auto.sum() if rules_auto.sum() else None},
        {"niveau": "Étapes 4 + 5 — règles puis ML", "taux_automatisation": rate,
         "précision": precision if auto else None},
    ])
    ref = current_automation_rate
    cascade["gain_points"] = [None] + [None if ref is None else 100 * (r - ref)
                                       for r in cascade["taux_automatisation"].iloc[1:]]

    auto_rows = p[p["is_auto"]]
    by_step = _rate_table(auto_rows.assign(step=auto_rows["step"].fillna("?")), "step", n) if auto else \
        pd.DataFrame(columns=["step", "paiements", "auto", "corrects", "précision", "taux_automatisation"])
    by_rule = _rate_table(auto_rows.assign(rule=auto_rows["rule_id"].fillna("—")), "rule", n) if auto else \
        pd.DataFrame(columns=["rule", "paiements", "auto", "corrects", "précision", "taux_automatisation"])
    by_group = _rate_table(p, "group_type")
    by_group["group_type"] = pd.Categorical(by_group["group_type"], categories=GROUP_TYPES, ordered=True)
    by_group = by_group.sort_values("group_type").reset_index(drop=True)
    by_group["group_type"] = by_group["group_type"].astype(str)
    by_month = _rate_table(p, "month")

    detail = p[["payment_id", "arrival_day", "group_type", "action", "day", "step", "rule_id", "score",
                "correct"]].copy()
    return Evaluation(summary, by_step, by_rule, by_group, by_month, curve, cascade, detail)


def ml_diagnostics(candidates: pd.DataFrame, truth: pd.DataFrame, scope: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Diagnostics du scoring sur les paiements passés en ML : rappel des candidats, precision@1, MRR, calibration.

    `candidates` : (payment_id, invoice_id, p) au premier passage en ML de chaque paiement.
    """
    c = candidates[candidates["payment_id"].isin(set(scope["payment_id"]))]
    t = truth.set_index("payment_id")["truth_invoices"]
    c = c[c["payment_id"].isin(t.index)]
    if c.empty:
        return {}, pd.DataFrame()
    pairs = {(pid, inv) for pid, invs in t.items() for inv in invs}
    c = c.assign(label=[(pid, inv) in pairs for pid, inv in zip(c["payment_id"], c["invoice_id"])])
    c = c.sort_values(["payment_id", "p"], ascending=[True, False], kind="mergesort")
    c["rank"] = c.groupby("payment_id").cumcount() + 1
    got = c.groupby("payment_id")["invoice_id"].agg(set)
    recall = np.mean([set(t[pid]) <= got[pid] for pid in got.index])
    first = c[c["label"]].groupby("payment_id")["rank"].min().reindex(got.index)
    edges = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(c["p"].to_numpy(), edges[1:-1]), 0, 9)
    calib = c.assign(bin=b).groupby("bin").agg(paires=("label", "size"), score_moyen=("p", "mean"),
                                               taux_observé=("label", "mean")).reset_index()
    calib["tranche"] = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in calib["bin"]]
    return {
        "paiements_en_ml": int(len(got)),
        "rappel_candidats": float(recall),
        "precision_at_1": float((first == 1).mean()),
        "mrr": float((1.0 / first).fillna(0).mean()),
    }, calib[["tranche", "paires", "score_moyen", "taux_observé"]]


def by_flag(evaluation_payments: pd.DataFrame, flag: pd.Series, name: str) -> pd.DataFrame:
    """Taux d'automatisation et précision selon un indicateur par paiement (ex. client file rattaché)."""
    p = evaluation_payments.assign(**{name: evaluation_payments["payment_id"].map(flag).fillna(False).astype(bool)})
    g = p.groupby(name)
    out = pd.DataFrame({"paiements": g.size(), "auto": g["action"].apply(lambda a: (a == AUTO).sum()),
                        "corrects": g["correct"].sum()}).reset_index()
    out["précision"] = out["corrects"] / out["auto"].where(out["auto"] > 0)
    out["taux_automatisation"] = out["auto"] / out["paiements"]
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/evaluation/report.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Rendu markdown d'une évaluation (rapport reproductible, brief §8)."""





def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.2%}" if value <= 1 else f"{value:.2f}"
    return str(value)


def render_markdown(summary: dict, tables: dict[str, pd.DataFrame], context: dict) -> str:
    lines = [f"# Évaluation — rapprocheur « {context['matcher']} », période {context['period']}", "",
             f"Du {context['start']} au {context['end']} · journal `{context['journal_sha256'][:16]}…`", "",
             "| indicateur | valeur |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in summary.items()]
    for name in ("cascade", "by_group", "by_step", "by_rule", "by_rule_alone", "by_client_file", "ml_calibration",
                 "by_month"):
        if name not in tables:
            continue
        df = tables[name]
        lines += ["", f"## {name}", ""]
        if df.empty:
            lines.append("_(vide)_")
            continue
        lines += ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/reconcile_rules/subset.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Recherche bornée de sous-ensembles à somme exacte (règle R4, ensembles de l'étape 5).

Parcours en profondeur sur les montants triés, avec élagage :
- la somme courante dépasse la cible → on coupe (montants croissants) ;
- même en ajoutant les plus grands montants restants, la cible est hors d'atteinte → on coupe ;
- profondeur > `max_size` → on coupe ;
- budget de nœuds épuisé → résultat « indéterminé ».

On s'arrête dès la deuxième solution : l'unicité est tout ce qui compte pour valider.
"""




UNIQUE, NONE, AMBIGUOUS, BUDGET = "unique", "none", "ambiguous", "budget"


@dataclass(frozen=True)
class SubsetResult:
    status: str
    indices: tuple[int, ...] = ()     # indices dans le tableau d'entrée (si unique)
    nodes: int = 0


def exact_subset(amounts: np.ndarray, target: int, max_size: int = 5, min_size: int = 1,
                 node_budget: int = 100_000) -> SubsetResult:
    """Sous-ensemble unique de `amounts` (entiers > 0) sommant exactement à `target`."""
    values = np.asarray(amounts, dtype=np.int64)
    order = np.argsort(values, kind="stable")
    a = values[order].tolist()
    n = len(a)
    if n == 0 or target <= 0:
        return SubsetResult(NONE)
    # suffix_max[i][k] n'est pas nécessaire : borne supérieure = somme des k plus grands à partir de i,
    # soit les k derniers éléments (tableau trié).
    top = [0] * (max_size + 1)
    for k in range(1, max_size + 1):
        top[k] = sum(a[-k:]) if k <= n else sum(a)
    solutions: list[tuple[int, ...]] = []
    nodes = 0
    stack: list[int] = []

    def dfs(start: int, total: int) -> bool:
        """Retourne False pour interrompre (deux solutions ou budget épuisé)."""
        nonlocal nodes
        depth = len(stack)
        for i in range(start, n):
            nodes += 1
            if nodes > node_budget:
                return False
            s = total + a[i]
            if s > target:
                break
            stack.append(i)
            if s == target and depth + 1 >= min_size:
                solutions.append(tuple(stack))
                if len(solutions) > 1:
                    return False
            elif depth + 1 < max_size:
                remaining = max_size - depth - 1
                if s + top[min(remaining, n)] >= target and not dfs(i + 1, s):
                    return False
            stack.pop()
        return True

    completed = dfs(0, 0)
    if len(solutions) > 1:
        return SubsetResult(AMBIGUOUS, nodes=nodes)
    if not completed:
        return SubsetResult(BUDGET, nodes=nodes)
    if not solutions:
        return SubsetResult(NONE, nodes=nodes)
    return SubsetResult(UNIQUE, tuple(sorted(int(order[i]) for i in solutions[0])), nodes)


def near_subsets(amounts: np.ndarray, target: int, tolerance: int, max_size: int = 5, min_size: int = 2,
                 node_budget: int = 100_000, max_solutions: int = 20) -> tuple[list[tuple[int, ...]], bool]:
    """Sous-ensembles dont la somme est à `tolerance` près de `target` (ensembles de l'étape 5).

    Retourne (solutions en indices de `amounts`, parcours complet ?). Solutions triées par
    cardinalité croissante : la parcimonie évite les combinaisons fortuites.
    """
    values = np.asarray(amounts, dtype=np.int64)
    order = np.argsort(values, kind="stable")
    a = values[order].tolist()
    n = len(a)
    lo, hi = target - tolerance, target + tolerance
    top = [sum(a[-k:]) if 0 < k <= n else sum(a) for k in range(max_size + 1)]
    solutions: list[tuple[int, ...]] = []
    nodes = 0
    stack: list[int] = []

    def dfs(start: int, total: int) -> bool:
        nonlocal nodes
        depth = len(stack)
        for i in range(start, n):
            nodes += 1
            if nodes > node_budget:
                return False
            s = total + a[i]
            if s > hi:
                break
            stack.append(i)
            if s >= lo and depth + 1 >= min_size:
                solutions.append(tuple(sorted(int(order[j]) for j in stack)))
                if len(solutions) >= max_solutions:
                    return False
            if depth + 1 < max_size and s + top[min(max_size - depth - 1, n)] >= lo and not dfs(i + 1, s):
                return False
            stack.pop()
        return True

    complete = dfs(0, 0)
    solutions.sort(key=len)
    return solutions, complete


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/reconcile_rules/matcher.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Étape 4 — réconciliation algorithmique par règles déterministes (brief §6).

Chaque jour, pour chaque paiement du lot :
1. allocation (étape 3) → débiteurs candidats, allocation ferme éventuelle, client file ;
2. règles actives de `config/rules.yaml`, par priorité croissante, sur les factures
   ouvertes à D de ces débiteurs ; la première qui trouve une solution **unique** l'emporte ;
3. arbitrage au niveau du lot : une facture n'est bloquée que si elle est soldée
   totalement ; deux paiements soldant la même facture par la même règle sont rejetés.

Ce qui n'est pas résolu part à l'étape 5. Chaque décision porte l'identifiant et la
version de la règle.

Réservations du moteur : en rejeu l'état suit les imputations réelles. Le moteur tient
le registre des montants qu'il a lui-même imputés et les retire du restant dû, jusqu'à
ce que l'imputation réelle du paiement apparaisse dans l'état.

Mesure « seule » : les règles R1, R2, R3 et R5 sont aussi évaluées indépendamment sur
tous les paiements (propositions enregistrées dans `proposals`) ; R4, coûteuse, sur le
résiduel des règles de priorité inférieure.
"""




R1, R2, R3, R4, R5 = "R1_CLIENT_FILE", "R2_REFERENCE_UNIQUE", "R3_AMOUNT_UNIQUE", "R4_EXACT_SUM", "R5_PARTIAL_REFERENCED"
_PROPOSAL_COLUMNS = ["row", "inv", "amount"]


def _no_proposal() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="int64") for c in _PROPOSAL_COLUMNS})


def _first_two(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Au plus deux lignes par groupe : les règles exigent une facture unique par paiement, deux factures
    suffisent à établir l'ambiguïté. Évite de développer les milliers de factures d'une clé fréquente ou
    d'un montant rond."""
    return frame[frame.groupby(by, sort=False).cumcount().to_numpy() < 2]


def _unique_per_row(pairs: pd.DataFrame) -> pd.DataFrame:
    """Lignes (row, inv, ...) des paiements pour lesquels une seule facture distincte ressort."""
    pairs = pairs.drop_duplicates(["row", "inv"])
    n = pairs.groupby("row")["inv"].transform("size").to_numpy()
    return pairs[n == 1]


class RulesMatcher:
    name = "rules"

    def __init__(self, state: LedgerState, settings: Settings, rules: RulesConfig,
                 allocator: Allocator | None = None):
        self.state = state
        self.allocator = allocator or Allocator(state, settings.allocation)
        self.ref = self.allocator.ref
        self.rules: list[RuleConfig] = rules.active()
        self.version = rules.version
        self.min_key_length = settings.allocation.signals.reference.min_key_length
        inv, pay = state.table("invoice"), state.table("payment")
        self._inv_ids = inv["invoice_id"].astype(object).to_numpy()
        self._inv_debtor = state.invoice_debtor_positions
        self._pay_amount = pay["amount"].fillna(-1).to_numpy(dtype=np.int64)
        self._claimed = np.zeros(len(inv), dtype=np.int64)          # montants réservés par le moteur
        self._claims: dict[int, list[tuple[int, int]]] = {}         # paiement → [(facture, montant)]
        self._proposed: set[tuple[str, int]] = set()
        self._proposals: list[pd.DataFrame] = []
        self.last_allocation = None

    # --- Registre des réservations ------------------------------------------------------------------

    def _release_realised(self, as_of) -> None:
        if not self._claims:
            return
        payments = np.fromiter(self._claims, dtype=np.int64, count=len(self._claims))
        done = payments[self.state.payment_imputed_at(payments, as_of)]
        for p in done.tolist():
            for inv, amount in self._claims.pop(p):
                self._claimed[inv] -= amount

    def _effective(self, inv: np.ndarray, balance: np.ndarray) -> np.ndarray:
        return balance - self._claimed[inv]

    # --- Portée ---------------------------------------------------------------------------------------

    def _scopes(self, alloc, batch_ids: np.ndarray) -> dict[int, pd.DataFrame]:
        """(row, debtor) : 0 → tous les candidats de l'allocation ; 1 → débiteur de l'allocation ferme."""
        row_of = pd.Index(batch_ids)
        debtor_pos = self.state.party_pos["debtor"]
        cand = alloc.candidates
        all_ = pd.DataFrame({"row": row_of.get_indexer(cand["payment_id"].astype(object).to_numpy()),
                             "debtor": debtor_pos(cand["debtor_id"]), "rank": cand["rank"].to_numpy()})
        firm = alloc.payments[alloc.payments["firm_debtor_id"].notna()]
        firm_ = pd.DataFrame({"row": row_of.get_indexer(firm["payment_id"].astype(object).to_numpy()),
                              "debtor": debtor_pos(firm["firm_debtor_id"]), "rank": 1})
        return {0: all_[all_["debtor"] >= 0], 1: firm_[firm_["debtor"] >= 0]}

    @staticmethod
    def _in_scope(pairs: pd.DataFrame, scope: pd.DataFrame) -> pd.DataFrame:
        return pairs.merge(scope, on=["row", "debtor"])

    # --- Règles ----------------------------------------------------------------------------------------

    def _referenced(self, pos: np.ndarray, as_of, scope: pd.DataFrame) -> pd.DataFrame:
        """(row, inv, balance) : factures ouvertes des débiteurs de la portée citées par le libellé."""
        row, keys = self.ref.payment_keys.gather(pos)
        keep = self.ref.vocab.lengths[keys] >= self.min_key_length
        row, keys = row[keep], keys[keep]
        if len(keys) == 0:
            return pd.DataFrame({c: pd.Series(dtype="int64") for c in ("row", "inv", "balance")})
        ukeys, k_of_row = np.unique(keys, return_inverse=True)
        k_owner, inv = self.ref.key_invoices.gather(ukeys)
        balance = self._effective(inv, self.state.open_balance_at(inv, as_of))
        ok = balance > 0
        invoices = _first_two(pd.DataFrame({"k": k_owner[ok], "debtor": self._inv_debtor[inv[ok]], "inv": inv[ok],
                                            "balance": balance[ok]}), ["k", "debtor"])
        # Portée d'abord (quelques débiteurs par paiement), puis factures par (clé, débiteur).
        asked = pd.DataFrame({"row": row, "k": k_of_row}).drop_duplicates().merge(
            scope[["row", "debtor"]].drop_duplicates(), on="row")
        pairs = asked.merge(invoices, on=["k", "debtor"])
        return (pairs[["row", "inv", "balance"]].drop_duplicates(["row", "inv"])
                .sort_values("row", kind="stable").reset_index(drop=True))

    def _r1(self, pos, amount, alloc, as_of, scope, params) -> pd.DataFrame:
        files = alloc.payments["client_file_id"].to_numpy(dtype=object)
        rows = np.flatnonzero(pd.notna(files))
        if len(rows) == 0:
            return _no_proposal()
        row_of_file = pd.Series(rows, index=files[rows])
        lines = self.state.client_file_lines(files[rows], as_of).reset_index(drop=True)
        if lines.empty:
            return _no_proposal()
        lengths, flat = _flatten(lines["invoice_reference_keys"])
        line_of_key = np.repeat(np.arange(len(lines)), lengths)
        keys = self.ref.vocab.lookup(flat)
        ok = (keys >= 0)
        ok[ok] = self.ref.vocab.lengths[keys[ok]] >= self.min_key_length
        line_of_key, keys = line_of_key[ok], keys[ok]
        k_owner, inv = self.ref.key_invoices.gather(keys)
        balance = self._effective(inv, self.state.open_balance_at(inv, as_of))
        line = line_of_key[k_owner]
        cand = pd.DataFrame({"line": line, "inv": inv, "balance": balance, "debtor": self._inv_debtor[inv]})
        cand = cand[cand["balance"] > 0]
        cand["row"] = row_of_file.reindex(lines["file_id"].to_numpy()[cand["line"].to_numpy()]).to_numpy()
        cand = self._in_scope(cand, scope).drop_duplicates(["line", "inv"])
        n_per_line = cand.groupby("line")["inv"].transform("size").to_numpy()
        resolved = cand[n_per_line == 1].copy()
        line_amount = lines["amount"].to_numpy(dtype=object)[resolved["line"].to_numpy()]
        resolved["amount"] = [int(b) if pd.isna(a) else int(a) for a, b in zip(line_amount, resolved["balance"])]
        tol_abs, tol_rel = params.get("tolerance_abs_cents", 0), params.get("tolerance_rel", 0.0)
        gap = np.abs(resolved["amount"].to_numpy() - resolved["balance"].to_numpy())
        resolved["concordant"] = gap <= tol_abs + tol_rel * resolved["balance"].to_numpy()
        n_lines = lines.groupby("file_id").size()
        out = []
        for row, group in resolved.groupby("row"):
            file_id = files[row]
            if (len(group) != n_lines[file_id] or not group["concordant"].all()
                    or group["inv"].duplicated().any()
                    or abs(int(group["amount"].sum()) - int(amount[row])) > tol_abs):
                continue
            out.append(group[["row", "inv", "amount"]])
        return pd.concat(out, ignore_index=True) if out else _no_proposal()

    def _r2(self, referenced: pd.DataFrame, amount: np.ndarray, params) -> pd.DataFrame:
        u = _unique_per_row(referenced)
        gap = amount[u["row"].to_numpy()] - u["balance"].to_numpy()
        u = u[np.abs(gap) <= params.get("tolerance_abs_cents", 0)]
        return pd.DataFrame({"row": u["row"].to_numpy(), "inv": u["inv"].to_numpy(),
                             "amount": u["balance"].to_numpy()})

    def _r5(self, referenced: pd.DataFrame, amount: np.ndarray, params) -> pd.DataFrame:
        u = _unique_per_row(referenced)
        a = amount[u["row"].to_numpy()]
        u, a = u[(a > 0) & (a < u["balance"].to_numpy())], a[(a > 0) & (a < u["balance"].to_numpy())]
        return pd.DataFrame({"row": u["row"].to_numpy(), "inv": u["inv"].to_numpy(), "amount": a})

    def _r3(self, amount: np.ndarray, as_of, scope: pd.DataFrame) -> pd.DataFrame:
        inv, balance = self.state.open_invoice_positions(as_of)
        eff = self._effective(inv, balance)
        open_ = pd.DataFrame({"amount": eff, "debtor": self._inv_debtor[inv], "inv": inv})
        open_ = _first_two(open_[open_["amount"] > 0], ["amount", "debtor"])
        # Portée d'abord, puis factures par (montant, débiteur) : pas de jointure sur le seul montant.
        rows = scope[["row", "debtor"]].drop_duplicates()
        rows = rows.assign(amount=amount[rows["row"].to_numpy()])
        pairs = rows[rows["amount"] > 0].merge(open_, on=["amount", "debtor"])
        u = _unique_per_row(pairs).sort_values("row", kind="stable")
        return u[["row", "inv", "amount"]].reset_index(drop=True)

    def _r4(self, rows: np.ndarray, amount: np.ndarray, as_of, scope: pd.DataFrame, params) -> pd.DataFrame:
        scope = scope[scope["row"].isin(rows)]
        if scope.empty:
            return _no_proposal()
        debtors = np.unique(scope["debtor"].to_numpy())
        owner, inv, balance = self.state.debtor_open_invoices_at(debtors, as_of)
        eff = self._effective(inv, balance)
        ok = eff > 0
        by_debtor = pd.DataFrame({"debtor": debtors[owner[ok]], "inv": inv[ok], "balance": eff[ok]})
        grouped = {d: (g["inv"].to_numpy(), g["balance"].to_numpy()) for d, g in by_debtor.groupby("debtor")}
        max_open = int(params.get("max_open_invoices", 30))
        out = []
        for row, group in scope.groupby("row"):
            pools = [grouped[d] for d in group["debtor"] if d in grouped]
            if not pools:
                continue
            invs = np.concatenate([p[0] for p in pools])
            bals = np.concatenate([p[1] for p in pools])
            fits = bals <= amount[row]
            invs, bals = invs[fits], bals[fits]
            if len(invs) == 0 or len(invs) > max_open:
                continue
            res = exact_subset(bals, int(amount[row]), max_size=int(params.get("max_invoices", 5)),
                               node_budget=int(params.get("node_budget", 100_000)))
            if res.status == UNIQUE:
                idx = list(res.indices)
                out.append(pd.DataFrame({"row": row, "inv": invs[idx], "amount": bals[idx]}))
        return pd.concat(out, ignore_index=True) if out else _no_proposal()

    # --- Journée ------------------------------------------------------------------------------------------

    def process(self, ctx: DayContext) -> pd.DataFrame:
        as_of = ctx.as_of
        self._release_realised(as_of)
        batch_ids = ctx.batch["payment_id"].astype(object).to_numpy()
        pos = self.state.pay_pos(pd.Series(batch_ids, dtype=object))
        amount = self._pay_amount[pos]
        alloc = self.allocator.allocate(ctx)
        self.last_allocation = alloc
        scopes = self._scopes(alloc, batch_ids)

        by_rule: dict[str, pd.DataFrame] = {}
        referenced = {}
        decided = np.zeros(len(pos), dtype=bool)
        for rule in self.rules:
            memory.mark(f"règles · {rule.id}")
            scope = scopes[int(rule.params.get("firm_only", 0))]
            if rule.id == R1:
                found = self._r1(pos, amount, alloc, as_of, scope, rule.params)
            elif rule.id in (R2, R5):
                key = int(rule.params.get("firm_only", 0))
                if key not in referenced:
                    referenced[key] = self._referenced(pos, as_of, scope)
                found = (self._r2 if rule.id == R2 else self._r5)(referenced[key], amount, rule.params)
            elif rule.id == R3:
                found = self._r3(amount, as_of, scope)
            elif rule.id == R4:
                found = self._r4(np.flatnonzero(~decided), amount, as_of, scope, rule.params)
            else:
                raise ValueError(f"règle inconnue : {rule.id}")
            by_rule[rule.id] = found
            decided[found["row"].to_numpy()] = True
        self._record(by_rule, batch_ids, pos, ctx.day)

        # Cascade : la règle de plus haute priorité ayant une solution unique l'emporte.
        chosen, taken = [], np.zeros(len(pos), dtype=bool)
        for prio, rule in enumerate(self.rules):
            found = by_rule[rule.id]
            f = found[~taken[found["row"].to_numpy()]]
            if len(f):
                chosen.append(f.assign(rule_id=rule.id, prio=prio))
                taken[f["row"].to_numpy()] = True
        if not chosen:
            return pd.DataFrame(columns=DECISION_COLUMNS)
        accepted = self._arbitrate(pd.concat(chosen, ignore_index=True), batch_ids, as_of)
        for row, group in accepted.groupby("row"):
            claims = list(zip(group["inv"].tolist(), group["amount"].tolist()))
            self._claims.setdefault(int(pos[row]), []).extend(claims)
            np.add.at(self._claimed, group["inv"].to_numpy(), group["amount"].to_numpy())
        return pd.DataFrame({
            "payment_id": batch_ids[accepted["row"].to_numpy()], "invoice_id": self._inv_ids[accepted["inv"].to_numpy()],
            "amount": pd.array(accepted["amount"].to_numpy(), dtype="Int64"), "action": AUTO, "step": "rules",
            "rule_id": accepted["rule_id"].to_numpy(), "rule_version": pd.array([self.version] * len(accepted), "Int64"),
            "score": 1.0,
        })

    def _arbitrate(self, props: pd.DataFrame, batch_ids: np.ndarray, as_of) -> pd.DataFrame:
        """Une facture n'est bloquée que si elle est soldée totalement ; ex æquo de même règle → rejet."""
        props = props.assign(pid=batch_ids[props["row"].to_numpy()])
        per_inv = props.groupby("inv")["row"].nunique()
        contested = per_inv[per_inv > 1].index.to_numpy()
        if len(contested) == 0:
            return props
        balance = dict(zip(contested.tolist(), self._effective(
            contested, self.state.open_balance_at(contested, as_of)).tolist()))
        touching = props[props["inv"].isin(contested)]
        # Ex æquo : plusieurs paiements, même règle, soldant chacun la facture.
        full = touching[touching["amount"] >= touching["inv"].map(balance)]
        tied = full.groupby(["inv", "prio"])["row"].nunique()
        tied_rows = set(full.merge(tied[tied > 1].reset_index()[["inv", "prio"]], on=["inv", "prio"])["row"])
        rejected = set(tied_rows)
        for row, group in touching.sort_values(["prio", "pid"]).groupby(["prio", "pid"], sort=False):
            r = int(group["row"].iloc[0])
            if r in rejected:
                continue
            need = group.groupby("inv")["amount"].sum()
            if all(balance[i] >= a for i, a in need.items()):
                for i, a in need.items():
                    balance[i] -= a
            else:
                rejected.add(r)
        return props[~props["row"].isin(rejected)]

    # --- Mesure « seule » ---------------------------------------------------------------------------------

    def _record(self, by_rule: dict[str, pd.DataFrame], batch_ids: np.ndarray, pos: np.ndarray, day) -> None:
        for rule_id, found in by_rule.items():
            if found.empty:
                continue
            f = found.sort_values(["row", "inv"])
            rows = f["row"].to_numpy()
            starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
            ends = np.r_[starts[1:], len(rows)]
            inv = self._inv_ids[f["inv"].to_numpy()]
            new = [(rows[s], tuple(inv[s:e])) for s, e in zip(starts, ends)
                   if (rule_id, int(pos[rows[s]])) not in self._proposed]
            if not new:
                continue
            self._proposed.update((rule_id, int(pos[r])) for r, _ in new)
            self._proposals.append(pd.DataFrame({"payment_id": batch_ids[[r for r, _ in new]], "rule_id": rule_id,
                                                 "invoices": [t for _, t in new], "day": day}))

    def proposals(self) -> pd.DataFrame:
        if not self._proposals:
            return pd.DataFrame(columns=["payment_id", "rule_id", "invoices", "day"])
        return pd.concat(self._proposals, ignore_index=True)


def rule_alone_metrics(proposals: pd.DataFrame, truth: pd.DataFrame, scope: pd.DataFrame,
                       rules: list[RuleConfig]) -> pd.DataFrame:
    """Précision et couverture de chaque règle prise seule, sur les paiements du périmètre."""
    truth_map = truth.set_index("payment_id")["truth_invoices"]
    in_scope = set(scope["payment_id"])
    n = len(scope)
    rows = []
    for rule in rules:
        p = proposals[(proposals["rule_id"] == rule.id) & proposals["payment_id"].isin(in_scope)]
        correct = sum(1 for pid, inv in zip(p["payment_id"], p["invoices"]) if truth_map.get(pid) == tuple(inv))
        rows.append({"rule_id": rule.id, "règle": rule.name, "propositions": len(p), "correctes": correct,
                     "précision": correct / len(p) if len(p) else None, "couverture": len(p) / n if n else 0.0})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/reconcile_ml/features.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Étape 5 — candidats et features des paires (paiement, facture) (brief §7.1-7.2, spec §4, §5.3).

Candidats (union des clés), pour chaque paiement du résiduel :
- K1 débiteur : factures ouvertes des débiteurs candidats de l'allocation, échéance dans la fenêtre ;
- K2 référence : factures ouvertes dont une clé figure dans le libellé, sans fenêtre ;
- K3 montant : factures ouvertes de restant dû égal au paiement, créées à ± 90 jours ;
- K4 client file : factures citées par le client file rattaché.
Filtres durs : même devise, facture ouverte à D (réservations du moteur déduites), contrat actif à D.

Toutes les features se calculent sur l'état à D ; les agrégats comportementaux sont ceux de
`LedgerState.debtor_stats` (fenêtre strictement antérieure).
"""




FEATURIZATION_VERSION = "1.1.0"
SRC_DEBTOR, SRC_REFERENCE, SRC_AMOUNT, SRC_CLIENT_FILE = 1, 2, 4, 8
ROUTES = ("DEBTOR_DIRECT", "ASSIGNOR", "TECHNICAL_ACCOUNT", "UNKNOWN")

FAMILIES: dict[str, list[str]] = {
    "amount": ["amount_diff", "amount_diff_rel", "amount_exact", "payment_covers", "amount_ratio",
               "is_typical_discount", "is_bank_fee_gap", "is_retention_gap", "ratio_to_initial", "balance_is_partial"],
    "temporal": ["days_to_due", "days_since_creation", "is_before_creation", "days_to_due_zscore"],
    "textual": ["ref_full_in_label", "ref_key_in_label", "internal_ref_in_label", "label_length",
                "label_has_no_alpha", "n_label_numbers"],
    "identity": ["iban_route", "iban_matches_invoice_debtor", "channel", "bankroll_code"],
    "behavioral": ["debtor_mean_payment_delay", "debtor_std_payment_delay", "debtor_partial_payment_rate",
                   "debtor_grouping_rate", "debtor_ref_citation_rate", "debtor_payment_count",
                   "debtor_open_invoice_count", "debtor_open_invoice_amount"],
    "contract": ["market", "product", "recourse"],
    "allocation": ["alloc_rank", "alloc_score", "alloc_is_firm_debtor", "debtor_in_allocation",
                   *[f"alloc_sig_{s}" for s in SIGNALS]],
    "client_file": ["has_client_file", "invoice_cited_in_client_file", "client_file_line_amount_diff",
                    "client_file_total_matches_payment"],
}
BASE = ["src_debtor", "src_reference", "src_amount", "src_client_file", "n_candidates", "amount_rank_in_payment"]
COMPETITION = ["rank_in_payment", "score_margin", "score_best_other"]
CATEGORICAL = ["iban_route", "channel", "bankroll_code", "market", "product", "recourse"]


def active_features(ml: MLSettings) -> list[str]:
    """Features du modèle selon les familles activées dans les paramètres."""
    fam = ml.features
    return BASE + [f for name, cols in FAMILIES.items() if getattr(fam, name) for f in cols]


def _codes(values: pd.Series, categories: list) -> np.ndarray:
    return pd.Categorical(pd.Series(values).astype(object), categories=categories).codes.astype(np.float32)


def _group_starts(keys: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.array([], dtype=np.int64)


# Paires (paiement, facture) examinées au plus par bloc avant plafonnement : les gros débiteurs, les
# références et montants fréquents produisent des milliers de paires par paiement.
CANDIDATE_PAIR_BUDGET = 2_000_000


def _row_blocks(load: np.ndarray, budget: int) -> list[tuple[int, int]]:
    """Découpe consécutive de lignes dont la charge cumulée reste de l'ordre de `budget`."""
    if len(load) == 0:
        return [(0, 0)]
    block = (np.cumsum(load) - load) // max(budget, 1)
    starts = np.flatnonzero(np.r_[True, block[1:] != block[:-1]])
    return list(zip(starts.tolist(), np.r_[starts[1:], len(load)].tolist()))


class Featurizer:
    """Candidats et features ; attributs statiques pré-calculés une fois."""

    def __init__(self, state: LedgerState, allocator: Allocator, ml: MLSettings, min_key_length: int,
                 categories: dict[str, list] | None = None):
        """`categories` : vocabulaires figés du modèle (inférence) ; à l'entraînement, construits sur les
        données disponibles puis sauvegardés avec le modèle."""
        self.state, self.allocator, self.ml = state, allocator, ml
        self.ref = allocator.ref
        self.min_key_length = min_key_length
        inv, pay = state.table("invoice"), state.table("payment")
        agr = state.table("agreement")
        self.inv_debtor = state.invoice_debtor_positions
        self.inv_due = _days(inv["due_date"])
        self.inv_created = _days(inv["creation_date"])
        self.inv_initial = inv["initial_amount"].fillna(0).to_numpy(dtype=np.int64)
        self.inv_agr = state.agr_pos(inv["agreement_id"])
        self.pay_amount = pay["amount"].fillna(-1).to_numpy(dtype=np.int64)
        self.pay_value = _days(pay["value_date"])
        currencies = sorted(set(inv["currency"].dropna()) | set(pay["currency"].dropna()))
        self.inv_ccy = _codes(inv["currency"], currencies)
        self.pay_ccy = _codes(pay["currency"], currencies)
        self.categories = categories or {
            "iban_route": list(ROUTES),
            "channel": sorted(pay["channel"].dropna().astype(str).unique()),
            "bankroll_code": sorted(pay["bankroll_code"].dropna().astype(str).unique()),
            "market": sorted(agr["market"].dropna().astype(str).unique()),
            "product": sorted(agr["product"].dropna().astype(str).unique()),
            "recourse": sorted(agr["recourse"].dropna().astype(str).unique()),
        }
        self.pay_channel = _codes(pay["channel"], self.categories["channel"])
        self.pay_bankroll = _codes(pay["bankroll_code"], self.categories["bankroll_code"])
        agr_idx = np.maximum(self.inv_agr, 0)
        has_agr = self.inv_agr >= 0
        self.inv_market = np.where(has_agr, _codes(agr["market"], self.categories["market"])[agr_idx], -1)
        self.inv_product = np.where(has_agr, _codes(agr["product"], self.categories["product"])[agr_idx], -1)
        self.inv_recourse = np.where(has_agr, _codes(agr["recourse"], self.categories["recourse"])[agr_idx], -1)
        labels = pay["label_norm"].fillna("").astype(str)
        self.pay_label_len = labels.str.len().to_numpy(dtype=np.float32)
        self.pay_no_alpha = (~labels.str.contains(r"[A-Z]", regex=True)).to_numpy(dtype=np.float32)
        # Nombre de clés du libellé lui-même (et non de celles présentes dans le vocabulaire des références,
        # qui dépend des factures futures).
        self.pay_n_numbers = pay["label_numbers"].map(len).to_numpy(dtype=np.float32)
        # Facture → clés de sa référence client / de sa référence interne ; clé complète.
        self.inv_client_keys = self._inv_keys(inv["client_reference_keys"])
        self.inv_internal_keys = self._inv_keys(inv["internal_reference_keys"])
        compact = inv["client_reference_norm"].fillna("").astype(str).str.replace(" ", "", regex=False)
        self.inv_full_key = self.ref.vocab.lookup(compact.astype(object).to_numpy())

    def _inv_keys(self, column: pd.Series) -> Postings:
        lengths, flat = _flatten(column)
        return Postings.from_lists(lengths, self.ref.vocab.lookup(flat))

    # --- Candidats ------------------------------------------------------------------------------------

    def candidates(self, rows: np.ndarray, pos: np.ndarray, alloc: Allocation, scope: pd.DataFrame, as_of,
                   claimed: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Candidats (row, inv, balance, src) des lignes `rows` du lot, et factures citées par client file.

        `pos` : positions des paiements de tout le lot ; `claimed` : réservations du moteur par facture.
        Les lignes sont traitées par blocs de charge bornée (`CANDIDATE_PAIR_BUDGET` paires avant
        plafonnement) : même résultat qu'en un seul passage, mémoire indépendante de la taille du lot.
        """
        rows = np.asarray(rows, dtype=np.int64)
        blocks = _row_blocks(self._pair_load(rows, pos, scope, as_of), CANDIDATE_PAIR_BUDGET)
        if len(blocks) == 1:
            return self._candidates_rows(rows, pos, alloc, scope, as_of, claimed)
        parts = [self._candidates_rows(rows[a:b], pos, alloc, scope, as_of, claimed) for a, b in blocks]
        c = pd.concat([p[0] for p in parts], ignore_index=True)
        c = c.iloc[np.argsort(c["row"].to_numpy(), kind="stable")].reset_index(drop=True)
        return c, pd.concat([p[1] for p in parts], ignore_index=True)

    def _pair_load(self, rows: np.ndarray, pos: np.ndarray, scope: pd.DataFrame, as_of) -> np.ndarray:
        """Majorant du nombre de paires examinées par ligne : factures ouvertes des débiteurs alloués,
        factures portant une clé du libellé, factures ouvertes de même montant."""
        cfg = self.ml.candidates
        load = np.zeros(len(rows), dtype=np.int64)
        index_of = pd.Series(np.arange(len(rows)), index=rows)
        if cfg.allocated_debtors and len(scope):
            sc = scope[scope["row"].isin(rows)]
            if "rank" in sc.columns:
                sc = sc[sc["rank"] <= cfg.max_debtors]
            if len(sc):
                debtors, d_of = np.unique(sc["debtor"].to_numpy(), return_inverse=True)
                owner, _, _ = self.state.debtor_open_invoices_at(debtors, as_of)
                n_open = np.bincount(owner, minlength=len(debtors))
                np.add.at(load, index_of.loc[sc["row"].to_numpy()].to_numpy(), n_open[d_of])
        if cfg.reference_no_window:
            r, keys = self.ref.payment_keys.gather(pos[rows])
            np.add.at(load, r, self.ref.key_invoices.lengths(keys))
        if cfg.amount_exact:
            _, balance = self.state.open_invoice_positions(as_of)
            balance = np.sort(balance)
            amount = self.pay_amount[pos[rows]]
            load += np.searchsorted(balance, amount, side="right") - np.searchsorted(balance, amount, side="left")
        return load

    def _candidates_rows(self, rows: np.ndarray, pos: np.ndarray, alloc: Allocation, scope: pd.DataFrame, as_of,
                         claimed: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.ml.candidates
        state = self.state
        rows = np.asarray(rows, dtype=np.int64)
        parts = []
        if cfg.allocated_debtors:
            sc = scope[scope["row"].isin(rows)]
            if "rank" in sc.columns:
                sc = sc[sc["rank"] <= cfg.max_debtors][["row", "debtor"]]
            if len(sc):
                debtors = np.unique(sc["debtor"].to_numpy())
                owner, inv, _ = state.debtor_open_invoices_at(debtors, as_of)
                k1 = sc.merge(pd.DataFrame({"debtor": debtors[owner], "inv": inv}), on="debtor")
                due = self.inv_due[k1["inv"].to_numpy()]
                value = self.pay_value[pos[k1["row"].to_numpy()]]
                ok = (due >= value - cfg.debtor_window_before_days) & (due <= value + cfg.debtor_window_after_days)
                parts.append(pd.DataFrame({"row": k1["row"].to_numpy()[ok], "inv": k1["inv"].to_numpy()[ok],
                                           "src": SRC_DEBTOR}))
        if cfg.reference_no_window:
            r, keys = self.ref.payment_keys.gather(pos[rows])
            keep = self.ref.vocab.lengths[keys] >= self.min_key_length
            r, keys = rows[r[keep]], keys[keep]
            if len(keys):
                uk, k_of = np.unique(keys, return_inverse=True)
                k_owner, inv = self.ref.key_invoices.gather(uk)
                pairs = pd.DataFrame({"row": r, "k": k_of}).drop_duplicates().merge(
                    pd.DataFrame({"k": k_owner, "inv": inv}), on="k")
                parts.append(pd.DataFrame({"row": pairs["row"].to_numpy(), "inv": pairs["inv"].to_numpy(),
                                           "src": SRC_REFERENCE}))
        if cfg.amount_exact:
            inv, balance = state.open_invoice_positions(as_of)
            opens = pd.DataFrame({"amount": balance - claimed[inv], "inv": inv})
            pays = pd.DataFrame({"row": rows, "amount": self.pay_amount[pos[rows]]})
            k3 = pays[pays["amount"] > 0].merge(opens, on="amount")
            gap = np.abs(self.inv_created[k3["inv"].to_numpy()] - self.pay_value[pos[k3["row"].to_numpy()]])
            k3 = k3[gap <= cfg.amount_window_days]
            parts.append(pd.DataFrame({"row": k3["row"].to_numpy(), "inv": k3["inv"].to_numpy(), "src": SRC_AMOUNT}))
        cited = self._client_file_invoices(rows, alloc, as_of)
        if cfg.client_file_cited and len(cited):
            parts.append(pd.DataFrame({"row": cited["row"].to_numpy(dtype=np.int64),
                                       "inv": cited["inv"].to_numpy(dtype=np.int64), "src": SRC_CLIENT_FILE}))
        parts = [p for p in parts if len(p)]
        if not parts:
            empty = pd.DataFrame({c: pd.Series(dtype="int64") for c in ("row", "inv", "balance", "src")})
            return empty, cited

        c = self._or_sources(pd.concat(parts, ignore_index=True))
        inv = c["inv"].to_numpy()
        balance = state.open_balance_at(inv, as_of) - claimed[inv]
        agr = self.inv_agr[inv]
        ok = (balance > 0) & (self.inv_ccy[inv] == self.pay_ccy[pos[c["row"].to_numpy()]])
        ok &= (agr < 0) | state.agreement_active_at(np.maximum(agr, 0), as_of)
        c = c[ok].assign(balance=balance[ok])
        return self._cap(c, pos, cfg.max_per_payment), cited

    def _cap(self, c: pd.DataFrame, pos: np.ndarray, cap: int) -> pd.DataFrame:
        """Plafond par paiement : clés précises (référence, montant, client file) d'abord ; puis, parmi les
        factures du débiteur, l'union des plus proches en montant et des échéances les plus proches de la
        date de valeur parmi celles qui peuvent entrer dans le paiement (paiements groupés)."""
        row = c["row"].to_numpy()
        amount = self.pay_amount[pos[row]]
        balance = c["balance"].to_numpy()
        strong = (c["src"].to_numpy() & ~SRC_DEBTOR) > 0
        gap = np.abs(balance - amount) / np.maximum(amount, 1)
        fits = balance <= amount + np.maximum(500, 0.03 * amount)
        due_gap = np.abs(self.pay_value[pos[row]] - self.inv_due[c["inv"].to_numpy()]).astype(np.float64)
        due_gap = np.where(fits, due_gap, np.inf)

        def rank_by(key: np.ndarray) -> np.ndarray:
            order = np.lexsort((c["inv"].to_numpy(), key, row))
            r = np.empty(len(row), dtype=np.int64)
            starts = np.flatnonzero(np.r_[True, row[order][1:] != row[order][:-1]]) if len(row) else np.array([], int)
            r[order] = np.arange(len(row)) - np.repeat(starts, np.diff(np.r_[starts, len(row)]))
            return r

        half = cap // 3
        keep = strong | (rank_by(gap) < half) | ((rank_by(due_gap) < cap - half) & fits)
        c = c[keep].assign(_order=np.where(strong, 0, 1)[keep], _gap=gap[keep])
        c = c.sort_values(["row", "_order", "_gap", "inv"], kind="mergesort")
        c = c[c.groupby("row").cumcount().to_numpy() < cap]
        return c.drop(columns=["_order", "_gap"]).reset_index(drop=True)

    @staticmethod
    def _or_sources(c: pd.DataFrame) -> pd.DataFrame:
        c = c.sort_values(["row", "inv"], kind="mergesort")
        row, inv, src = c["row"].to_numpy(), c["inv"].to_numpy(), c["src"].to_numpy(dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, (row[1:] != row[:-1]) | (inv[1:] != inv[:-1])])
        return pd.DataFrame({"row": row[starts], "inv": inv[starts], "src": np.bitwise_or.reduceat(src, starts)})

    def _client_file_invoices(self, rows: np.ndarray, alloc: Allocation, as_of) -> pd.DataFrame:
        """(row, inv, line_amount, inv_cited) des factures citées par le client file rattaché."""
        empty = pd.DataFrame({"row": pd.Series(dtype="int64"), "inv": pd.Series(dtype="int64"),
                              "line_amount": pd.Series(dtype="float64"), "inv_cited": pd.Series(dtype="bool")})
        files = alloc.payments["client_file_id"].to_numpy(dtype=object)[rows]
        has = pd.notna(files)
        if not has.any():
            return empty
        row_of_file = pd.Series(rows[has], index=files[has])
        lines = self.state.client_file_lines(files[has], as_of).reset_index(drop=True)
        if lines.empty:
            return empty
        lengths, flat = _flatten(lines["invoice_reference_keys"])
        line_of_key = np.repeat(np.arange(len(lines)), lengths)
        keys = self.ref.vocab.lookup(flat)
        ok = keys >= 0
        ok[ok] = self.ref.vocab.lengths[keys[ok]] >= self.min_key_length
        k_owner, inv = self.ref.key_invoices.gather(keys[ok])
        line = line_of_key[ok][k_owner]
        amount = pd.to_numeric(lines["amount"], errors="coerce").to_numpy(dtype=np.float64)
        out = pd.DataFrame({"row": row_of_file.reindex(lines["file_id"].to_numpy()[line]).to_numpy(),
                            "inv": inv, "line_amount": amount[line], "inv_cited": True})
        return out.drop_duplicates(["row", "inv"])

    # --- Features ---------------------------------------------------------------------------------------

    def features(self, c: pd.DataFrame, pos: np.ndarray, alloc: Allocation, batch_ids: np.ndarray, as_of,
                 cited: pd.DataFrame | None = None) -> pd.DataFrame:
        """Une ligne de features par candidat (même ordre que `c`)."""
        n = len(c)
        row, inv = c["row"].to_numpy(), c["inv"].to_numpy()
        p = pos[row]
        amount = self.pay_amount[p].astype(np.float64)
        balance = c["balance"].to_numpy(dtype=np.float64)
        diff = amount - balance
        gap_rel = (balance - amount) / np.maximum(balance, 1)
        f: dict[str, np.ndarray] = {}
        src = c["src"].to_numpy(dtype=np.int64)
        f["src_debtor"] = (src & SRC_DEBTOR) > 0
        f["src_reference"] = (src & SRC_REFERENCE) > 0
        f["src_amount"] = (src & SRC_AMOUNT) > 0
        f["src_client_file"] = (src & SRC_CLIENT_FILE) > 0
        f["n_candidates"] = np.bincount(row, minlength=len(pos))[row]
        order = np.lexsort((np.abs(diff), row))
        starts = _group_starts(row[order])
        rank = np.empty(n, dtype=np.float32)
        rank[order] = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n])) + 1
        f["amount_rank_in_payment"] = rank

        f["amount_diff"] = diff / 100.0
        f["amount_diff_rel"] = diff / np.maximum(balance, 1)
        f["amount_exact"] = diff == 0
        f["payment_covers"] = amount >= balance
        f["amount_ratio"] = amount / np.maximum(balance, 1)
        f["is_typical_discount"] = (gap_rel >= 0.005) & (gap_rel <= 0.03)
        f["is_bank_fee_gap"] = (balance - amount >= 500) & (balance - amount <= 4000)
        f["is_retention_gap"] = (gap_rel >= 0.04) & (gap_rel <= 0.06)
        f["ratio_to_initial"] = amount / np.maximum(self.inv_initial[inv], 1)
        f["balance_is_partial"] = balance < self.inv_initial[inv]

        value = self.pay_value[p]
        f["days_to_due"] = (value - self.inv_due[inv]).astype(np.float64)
        f["days_since_creation"] = (value - self.inv_created[inv]).astype(np.float64)
        f["is_before_creation"] = value < self.inv_created[inv]

        f.update(self._text(inv, p))
        f["label_length"] = self.pay_label_len[p]
        f["label_has_no_alpha"] = self.pay_no_alpha[p]
        f["n_label_numbers"] = self.pay_n_numbers[p]

        # Allocation : rang, score et signaux du débiteur de la facture pour ce paiement.
        deb = self.inv_debtor[inv]
        a = alloc.candidates
        debtor_pos = self.state.party_pos["debtor"]
        row_of = pd.Index(batch_ids)
        alloc_tab = pd.DataFrame({"row": row_of.get_indexer(a["payment_id"].astype(object).to_numpy()),
                                  "deb": debtor_pos(a["debtor_id"]), "rank": a["rank"].to_numpy(dtype=np.float64),
                                  "score": a["score"].to_numpy(dtype=np.float64), "signals": a["signals"].to_numpy()})
        joined = pd.DataFrame({"row": row, "deb": deb}).merge(alloc_tab, on=["row", "deb"], how="left")
        f["alloc_rank"] = joined["rank"].to_numpy(dtype=np.float64)
        f["alloc_score"] = joined["score"].fillna(0).to_numpy(dtype=np.float64)
        f["debtor_in_allocation"] = joined["rank"].notna().to_numpy()
        sig = joined["signals"].fillna("").astype(str)
        for s in SIGNALS:
            f[f"alloc_sig_{s}"] = sig.str.contains(s, regex=False).to_numpy()
        firm = alloc.payments["firm_debtor_id"].to_numpy(dtype=object)
        firm_pos = np.full(len(firm), -1, dtype=np.int64)
        has_firm = pd.notna(firm)
        if has_firm.any():
            firm_pos[has_firm] = debtor_pos(pd.Series(firm[has_firm], dtype=object))
        f["alloc_is_firm_debtor"] = firm_pos[row] == deb
        route = alloc.payments["iban_route"].to_numpy(dtype=object)
        f["iban_route"] = _codes(pd.Series(route[row]), self.categories["iban_route"])
        f["iban_matches_invoice_debtor"] = (route[row] == "DEBTOR_DIRECT") & f["alloc_sig_iban"]
        f["channel"] = self.pay_channel[p]
        f["bankroll_code"] = self.pay_bankroll[p]
        f["market"], f["product"], f["recourse"] = self.inv_market[inv], self.inv_product[inv], self.inv_recourse[inv]

        udeb = np.unique(deb[deb >= 0])
        stats = self.state.debtor_stats(self.state.table("debtor")["party_id"].to_numpy()[udeb], as_of)
        stats.index = udeb
        st = stats.reindex(deb)
        for col in ("mean_payment_delay", "std_payment_delay", "partial_payment_rate", "grouping_rate",
                    "ref_citation_rate", "payment_count", "open_invoice_count", "open_invoice_amount"):
            f[f"debtor_{col}"] = st[col].to_numpy(dtype=np.float64)
        f["debtor_open_invoice_amount"] = f["debtor_open_invoice_amount"] / 100.0
        f["days_to_due_zscore"] = (f["days_to_due"] - f["debtor_mean_payment_delay"]) / \
            np.maximum(np.nan_to_num(f["debtor_std_payment_delay"], nan=1.0), 1.0)

        files = alloc.payments["client_file_id"].to_numpy(dtype=object)
        f["has_client_file"] = pd.notna(files)[row]
        # Le rattachement du client file exige l'égalité du montant total et du paiement.
        f["client_file_total_matches_payment"] = f["has_client_file"]
        if cited is not None and len(cited):
            cj = pd.DataFrame({"row": row, "inv": inv}).merge(cited, on=["row", "inv"], how="left")
            f["invoice_cited_in_client_file"] = cj["inv_cited"].fillna(False).to_numpy(dtype=bool)
            f["client_file_line_amount_diff"] = (cj["line_amount"].to_numpy(dtype=np.float64) - balance) / 100.0
        else:
            f["invoice_cited_in_client_file"] = np.zeros(n, dtype=bool)
            f["client_file_line_amount_diff"] = np.full(n, np.nan)
        out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
        for col in CATEGORICAL:
            out.loc[out[col] < 0, col] = np.nan          # catégorie inconnue
        return out

    def _text(self, inv: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
        """La référence (une clé quelconque, la clé complète, la référence interne) figure-t-elle au libellé ?"""
        n = len(inv)
        up, u_of = np.unique(p, return_inverse=True)
        lo, lk = self.ref.payment_keys.gather(up)
        label = pd.DataFrame({"u": lo, "k": lk}).drop_duplicates()

        def hits(i: np.ndarray, k: np.ndarray) -> np.ndarray:
            found = pd.DataFrame({"i": i, "u": u_of[i], "k": k}).merge(label, on=["u", "k"])["i"].unique()
            v = np.zeros(n, dtype=bool)
            v[found] = True
            return v

        out = {}
        for name, postings in (("ref_key_in_label", self.inv_client_keys),
                               ("internal_ref_in_label", self.inv_internal_keys)):
            out[name] = hits(*postings.gather(inv))
        full = self.inv_full_key[inv]
        idx = np.flatnonzero(full >= 0)
        out["ref_full_in_label"] = hits(idx, full[idx])
        return out


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/reconcile_ml/model.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Modèle de scoring des paires (brief §7.2) : LightGBM binaire en deux passes + calibration isotonique.

- Passe 1 : features des familles actives.
- Features de compétition dérivées des scores de passe 1 au sein d'un même paiement : rang,
  marge au meilleur autre candidat, meilleur score concurrent. En entraînement, les scores de
  passe 1 sont obtenus hors échantillon (validation croisée par paiement) pour ne pas biaiser
  la passe 2.
- Passe 2 : features actives + compétition.
- Calibration isotonique des scores de passe 2 sur la période de validation.

Le modèle est sauvegardé avec l'empreinte du journal, la version de la featurisation et les
paramètres, pour être rejouable à l'identique.
"""





GROUP = "group"      # identifiant du paiement dans le jeu (paiement × jour de scoring)


def competition_features(scores: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    """Rang, marge au meilleur concurrent et meilleur score concurrent, au sein de chaque groupe."""
    df = pd.DataFrame({"g": group, "s": scores})
    order = np.lexsort((-df["s"].to_numpy(), df["g"].to_numpy()))
    g, s = df["g"].to_numpy()[order], df["s"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]) if len(g) else np.array([], dtype=np.int64)
    sizes = np.diff(np.r_[starts, len(g)])
    first = np.repeat(starts, sizes)
    rank = np.arange(len(g)) - first + 1
    best = s[first]
    second = np.where(sizes > 1, s[np.minimum(starts + 1, len(g) - 1)], 0.0)
    second = np.repeat(second, sizes)
    best_other = np.where(rank == 1, second, best)
    out = np.empty((len(g), 3), dtype=np.float32)
    out[order, 0] = rank
    out[order, 1] = s - best_other
    out[order, 2] = best_other
    return pd.DataFrame(out, columns=COMPETITION)


def _params(cfg: TrainingSettings) -> dict:
    return {"objective": "binary", "learning_rate": cfg.learning_rate, "num_leaves": cfg.num_leaves,
            "min_data_in_leaf": cfg.min_data_in_leaf, "feature_fraction": 0.9, "seed": cfg.seed,
            "deterministic": True, "force_row_wise": True, "verbosity": -1, "num_threads": 0}


def _train(X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame | None, yv: np.ndarray | None,
           cfg: TrainingSettings) -> lgb.Booster:
    cats = [c for c in CATEGORICAL if c in X.columns]
    train = lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=True)
    valid = [lgb.Dataset(Xv, label=yv, categorical_feature=cats, reference=train)] if Xv is not None else []
    callbacks = [lgb.early_stopping(30, verbose=False)] if valid else []
    return lgb.train(_params(cfg), train, num_boost_round=cfg.num_boost_round, valid_sets=valid, callbacks=callbacks)


@dataclass
class PairModel:
    features: list[str]
    pass1: lgb.Booster
    pass2: lgb.Booster | None
    calibrator: IsotonicRegression | None
    meta: dict = field(default_factory=dict)

    # --- Scoring -------------------------------------------------------------------------------------

    def raw(self, X: pd.DataFrame, group: np.ndarray) -> np.ndarray:
        s1 = self.pass1.predict(X[self.features], num_threads=0)
        if self.pass2 is None:
            return s1
        comp = competition_features(s1, group)
        X2 = pd.concat([X[self.features].reset_index(drop=True), comp], axis=1)
        return self.pass2.predict(X2, num_threads=0)

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    def predict(self, X: pd.DataFrame, group: np.ndarray) -> np.ndarray:
        """Probabilité calibrée que la paire soit une vraie imputation."""
        return self.calibrate(self.raw(X, group))

    # --- Entraînement ----------------------------------------------------------------------------------

    @classmethod
    def fit(cls, train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: TrainingSettings,
            second_pass: bool = True, calibration: bool = True, folds: int = 3) -> PairModel:
        y, yv = train["label"].to_numpy(), valid["label"].to_numpy()
        X, Xv = train[features], valid[features]
        pass1 = _train(X, y, Xv, yv, cfg)
        pass2 = None
        if second_pass:
            # Scores de passe 1 hors échantillon sur l'entraînement (plis par paiement).
            fold = (pd.util.hash_array(train[GROUP].astype(str).to_numpy()) % folds).astype(int)
            oof = np.zeros(len(train))
            fixed = cfg.model_copy(update={"num_boost_round": max(pass1.best_iteration, 10)})
            for k in range(folds):
                m = _train(X[fold != k], y[fold != k], None, None, fixed)
                oof[fold == k] = m.predict(X[fold == k], num_threads=0)
            X2 = pd.concat([X.reset_index(drop=True), competition_features(oof, train[GROUP].to_numpy())], axis=1)
            s1v = pass1.predict(Xv, num_threads=0)
            Xv2 = pd.concat([Xv.reset_index(drop=True), competition_features(s1v, valid[GROUP].to_numpy())], axis=1)
            pass2 = _train(X2, y, Xv2, yv, cfg)
        model = cls(features, pass1, pass2, None)
        if calibration:
            raw_v = model.raw(valid, valid[GROUP].to_numpy())
            model.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_v, yv)
        return model

    # --- Persistance -------------------------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.pass1.save_model(str(d / "pass1.txt"))
        if self.pass2 is not None:
            self.pass2.save_model(str(d / "pass2.txt"))
        if self.calibrator is not None:
            (d / "calibrator.pkl").write_bytes(pickle.dumps(self.calibrator))
        meta = {**self.meta, "features": self.features}
        (d / "model.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> PairModel:
        d = Path(directory)
        meta = json.loads((d / "model.json").read_text(encoding="utf-8"))
        pass2 = lgb.Booster(model_file=str(d / "pass2.txt")) if (d / "pass2.txt").exists() else None
        cal = pickle.loads((d / "calibrator.pkl").read_bytes()) if (d / "calibrator.pkl").exists() else None
        return cls(meta["features"], lgb.Booster(model_file=str(d / "pass1.txt")), pass2, cal, meta)


def pair_metrics(df: pd.DataFrame, score: np.ndarray) -> dict:
    """Diagnostics du scoring : AUC, precision@1, MRR (sur les groupes ayant au moins un positif)."""
    d = pd.DataFrame({"g": df[GROUP].to_numpy(), "y": df["label"].to_numpy(), "s": score})
    d = d.sort_values(["g", "s"], ascending=[True, False], kind="mergesort")
    d["rank"] = d.groupby("g").cumcount() + 1
    pos = d[d["y"] == 1]
    first = pos.groupby("g")["rank"].min()
    return {
        "auc": float(roc_auc_score(d["y"], d["s"])) if d["y"].nunique() == 2 else None,
        "precision_at_1": float((first == 1).mean()) if len(first) else None,
        "mrr": float((1.0 / first).mean()) if len(first) else None,
        "groupes_avec_positif": int(len(first)),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y})
    out = df.groupby("bin").agg(paires=("y", "size"), score_moyen=("p", "mean"), taux_observé=("y", "mean"))
    out.index = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in out.index]
    return out.rename_axis("tranche").reset_index()


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/reconcile_ml/decision.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Étape 5 — ensembles et décision (brief §7.4-7.5).

Pour chaque paiement, à partir des candidats scorés (score brut de passe 2 : même ordre que la
probabilité calibrée, qui est monotone, mais sans les plateaux de la régression isotonique) :
- passe « candidat unique » : les deux meilleurs candidats capables d'absorber le paiement ;
- passe « somme proche » : sous-ensembles des 25 meilleurs candidats dont la somme est à la
  tolérance près du paiement (5 € ou 3 %), cardinalité minimale préférée ; score = moyenne des
  scores de paire ;
- la meilleure proposition l'emporte ; marge = écart au score de la seconde.
La passe n↔n (paiements d'un même débiteur agrégés) est dans le rapprocheur : elle ne produit
que des propositions en revue.

Décision : auto-validation si score ≥ τ_high et marge ≥ δ ; revue si score ≥ τ_low ; sinon rien.
τ_high se calibre sur la validation pour la précision cible, globalement et par segment si le
volume le permet.
"""




PROPOSAL_COLUMNS = ["row", "invoices", "amounts", "score", "margin", "kind"]
MIN_KIND_VOLUME = 20


def propose(scored: pd.DataFrame, amount: np.ndarray, cfg: SetSettings) -> pd.DataFrame:
    """`scored` : (row, inv, balance, p) ; `amount` : montant par ligne. Une proposition par ligne."""
    if scored.empty:
        return pd.DataFrame(columns=PROPOSAL_COLUMNS)
    s = scored.sort_values(["row", "p", "inv"], ascending=[True, False, True], kind="mergesort")
    row, inv = s["row"].to_numpy(), s["inv"].to_numpy()
    bal, p = s["balance"].to_numpy(dtype=np.int64), s["p"].to_numpy(dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
    ends = np.r_[starts[1:], len(row)]
    out = []
    for a, b in zip(starts, ends):
        r = int(row[a])
        amt = int(amount[r])
        tol = max(cfg.tolerance_abs_cents, cfg.tolerance_rel * amt)
        props = []
        # Une facture seule n'est recevable que si elle absorbe le paiement (soldée ou partielle).
        absorbs = amt <= bal[a:b] + tol
        if cfg.single_candidate or not cfg.enabled:
            for j in (a + np.flatnonzero(absorbs))[:2]:
                settle = abs(amt - bal[j]) <= tol
                props.append((p[j], (int(inv[j]),), (int(bal[j]) if settle else amt,),
                              "single" if settle else "partial"))
        # Ensembles recherchés aussi quand la meilleure facture seule laisse un écart : un groupe exact
        # entre alors en concurrence (marge faible → revue plutôt qu'une auto-validation hasardeuse).
        exact_single = absorbs[0] and bal[a] == amt
        if cfg.enabled and cfg.near_sum and b - a >= 2 and not exact_single:
            k = min(b - a, cfg.max_candidates)
            sols, _ = near_subsets(bal[a:a + k], amt, int(tol), max_size=cfg.max_invoices,
                                   node_budget=cfg.node_budget, max_solutions=20)
            ranked = sorted(sols, key=lambda sol: (len(sol), -float(np.mean(p[a:a + k][list(sol)]))))
            exact = [sol for sol in ranked if abs(amt - int(bal[a:a + k][list(sol)].sum())) <= cfg.tolerance_abs_cents]
            for sol in list(dict.fromkeys(ranked[:2] + exact[:1])):
                idx = a + np.array(sol)
                props.append((float(np.mean(p[idx])), tuple(int(i) for i in inv[idx]),
                              tuple(int(x) for x in bal[idx]), "set"))
        if not props:
            continue
        props.sort(key=lambda x: (-x[0], len(x[1])))
        best = props[0]
        others = [q for q in props[1:] if set(q[1]) != set(best[1])]
        margin = best[0] - others[0][0] if others else best[0]
        # Une facture seule qui n'explique le paiement qu'à un écart d'escompte près, face à un ensemble qui
        # l'explique exactement (aux frais près) : ambigu, quel que soit l'écart de score → revue.
        if best[3] == "single" and abs(amt - sum(best[2])) > cfg.tolerance_abs_cents and any(
                q[3] == "set" and abs(amt - sum(q[2])) <= cfg.tolerance_abs_cents for q in props[1:]):
            margin = 0.0
        out.append((r, best[1], best[2], best[0], margin, best[3]))
    return pd.DataFrame(out, columns=PROPOSAL_COLUMNS)


def segment_keys(frame: pd.DataFrame, dims: list[str], edges: list[float] | None) -> np.ndarray:
    """Clé de segment par ligne : valeurs des dimensions jointes (montant découpé en quartiles)."""
    parts = []
    for d in dims:
        if d == "amount_bucket":
            e = edges or []
            parts.append(np.digitize(frame["payment_amount"].to_numpy(dtype=np.float64), e).astype(str))
        elif d in frame.columns:
            parts.append(frame[d].astype(str).to_numpy())
    if not parts:
        return np.full(len(frame), "global", dtype=object)
    key = parts[0].astype(object)
    for p in parts[1:]:
        key = key + "|" + p.astype(object)
    return key


def calibrate_thresholds(proposals: pd.DataFrame, correct: np.ndarray, target: float, cfg: DecisionSettings,
                         amount_edges: list[float] | None = None) -> dict:
    """τ_high : plus petit seuil tenant la précision cible parmi les propositions de marge ≥ δ."""

    def one(scores: np.ndarray, ok: np.ndarray) -> float | None:
        if len(scores) == 0:
            return None
        order = np.argsort(-scores, kind="stable")
        s, c = scores[order], ok[order].astype(np.float64)
        prec = np.cumsum(c) / np.arange(1, len(s) + 1)
        last_of_score = np.r_[s[1:] != s[:-1], True]
        valid = np.flatnonzero((prec >= target) & last_of_score)
        return float(s[valid[-1]]) if len(valid) else None

    eligible = proposals["margin"].to_numpy() >= cfg.min_margin
    scores = proposals["score"].to_numpy(dtype=np.float64)
    global_tau = one(scores[eligible], correct[eligible])
    result = {"tau_high": global_tau if global_tau is not None else 1.01, "tau_low": cfg.review_min_score,
              "min_margin": cfg.min_margin, "kinds": {}, "segments": {}, "segment_dims": [],
              "amount_edges": amount_edges}
    # Un seuil par type de proposition (facture soldée, paiement partiel, ensemble) : leurs fiabilités
    # diffèrent (un partiel peut masquer un n↔n). Sans volume suffisant, pas d'auto-validation.
    kinds = proposals["kind"].to_numpy()
    for kind in np.unique(kinds):
        m = (kinds == kind) & eligible
        tau = one(scores[m], correct[m]) if m.sum() >= MIN_KIND_VOLUME else None
        result["kinds"][str(kind)] = tau if tau is not None else 1.01
    if cfg.segmented_thresholds:
        dims = list(cfg.segments)
        keys = segment_keys(proposals, dims, amount_edges)
        result["segment_dims"] = dims
        for key in np.unique(keys):
            m = (keys == key) & eligible
            if m.sum() >= cfg.min_segment_volume:
                tau = one(scores[m], correct[m])
                result["segments"][str(key)] = tau if tau is not None else 1.01
    return result


def decide(proposals: pd.DataFrame, thresholds: dict) -> np.ndarray:
    """Action par proposition : auto / review / '' (pas de décision)."""
    if proposals.empty:
        return np.array([], dtype=object)
    tau = np.full(len(proposals), thresholds["tau_high"], dtype=np.float64)
    if thresholds.get("kinds"):
        by_kind = proposals["kind"].map(thresholds["kinds"]).to_numpy(dtype=np.float64)
        tau = np.where(np.isnan(by_kind), 1.01, by_kind)
    if thresholds.get("segments"):
        keys = segment_keys(proposals, thresholds["segment_dims"], thresholds.get("amount_edges"))
        seg = pd.Series(keys).map(thresholds["segments"]).to_numpy(dtype=np.float64)
        tau = np.where(np.isnan(seg), tau, seg)
    score, margin = proposals["score"].to_numpy(), proposals["margin"].to_numpy()
    auto = (score >= tau) & (margin >= thresholds["min_margin"])
    review = ~auto & (score >= thresholds["tau_low"])
    return np.where(auto, "auto", np.where(review, "review", ""))


def calibrate_online(records: pd.DataFrame, correct: np.ndarray, target: float, cfg: DecisionSettings,
                     grid: int = 2000) -> dict:
    """Seuils calibrés sur la boucle réelle de validation (propositions quotidiennes enregistrées).

    `records` : une ligne par (jour, paiement) — kind, score, margin — dans l'ordre des jours ;
    `correct` : la proposition du jour est-elle exactement la vérité ? En production, un paiement
    en attente est rescoré chaque jour : il est auto-validé le premier jour où son score franchit τ.
    Pour τ donné, la proposition retenue est donc celle du jour qui porte, pour la première fois,
    le maximum courant du score au-dessus de τ. Calibration par type de proposition.
    """
    result = {"tau_high": 1.01, "tau_low": cfg.review_min_score, "min_margin": cfg.min_margin, "kinds": {},
              "segments": {}, "segment_dims": [], "amount_edges": None, "method": "online"}
    r = records.assign(correct=correct)
    for kind in sorted(r["kind"].unique()):
        k = r[(r["kind"] == kind)].copy()
        k["s"] = np.where(k["margin"] >= cfg.min_margin, k["score"], -np.inf)
        k = k.sort_values(["payment_id", "day"], kind="mergesort")
        prev = k.groupby("payment_id")["s"].transform(lambda v: v.cummax().shift(fill_value=-np.inf)).to_numpy()
        s = k["s"].to_numpy()
        setter = s > prev                                  # jours qui portent un nouveau maximum
        s, prev, ok = s[setter], prev[setter], k["correct"].to_numpy()[setter]
        if len(s) < MIN_KIND_VOLUME:
            result["kinds"][kind] = 1.01
            continue
        taus = np.unique(s[np.isfinite(s)])[::-1]
        if len(taus) > grid:
            taus = taus[np.unique(np.linspace(0, len(taus) - 1, grid).round().astype(int))]
        best = None
        for tau in taus:                                   # du plus strict au plus permissif
            m = (prev < tau) & (tau <= s)
            if m.sum() >= MIN_KIND_VOLUME and ok[m].mean() >= target:
                best = float(tau)
        result["kinds"][kind] = best if best is not None else 1.01
    return result


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/reconcile_ml/pipeline.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Étape 5 — rapprocheur complet (règles puis ML) et construction / entraînement (brief §7).

- `PipelineMatcher` : étape 4 (règles), puis sur le résiduel du jour : candidats, features,
  scoring, ensembles, n↔n (revue), décision par seuils, arbitrage du lot. Les réservations du
  moteur sont partagées entre règles et ML.
- `DatasetRecorder` : même boucle, mais enregistre les features des candidats de chaque paiement
  du résiduel à son jour d'arrivée (état avant les événements du jour), sans décider en ML.
- `fit_ml` : rejoue entraînement + validation, étiquette dans une seconde passe séparée à partir
  des imputations, entraîne, calibre, fixe τ_high et sauvegarde le modèle avec l'empreinte du
  journal et la version de la featurisation.
"""





MODEL_DIRNAME = "pair_model"

# Paires dont les features sont calculées ensemble : borne les intermédiaires d'un jour.
FEATURE_BLOCK_PAIRS = 500_000


def _sampled(payment_ids: np.ndarray, share: float) -> np.ndarray:
    """Échantillon déterministe de paiements (hachage de l'identifiant)."""
    if share >= 1:
        return np.ones(len(payment_ids), dtype=bool)
    h = pd.util.hash_array(np.asarray(payment_ids, dtype=object)) % 10_000
    return h < share * 10_000


class _ResidualMixin(RulesMatcher):
    """Candidats et features du résiduel des règles."""

    def _setup_ml(self, settings: Settings, categories: dict | None = None) -> None:
        self.ml = settings.reconcile_ml
        self.featurizer = Featurizer(self.state, self.allocator, self.ml, self.min_key_length, categories)
        self._pay_value = _days(self.state.table("payment")["value_date"])

    def _residual(self, ctx: DayContext, rules_decisions: pd.DataFrame, rows_filter=None):
        batch_ids = ctx.batch["payment_id"].astype(object).to_numpy()
        pos = self.state.pay_pos(pd.Series(batch_ids, dtype=object))
        decided = np.isin(batch_ids, rules_decisions["payment_id"].astype(object).to_numpy())
        keep = ~decided if rows_filter is None else (~decided & rows_filter)
        rows = np.flatnonzero(keep)
        alloc = self.last_allocation
        scope = self._scopes(alloc, batch_ids)[0]
        memory.mark("ML · candidats")
        cands, cited = self.featurizer.candidates(rows, pos, alloc, scope, ctx.as_of, self._claimed)
        memory.mark(f"ML · features ({len(cands)} paires)")
        X = self._features(cands, pos, alloc, batch_ids, ctx.as_of, cited) if len(cands) else None
        return batch_ids, pos, alloc, cands, X

    def _features(self, cands, pos, alloc, batch_ids, as_of, cited) -> pd.DataFrame:
        """Features par blocs de paiements entiers (≈ `FEATURE_BLOCK_PAIRS` paires) : mêmes valeurs,
        intermédiaires bornés. Les candidats sont triés par paiement."""
        row = cands["row"].to_numpy()
        if len(row) <= FEATURE_BLOCK_PAIRS:
            return self.featurizer.features(cands, pos, alloc, batch_ids, as_of, cited)
        starts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
        cuts = np.unique(starts[np.searchsorted(starts, np.arange(0, len(row), FEATURE_BLOCK_PAIRS), side="right") - 1])
        bounds = np.r_[cuts, len(row)]
        parts = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            block = cands.iloc[a:b]
            rows = np.unique(block["row"].to_numpy())
            sub_cited = cited[cited["row"].isin(rows)] if cited is not None and len(cited) else cited
            parts.append(self.featurizer.features(block.reset_index(drop=True), pos, alloc, batch_ids, as_of,
                                                  sub_cited))
        return pd.concat(parts, ignore_index=True)


class DatasetRecorder(_ResidualMixin):
    """Rejoue les règles et enregistre les paires candidates du résiduel (jour d'arrivée)."""

    name = "dataset"

    def __init__(self, state: LedgerState, settings: Settings, rules: RulesConfig):
        super().__init__(state, settings, rules)
        self._setup_ml(settings)
        self.features = active_features(self.ml)
        self._frames: list[pd.DataFrame] = []
        self._pairs = 0

    def process(self, ctx: DayContext) -> pd.DataFrame:
        decisions = super().process(ctx)
        ids = ctx.batch["payment_id"].astype(object).to_numpy()
        rows_filter = ctx.batch["is_new"].to_numpy() & _sampled(ids, self.ml.training.payment_sample)
        batch_ids, pos, alloc, cands, X = self._residual(ctx, decisions, rows_filter)
        if X is not None and len(X):
            frame = X[self.features].copy()
            frame["payment_id"] = batch_ids[cands["row"].to_numpy()]
            frame["invoice_id"] = self._inv_ids[cands["inv"].to_numpy()]
            frame["balance"] = cands["balance"].to_numpy()
            frame["payment_amount"] = self._pay_amount[pos[cands["row"].to_numpy()]]
            frame["day"] = ctx.day
            self._frames.append(frame)
            self._pairs += len(frame)
            memory.mark(f"jeu d'entraînement : {self._pairs} paires cumulées")
        return decisions

    def dataset(self) -> pd.DataFrame:
        return pd.concat(self._frames, ignore_index=True) if self._frames else pd.DataFrame()


class PipelineMatcher(_ResidualMixin):
    """Étapes 4 puis 5 dans la boucle quotidienne."""

    name = "pipeline"

    def __init__(self, state: LedgerState, settings: Settings, rules: RulesConfig, model: PairModel,
                 record_proposals: bool = False):
        """`record_proposals` : mode calibration — aucune auto-validation ML, propositions quotidiennes
        enregistrées (`daily_proposals`) pour calibrer les seuils sur la boucle réelle."""
        super().__init__(state, settings, rules)
        self._setup_ml(settings, model.meta.get("categories"))
        self.model = model
        self.record_proposals = record_proposals
        self.thresholds = model.meta.get("thresholds") if not record_proposals else             {"tau_high": 2.0, "tau_low": 0.0, "min_margin": 0.0, "kinds": {}, "segments": {}}
        self._daily: list[pd.DataFrame] = []
        self._diag: list[pd.DataFrame] = []
        self._info: list[pd.DataFrame] = []
        self._seen: set[str] = set()

    def process(self, ctx: DayContext) -> pd.DataFrame:
        rules_decisions = super().process(ctx)
        batch_ids, pos, alloc, cands, X = self._residual(ctx, rules_decisions)
        self._record_info(ctx, alloc, batch_ids)
        if X is None or not len(X):
            return rules_decisions
        amount = self._pay_amount[pos]
        memory.mark(f"ML · score ({len(X)} paires)")
        raw = self.model.raw(X, cands["row"].to_numpy())
        scored = cands.assign(p=raw, p_cal=self.model.calibrate(raw))
        self._record_diag(scored, batch_ids)

        memory.mark("ML · propositions")
        props = propose(scored, amount, self.ml.sets)
        props = self._segment_columns(props, pos, alloc, X, cands)
        if self.record_proposals and len(props):
            self._daily.append(pd.DataFrame({
                "day": ctx.day, "payment_id": batch_ids[props["row"].to_numpy()], "kind": props["kind"].to_numpy(),
                "score": props["score"].to_numpy(), "margin": props["margin"].to_numpy(),
                "invoice_ids": [tuple(sorted(self._inv_ids[list(i)])) for i in props["invoices"]]}))
        actions = decide(props, self.thresholds) if len(props) else np.array([], dtype=object)
        props = props.assign(action=actions)
        props = props[props["action"] != ""]
        props = self._arbitrate_ml(props, ctx.as_of)
        nn = self._n_to_n(scored, props, alloc, pos, amount) if self.ml.sets.enabled and self.ml.sets.n_to_n \
            else pd.DataFrame()
        ml_decisions = self._to_decisions(props, nn, batch_ids)
        return pd.concat([rules_decisions, ml_decisions], ignore_index=True) if len(ml_decisions) else rules_decisions

    # --- Étapes ------------------------------------------------------------------------------------------

    def _segment_columns(self, props, pos, alloc, X, cands) -> pd.DataFrame:
        if props.empty:
            return props
        rows = props["row"].to_numpy()
        first_inv = np.array([inv[0] for inv in props["invoices"]], dtype=np.int64)
        market = pd.Series(self.featurizer.inv_market[first_inv]).astype(int).astype(str).to_numpy()
        files = alloc.payments["client_file_id"].to_numpy(dtype=object)
        return props.assign(payment_amount=self._pay_amount[pos[rows]], market=market,
                            has_client_file=pd.notna(files[rows]).astype(int),
                            bankroll_code=self.featurizer.pay_bankroll[pos[rows]].astype(int))

    def _arbitrate_ml(self, props: pd.DataFrame, as_of) -> pd.DataFrame:
        """Auto-validations ML par score décroissant ; une facture déjà soldée ramène la proposition en revue."""
        autos = props[props["action"] == AUTO].sort_values("score", ascending=False, kind="mergesort")
        if autos.empty:
            return props
        invs = np.unique(np.concatenate([np.array(i, dtype=np.int64) for i in autos["invoices"]]))
        remaining = dict(zip(invs.tolist(), (self.state.open_balance_at(invs, as_of) - self._claimed[invs]).tolist()))
        downgrade = []
        for idx, inv, amt in zip(autos.index, autos["invoices"], autos["amounts"]):
            if all(remaining[i] >= a for i, a in zip(inv, amt)):
                for i, a in zip(inv, amt):
                    remaining[i] -= a
            else:
                downgrade.append(idx)
        props = props.copy()
        props.loc[downgrade, "action"] = REVIEW
        return props

    def _n_to_n(self, scored: pd.DataFrame, props: pd.DataFrame, alloc, pos, amount) -> pd.DataFrame:
        """Paiements non auto-validés d'un même débiteur (allocation ferme) proches dans le temps : revue."""
        cfg = self.ml.sets
        firm = alloc.payments["firm_debtor_id"].to_numpy(dtype=object)
        auto_rows = set(props.loc[props["action"] == AUTO, "row"].tolist())
        rows = np.array([r for r in np.unique(scored["row"].to_numpy()) if r not in auto_rows and pd.notna(firm[r])],
                        dtype=np.int64)
        if len(rows) < 2:
            return pd.DataFrame()
        frame = pd.DataFrame({"row": rows, "debtor": firm[rows], "day": self._pay_value[pos[rows]]})
        window = max(cfg.n_to_n_window_hours // 24, 1)
        out = []
        for _, g in frame.groupby("debtor"):
            if len(g) < 2 or g["day"].max() - g["day"].min() > window:
                continue
            pool = scored[scored["row"].isin(g["row"])].drop_duplicates("inv")
            pool = pool.sort_values("p", ascending=False).head(cfg.max_candidates)
            target = int(amount[g["row"].to_numpy()].sum())
            tol = int(max(cfg.tolerance_abs_cents, cfg.tolerance_rel * target))
            sols, _ = near_subsets(pool["balance"].to_numpy(), target, tol, max_size=cfg.max_invoices * 2,
                                   min_size=2, node_budget=cfg.node_budget, max_solutions=2)
            if len(sols) != 1:
                continue
            sel = pool.iloc[list(sols[0])]
            for r in g["row"]:
                out.append({"row": int(r), "invoices": tuple(sel["inv"].tolist()), "score": float(sel["p"].mean())})
        return pd.DataFrame(out)

    def _to_decisions(self, props: pd.DataFrame, nn: pd.DataFrame, batch_ids: np.ndarray) -> pd.DataFrame:
        rows = []
        for r, inv, amt, score, action, kind in zip(props["row"], props["invoices"], props["amounts"], props["score"],
                                                   props["action"], props["kind"]):
            for i, a in zip(inv, amt):
                rows.append((batch_ids[r], self._inv_ids[i], a, action, "ml", f"ML_{kind.upper()}", score))
            if action == AUTO:
                pid = int(self.state.pay_pos(pd.Series([batch_ids[r]], dtype=object))[0])
                self._claims.setdefault(pid, []).extend(zip(inv, amt))
                np.add.at(self._claimed, np.array(inv, dtype=np.int64), np.array(amt, dtype=np.int64))
        done = {r for r, a in zip(props["row"], props["action"]) if a == AUTO}
        for rec in nn.to_dict("records") if len(nn) else []:
            if rec["row"] in done:
                continue
            for i in rec["invoices"]:
                rows.append((batch_ids[rec["row"]], self._inv_ids[i], None, REVIEW, "ml", "ML_NN", rec["score"]))
        if not rows:
            return pd.DataFrame(columns=DECISION_COLUMNS)
        df = pd.DataFrame(rows, columns=["payment_id", "invoice_id", "amount", "action", "step", "rule_id", "score"])
        df["amount"] = pd.array(df["amount"], dtype="Int64")
        df["rule_version"] = pd.array([None] * len(df), dtype="Int64")
        return df[DECISION_COLUMNS]

    # --- Diagnostics pour l'évaluation -----------------------------------------------------------------------

    def _record_info(self, ctx, alloc, batch_ids) -> None:
        new = ctx.batch["is_new"].to_numpy()
        self._info.append(pd.DataFrame({"payment_id": batch_ids[new],
                                        "client_file_id": alloc.payments["client_file_id"].to_numpy()[new],
                                        "allocation_status": alloc.payments["status"].to_numpy()[new]}))

    def _record_diag(self, scored: pd.DataFrame, batch_ids: np.ndarray) -> None:
        """Candidats et scores du premier passage en ML de chaque paiement."""
        s = scored.assign(payment_id=batch_ids[scored["row"].to_numpy()])
        s = s[~s["payment_id"].isin(self._seen)]
        if s.empty:
            return
        self._seen.update(s["payment_id"].unique().tolist())
        self._diag.append(pd.DataFrame({"payment_id": s["payment_id"].to_numpy(),
                                        "invoice_id": self._inv_ids[s["inv"].to_numpy()], "p": s["p_cal"].to_numpy(),
                                        "raw": s["p"].to_numpy()}))

    def daily_proposals(self) -> pd.DataFrame:
        return pd.concat(self._daily, ignore_index=True) if self._daily else pd.DataFrame()

    def side_outputs(self) -> dict[str, pd.DataFrame]:
        out = {"proposals": self.proposals()}
        if self._diag:
            out["ml_candidates"] = pd.concat(self._diag, ignore_index=True)
        if self._info:
            out["payment_info"] = pd.concat(self._info, ignore_index=True).drop_duplicates("payment_id")
        return out


# --- Entraînement ----------------------------------------------------------------------------------------------


def label_pairs(dataset: pd.DataFrame, imputation: pd.DataFrame) -> np.ndarray:
    """Seconde passe, séparée du rejeu : la paire figure-t-elle dans les imputations ?"""
    truth = imputation[["payment_id", "invoice_id"]].drop_duplicates().assign(_hit=1)
    merged = dataset[["payment_id", "invoice_id"]].merge(truth, on=["payment_id", "invoice_id"], how="left")
    return merged["_hit"].fillna(0).to_numpy(dtype=np.int8)


def candidate_recall(dataset: pd.DataFrame, truth: pd.DataFrame) -> float | None:
    """Part des paiements du jeu dont toutes les factures réellement imputées sont candidates."""
    t = truth.set_index("payment_id")["truth_invoices"]
    got = dataset.groupby("payment_id")["invoice_id"].agg(set)
    common = got.index.intersection(t.index)
    if not len(common):
        return None
    return float(np.mean([set(t[p]) <= got[p] for p in common]))


def offline_proposals(frame: pd.DataFrame, scores: np.ndarray, settings: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """Propositions reconstruites hors boucle sur un jeu étiqueté (calibration des seuils)."""
    pay_codes, pay_ids = pd.factorize(frame["payment_id"])
    inv_codes, inv_ids = pd.factorize(frame["invoice_id"])
    scored = pd.DataFrame({"row": pay_codes, "inv": inv_codes, "balance": frame["balance"].to_numpy(), "p": scores})
    amount = frame.groupby(pay_codes)["payment_amount"].first().to_numpy()
    props = propose(scored, amount, settings.reconcile_ml.sets)
    props["payment_id"] = pay_ids[props["row"].to_numpy()]
    props["invoice_ids"] = [tuple(sorted(inv_ids[list(i)])) for i in props["invoices"]]
    first = frame.groupby(pay_codes).first()
    props["payment_amount"] = amount[props["row"].to_numpy()]
    for col in ("market", "has_client_file", "bankroll_code"):
        props[col] = first[col].to_numpy()[props["row"].to_numpy()].astype(int).astype(str) if col in first else "0"
    return props, pd.Series(inv_ids)


def fit_ml(interim_dir: Path, model_dir: Path, settings: Settings, rules: RulesConfig,
           log: Callable[[str], None] = print) -> dict:
    """Rejoue entraînement + validation, entraîne le modèle, calibre les seuils, sauvegarde."""
    timings = {}
    t = time.perf_counter()
    data, journal, meta = load_interim(interim_dir)
    state = LedgerState(data, journal, settings.reconcile_ml.features.behavioral_window_days)
    first, last = state.payment_day_range()
    split = compute_split(settings.split, first.date(), last.date())
    train_p, valid_p = split.period("train"), split.period("validation")
    timings["préparation"] = round(time.perf_counter() - t, 1)

    t = time.perf_counter()
    recorder = DatasetRecorder(state, settings, rules)
    log(f"… rejeu {train_p.start} → {valid_p.end} pour construire le jeu d'entraînement")
    run_replay(state, recorder, train_p.start, valid_p.end, settings.split.retention_days,
               on_day=lambda ctx, row: log(f"  {ctx.day.date()}") if ctx.day.day == 1 else None)
    ds = recorder.dataset()
    timings["construction du jeu"] = round(time.perf_counter() - t, 1)
    if ds.empty:
        raise RuntimeError("jeu d'entraînement vide")

    ds["label"] = label_pairs(ds, data.tables["imputation"])
    ds[GROUP] = ds["payment_id"]
    ds["period"] = assign_period(ds["day"], split).to_numpy()
    ds["month"] = pd.to_datetime(ds["day"]).dt.strftime("%Y-%m")
    out_dir = interim_dir / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out_dir / "dataset", partition_cols=["month"], index=False)
    train, valid = ds[ds["period"] == "train"], ds[ds["period"] == "validation"]
    log(f"  jeu : {len(train):,} paires d'entraînement, {len(valid):,} de validation".replace(",", " "))

    t = time.perf_counter()
    features = recorder.features
    model = PairModel.fit(train, valid, features, settings.reconcile_ml.training,
                          settings.reconcile_ml.second_pass, settings.reconcile_ml.calibration)
    timings["entraînement"] = round(time.perf_counter() - t, 1)

    raw_valid = model.raw(valid, valid[GROUP].to_numpy())
    p_valid = model.calibrate(raw_valid)
    truth = ground_truth(data.tables["imputation"])
    props, _ = offline_proposals(valid, raw_valid, settings)
    t_map = truth.set_index("payment_id")["truth_invoices"]
    correct = np.array([t_map.get(pid) == inv for pid, inv in zip(props["payment_id"], props["invoice_ids"])])
    edges = list(np.quantile(valid.groupby("payment_id")["payment_amount"].first(), [0.25, 0.5, 0.75]))
    offline = calibrate_thresholds(props, correct, settings.evaluation.target_precision,
                                   settings.reconcile_ml.decision, edges)
    auto = pd.Series(decide(props, offline) == AUTO)

    # Calibration sur la boucle réelle de validation : propositions quotidiennes, sans auto-validation ML.
    t = time.perf_counter()
    model.meta = {"thresholds": offline, "categories": recorder.featurizer.categories}
    thresholds = online_thresholds(data, journal, model, settings, rules, valid_p, truth, log, out_dir)
    thresholds["offline"] = {k: offline[k] for k in ("tau_high", "kinds")}
    timings["calibration en ligne"] = round(time.perf_counter() - t, 1)
    metrics = {
        "validation": pair_metrics(valid, p_valid),
        "rappel_candidats_validation": candidate_recall(valid, truth),
        "résiduel_validation_paiements": int(valid["payment_id"].nunique()),
        "auto_validation": int(auto.sum()),
        "précision_auto_validation": float(correct[auto.to_numpy()].mean()) if auto.any() else None,
    }
    model.meta = {
        "journal_sha256": meta["journal_sha256"], "featurization_version": FEATURIZATION_VERSION,
        "rules_version": rules.version, "settings_fingerprint": fingerprint(settings.reconcile_ml.model_dump()),
        "settings": settings.reconcile_ml.model_dump(), "periods": {
            "train": [str(train_p.start), str(train_p.end)], "validation": [str(valid_p.start), str(valid_p.end)]},
        "thresholds": thresholds, "metrics": metrics, "timings_s": timings,
        "categories": recorder.featurizer.categories,
    }
    model.save(model_dir)
    calibration_table(valid["label"].to_numpy(), p_valid).to_csv(model_dir / "calibration_validation.csv", index=False)
    importance = pd.DataFrame({"feature": model.pass1.feature_name(),
                               "gain": model.pass1.feature_importance("gain")}).sort_values("gain", ascending=False)
    importance.to_csv(model_dir / "feature_importance.csv", index=False)
    return model.meta


def online_thresholds(data, journal, model: PairModel, settings: Settings, rules: RulesConfig, valid_p, truth,
                      log: Callable[[str], None], out_dir: Path) -> dict:
    """Rejoue la validation sans auto-validation ML, enregistre les propositions quotidiennes et en déduit τ_high
    par type de proposition (précision réelle de la boucle, rescorage quotidien compris)."""
    log(f"… rejeu de la validation pour calibrer les seuils ({valid_p.start} → {valid_p.end})")
    state = LedgerState(data, journal, settings.reconcile_ml.features.behavioral_window_days)
    probe = PipelineMatcher(state, settings, rules, model, record_proposals=True)
    run_replay(state, probe, valid_p.start, valid_p.end, settings.split.retention_days)
    daily = probe.daily_proposals()
    if daily.empty:
        return dict(model.meta.get("thresholds") or {})
    t_map = truth.set_index("payment_id")["truth_invoices"]
    ok = np.array([t_map.get(pid) == inv for pid, inv in zip(daily["payment_id"], daily["invoice_ids"])])
    daily.assign(correct=ok, invoice_ids=daily["invoice_ids"].map(list)).to_parquet(
        out_dir / "validation_daily_proposals.parquet", index=False)
    return calibrate_online(daily, ok, settings.evaluation.target_precision, settings.reconcile_ml.decision)


def recalibrate_ml(interim_dir: Path, model_dir: Path, settings: Settings, rules: RulesConfig,
                   log: Callable[[str], None] = print) -> dict:
    """Recalcule les seuils d'un modèle existant (après un changement de décision ou de cible), sans réentraîner."""
    data, journal, meta = load_interim(interim_dir)
    model = PairModel.load(model_dir)
    if model.meta.get("journal_sha256") != meta["journal_sha256"]:
        raise RuntimeError("modèle entraîné sur un autre journal : réentraîner")
    state = LedgerState(data, journal, settings.reconcile_ml.features.behavioral_window_days)
    first, last = state.payment_day_range()
    valid_p = compute_split(settings.split, first.date(), last.date()).period("validation")
    t = time.perf_counter()
    offline = model.meta["thresholds"].get("offline", {})
    thresholds = online_thresholds(data, journal, model, settings, rules, valid_p, ground_truth(data.tables["imputation"]),
                                   log, interim_dir / "ml")
    thresholds["offline"] = offline
    model.meta["thresholds"] = thresholds
    model.meta.setdefault("timings_s", {})["recalibration"] = round(time.perf_counter() - t, 1)
    model.save(model_dir)
    return model.meta


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# src/api.py
# ════════════════════════════════════════════════════════════════════════════════════════════════════

"""Point d'entrée unique de la pipeline, pour le notebook et l'interface.

    from reconciliation import Project
    project = Project(dataset="synthetic")          # ou "real" (sources de config/schema.yaml)
    project.load(n_payments=50_000)                 # étape 1
    project.split()                                 # étape 2 : périodes
    project.measure_allocation("validation")        # étape 3
    project.backtest("test", matcher="rules")       # étape 4 : baseline
    project.train()                                 # étape 5 : modèle
    project.backtest("test", matcher="pipeline")    # étapes 4 + 5 → étape 6
    project.report("pipeline", "test")              # résultats

Suivi mémoire : `Project(log=MemoryMonitor(...).start().log)` (voir src/memory.py).

Chaque méthode écrit ses sorties sur disque (paths de config/settings.yaml) et renvoie des
objets affichables (DataFrame, dict). Les paramètres se lisent dans config/settings.yaml et
config/rules.yaml, éditables à la main ou depuis l'interface.
"""





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
        return resolve_path(self.settings.paths.models_dir) / self.dataset / MODEL_DIRNAME

    def meta(self) -> dict | None:
        path = self.interim_dir / "journal_meta.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @contextmanager
    def _step(self, name: str, timings: dict[str, float]):
        start = time.perf_counter()
        memory.mark_step(name)
        self.log(f"… {name}")
        yield
        timings[name] = round(time.perf_counter() - start, 1)
        self.log(f"  {name} : {timings[name]} s")

    # --- Étape 1 : chargement ---------------------------------------------------------------------------------

    def load(self, n_payments: int | None = None, seed: int = 42, regenerate: bool = False) -> dict:
        """Charge les sources (ou génère le jeu synthétique), normalise, construit le journal et le profil."""

        settings = self.settings
        timings: dict[str, float] = {}
        if self.dataset == SYNTHETIC:
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
        data, journal, meta = load_interim(self.interim_dir)
        state = LedgerState(data, journal, self.settings.reconcile_ml.features.behavioral_window_days)
        return data, state, meta

    def split(self):
        """Périodes train / validation / test sur la plage des paiements chargés."""
        pay = pd.read_parquet(self.interim_dir / "payment.parquet", columns=["value_date", "booking_date"])
        days = pay["booking_date"].fillna(pay["value_date"])
        return compute_split(self.settings.split, days.min().date(), days.max().date())

    def _period(self, state, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        first, last = state.payment_day_range()
        if period == "all":
            return first, last
        p = compute_split(self.settings.split, first.date(), last.date()).period(period)
        return pd.Timestamp(p.start), pd.Timestamp(p.end)

    def _matcher(self, name: str, state, settings: Settings):
        if name == "null":
            return NullMatcher()
        if name == "rules":
            return RulesMatcher(state, settings, self.rules)
        if name == "pipeline":
            if not (self.model_dir / "model.json").exists():
                raise FileNotFoundError("aucun modèle entraîné : lancer d'abord project.train()")
            return PipelineMatcher(state, settings, self.rules, PairModel.load(self.model_dir))
        raise ValueError(f"rapprocheur inconnu : {name} ({', '.join(MATCHERS)})")

    def replay(self, period: str = "test", matcher: str = "null") -> dict:
        """Rejoue la période jour par jour avec un rapprocheur ; écrit décisions et statistiques quotidiennes."""
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
        return fit_ml(self.interim_dir, self.model_dir, self.settings, self.rules, log=self.log)

    def calibrate(self) -> dict:
        """Recalcule les seuils de décision du modèle existant sur la validation, sans réentraîner."""
        return recalibrate_ml(self.interim_dir, self.model_dir, self.settings, self.rules, log=self.log)

    # --- Étape 6 : évaluation -----------------------------------------------------------------------------------------

    def evaluate(self, period: str = "test", matcher: str = "null") -> dict:
        """Compare les décisions d'un rejeu aux imputations réelles ; écrit les rapports."""
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

# ════════════════════════════════════════════════════════════════════════════════════════════════════
# Configuration par défaut, écrite au premier import si absente
# ════════════════════════════════════════════════════════════════════════════════════════════════════

_EMBEDDED_FILES = {
    'config/settings.yaml': '''# Paramètres de la pipeline, par étape. Édité à la main, depuis le notebook ou par l'UI (streamlit run reconciliation_ui.py).
# Les règles de l'étape 4 sont dans config/rules.yaml.

# Chemins.
paths:
  # Tables normalisées et journal produits par l'étape 1.
  interim_dir: data/interim
  # Rapports CSV et markdown.
  reports_dir: reports
  # Jeux synthétiques, un sous-dossier par volume et seed.
  synthetic_dir: data/synthetic
  # Modèles de l'étape 5 (un sous-dossier par jeu de données).
  models_dir: models

# Étape 1 — chargement.
load:
  # Processus pour la normalisation des gros volumes (1 = séquentiel). Sans effet sur le résultat.
  workers: 12

# Étape 2 — découpage temporel.
split:
  # Durée de la période de test (backtest), en mois : les derniers mois de l'historique.
  test_months: 2
  # Durée de la période de validation, en mois, juste avant le test.
  validation_months: 2
  # Durée de la période d'entraînement, en mois. null = tout l'historique disponible avant la
  # validation.
  train_months: null
  # Jours exclus entre deux périodes (purge), pour éviter qu'un groupe d'imputations chevauche
  # deux blocs.
  purge_days: 5
  # Dernier jour de la période de test. null = dernier jour de paiement présent dans les données.
  anchor_end: null
  # Durée pendant laquelle un paiement non résolu reste dans le reliquat retraité chaque jour.
  retention_days: 60

# Étape 3 — allocation.
allocation:
  # Signaux d'allocation, du plus fort au plus faible.
  signals:
    # Signal 1 (le plus fort).
    client_file:
      # Rattacher le paiement au client file correspondant (référence de virement, montant total,
      # date, IBAN).
      enabled: true
      # Écart maximal entre la date du client file et la date de valeur du paiement pour
      # corroborer le rattachement.
      date_tolerance_days: 5
    # Signal 2.
    reference:
      # Débiteur de la facture dont la référence figure dans le libellé.
      enabled: true
      # Longueur minimale d'une clé de référence utilisée (les clés courtes sont du bruit :
      # années, numéros de rue).
      min_key_length: 4
      # Longueur minimale d'une clé désignant un seul débiteur pour constituer un signal fort
      # (allocation ferme).
      strong_min_key_length: 6
      # Une clé partagée par plus de débiteurs (factures existantes à D) est ignorée.
      max_debtors_per_key: 5
    # Signal 3.
    iban:
      # Routage IBAN (DEBTOR_DIRECT / ASSIGNOR / TECHNICAL_ACCOUNT / UNKNOWN) arbitré par
      # bankroll_code.
      enabled: true
    # Signal 4.
    name:
      # Similarité entre le libellé et le nom du débiteur (index inversé).
      enabled: true
      # Score de nom minimal : part du nom retrouvée dans le libellé (pondérée par la rareté des
      # mots, bigrammes inclus), ou spécificité du meilleur mot (1 / nombre de débiteurs le
      # portant). À calibrer.
      min_similarity: 0.5
      # Élagage : un mot présent dans plus que cette part des noms de débiteurs est ignoré. À
      # calibrer.
      max_token_share: 0.01
      # Élagage absolu : un mot partagé par plus de débiteurs est ignoré, quelle que soit leur
      # part.
      max_debtors_per_term: 200
      # Longueur minimale d'un mot indexé.
      min_token_length: 3
    # Signal 5 (le plus faible).
    amount:
      # Débiteurs ayant une facture ouverte à D dont le restant dû égale exactement le montant du
      # paiement. Extension du brief (clé K3 de la spec).
      enabled: true
      # Un montant partagé par plus de débiteurs (factures ouvertes à D) est ignoré, sauf
      # confirmation par le nom.
      max_debtors_per_amount: 3
      # Jusqu'à ce nombre de débiteurs partageant le montant, ceux dont un mot du nom (même
      # fréquent) figure dans le libellé sont retenus.
      max_debtors_with_name_hint: 30
  # Taille maximale de la liste classée de débiteurs candidats transmise aux étapes suivantes.
  max_candidates: 10
  # Rappel d'allocation visé : le vrai débiteur doit figurer dans la liste. Bloquant pour passer à
  # l'étape 4.
  target_recall: 0.99

# Étape 5 — réconciliation ML.
reconcile_ml:
  # Génération de candidats.
  candidates:
    # Factures ouvertes des débiteurs candidats de l'allocation.
    allocated_debtors: true
    # Factures dont la référence figure dans le libellé, sans fenêtre temporelle.
    reference_no_window: true
    # Factures dont le montant ouvert égale le paiement.
    amount_exact: true
    # Fenêtre ± jours de la clé montant.
    amount_window_days: 90
    # Factures citées dans le client file rattaché.
    client_file_cited: true
    # Clé débiteur : échéances jusqu'à ce nombre de jours avant la date de valeur.
    debtor_window_before_days: 180
    # Clé débiteur : échéances jusqu'à ce nombre de jours après la date de valeur.
    debtor_window_after_days: 30
    # Clé débiteur : factures des N premiers débiteurs de l'allocation seulement (les gros
    # débiteurs ont des milliers de factures ouvertes).
    max_debtors: 3
    # Candidats maximum par paiement : clés référence, montant et client file d'abord, puis
    # factures du débiteur les plus proches en montant ou en échéance.
    max_per_payment: 50
  # Familles de features actives.
  features:
    # Montant : écarts, escompte, frais bancaires, retenue de garantie.
    amount: true
    # Temporel : délai à l'échéance, ancienneté de la facture.
    temporal: true
    # Textuel : référence exacte / partielle, similarité de nom.
    textual: true
    # Identité : route IBAN, bankroll, canal, contrat.
    identity: true
    # Comportemental : agrégats du débiteur, fenêtre strictement antérieure.
    behavioral: true
    # Fenêtre glissante des agrégats comportementaux.
    behavioral_window_days: 180
    # Contexte contrat : market, product, recourse.
    contract: true
    # Score et signal d'allocation du débiteur.
    allocation: true
    # Présence et concordance du client file.
    client_file: true
  # Deuxième passe avec features de compétition (rang, marge au second, nombre de candidats).
  second_pass: true
  # Calibration isotonique des scores sur la validation.
  calibration: true
  # Entraînement.
  training:
    # Part des paiements du résiduel conservés pour l'entraînement (tous leurs candidats sont
    # gardés : pas d'échantillonnage des négatifs).
    payment_sample: 0.5
    # Nombre maximal d'arbres (arrêt précoce sur la validation).
    num_boost_round: 400
    # Taux d'apprentissage LightGBM.
    learning_rate: 0.05
    # Feuilles par arbre.
    num_leaves: 63
    # Observations minimales par feuille.
    min_data_in_leaf: 50
    # Graine (reproductibilité).
    seed: 42
  # Résolution des ensembles.
  sets:
    # Reconstitution des ensembles (1↔n, n↔n) par DFS borné.
    enabled: true
    # Passe 1 : sous-ensemble dont la somme est proche du paiement.
    near_sum: true
    # Passe 2 : agrégation des paiements d'un même débiteur.
    n_to_n: true
    # Fenêtre d'agrégation des paiements n↔n.
    n_to_n_window_hours: 72
    # Passe 3 : candidat unique.
    single_candidate: true
    # Cardinalité maximale d'un ensemble.
    max_invoices: 5
    # Candidats les mieux scorés en entrée du DFS.
    max_candidates: 25
    # Tolérance absolue sur la somme (centimes).
    tolerance_abs_cents: 500
    # Tolérance relative sur la somme.
    tolerance_rel: 0.03
    # Budget de nœuds explorés ; au-delà → revue.
    node_budget: 100000
  # Décision et seuils.
  decision:
    # τ_low : sous ce score, pas de proposition (le paiement reste en attente) ; entre τ_low et
    # τ_high, revue.
    review_min_score: 0.3
    # δ : marge minimale au second candidat pour l'auto-validation. À calibrer.
    min_margin: 0.05
    # Seuils par segment si le volume le permet.
    segmented_thresholds: false
    # Dimensions de segmentation des seuils.
    segments: [market, bankroll_code, amount_bucket, has_client_file]
    # Volume de validation minimal pour qu'un segment ait son propre seuil.
    min_segment_volume: 1000
  # LLM sur libellés.
  llm_labels:
    # Extraction LLM de références bruitées sur le résiduel (optionnel, gardé seulement si le gain
    # est mesurable).
    enabled: false
    # URL de l'API LLM on-premise. Aucun autre appel réseau.
    endpoint: null
    # Cache des réponses LLM.
    cache_dir: data/cache/llm

# Étape 6 — évaluation.
evaluation:
  # Précision visée : fixe τ_high et définit le taux d'automatisation à précision fixée.
  target_precision: 0.995
  # Taux d'automatisation de l'algorithme actuel (chiffre de référence client).
  current_automation_rate: null
  # Volume total de paiements de référence.
  reference_total_volume: null
  # Volume traité en manuel de référence.
  reference_manual_volume: null
''',
    'config/rules.yaml': '''# Règles déterministes de l'étape 4 (réconciliation algorithmique) — déclaratives et versionnées.
# Chaque décision garde l'identifiant de la règle et la version de ce fichier.

# Version du jeu de règles, tracée dans chaque décision. À incrémenter à toute modification.
version: 2

# Précision individuelle minimale : une règle mesurée en dessous est désactivée.
min_precision: 0.995

# Règles, appliquées par priorité croissante. Validation seulement si la solution est unique.
rules:
  - id: R1_CLIENT_FILE
    name: Client file concordant
    description: 'Chaque ligne du client file se résout vers une seule facture ouverte, montants concordants dans la tolérance : groupe entier validé.'
    enabled: true
    priority: 1
    params:
      tolerance_abs_cents: 0
      tolerance_rel: 0.0
      firm_only: 0
  - id: R2_REFERENCE_UNIQUE
    name: Référence exacte unique
    description: Une seule facture ouverte correspond à une référence du libellé, montant égal au restant dû (dans la tolérance).
    enabled: true
    priority: 2
    params:
      tolerance_abs_cents: 0
      firm_only: 0
  - id: R3_AMOUNT_UNIQUE
    name: Montant exact unique
    description: Une seule facture ouverte du débiteur a exactement le montant du paiement.
    enabled: true
    priority: 3
    params:
      firm_only: 1
  - id: R4_EXACT_SUM
    name: Somme exacte
    description: Un sous-ensemble unique de factures ouvertes du débiteur somme exactement au paiement (DFS borné).
    enabled: true
    priority: 4
    params:
      max_invoices: 5
      node_budget: 100000
      max_open_invoices: 30
      firm_only: 1
  - id: R5_PARTIAL_REFERENCED
    name: Paiement partiel sur facture référencée
    description: 'Référence unique, montant inférieur au restant dû : imputation PARTIAL.'
    enabled: true
    priority: 5
    params:
      firm_only: 0
''',
    'config/schema.yaml': '''# Mapping modèle canonique → noms réels des tables et colonnes.
# À REMPLIR PAR L'ÉQUIPE. Aucune valeur ne doit être devinée.
# Éditable à la main ou depuis l'UI (page Configuration).
#
# - Chaque `null` sous `columns` est une colonne réelle à renseigner.
#   Les champs marqués (requis) bloquent le chargement tant qu'ils sont vides.
#   Les autres peuvent rester à null : ils seront signalés dans le rapport.
# - `source` : fichier (csv ou parquet) relatif à `base_dir`.
# - `read_options` : options passées telles quelles à pandas.read_csv / read_parquet
#   (ex. sep: ";", encoding: "latin-1").
# - `value_maps` : traduction des valeurs réelles vers les valeurs canoniques,
#   ex. status: {FULL: [TOTAL, SOLDE], PARTIAL: PARTIEL}.

# Unité des montants dans les sources : "cents" (entiers de centimes) ou "units" (ex. 1234.56).
amount_unit: null
# Séparateur décimal des montants lus en texte ("." ou ",").
decimal_separator: null
# Fuseau des horodatages sans fuseau explicite (ex. "Europe/Paris"). null → UTC.
source_timezone: null
# Format strptime des dates (null → inférence pandas).
date_format: null
# Format strptime des horodatages (null → inférence pandas).
timestamp_format: null
# Répertoire des fichiers sources (relatif à la racine du dépôt ou absolu).
base_dir: null

tables:
  payment:
    source: null
    read_options: {}
    columns:
      payment_id: null  # (requis)
      value_date: null  # (requis)
      booking_date: null  # date de comptabilisation si elle existe (§3.5) — ordonne le journal
      amount: null  # (requis) signé
      currency: null  # (requis)
      iban_debtor: null
      iban_creditor: null
      label: null  # (requis)
      channel: null
      payment_type: null
      bankroll_code: null  # souvent absent (§3.5)

  invoice:
    source: null
    read_options: {}
    columns:
      invoice_id: null  # (requis)
      client_reference: null  # (requis)
      internal_reference: null
      creation_date: null  # (requis)
      due_date: null  # (requis)
      initial_amount: null  # (requis)
      current_amount: null  # audit uniquement, jamais utilisé par la pipeline
      currency: null  # (requis)
      debtor_id: null  # (requis)
      agreement_id: null  # (requis)

  imputation:
    source: null
    read_options: {}
    columns:
      payment_id: null  # (requis)
      invoice_id: null  # (requis)
      status: null  # (requis) FULL / PARTIAL après value_maps
      updated_at: null  # (requis)
      residual_amount: null  # (requis) solde de la facture APRÈS la ligne
    value_maps:
      status: {}

  assignor:
    source: null
    read_options: {}
    columns:
      party_id: null  # (requis)
      bankroll_code: null
      iban: null
      name: null  # (requis)
      opened_at: null
      closed_at: null

  debtor:
    source: null
    read_options: {}
    columns:
      party_id: null  # (requis)
      bankroll_code: null
      iban: null
      name: null  # (requis)
      opened_at: null
      closed_at: null  # peut ne pas exister (§3.5)

  agreement:
    source: null
    read_options: {}
    columns:
      agreement_id: null  # (requis)
      debtor_id: null  # (requis)
      client_id: null  # (requis) → assignor.party_id
      contract_number: null
      created_at: null  # (requis)
      disabled_at: null
      market: null
      product: null
      recourse: null

  # Optionnel — référentiel des IBAN de comptes techniques.
  technical_account:
    source: null
    read_options: {}
    columns:
      iban: null  # (requis)
      bankroll_code: null
      description: null

  # Optionnel — client files. FORMAT RÉEL À CONFIRMER (brief §3.2). Voie tabulaire uniquement.
  client_file:
    source: null
    read_options: {}
    columns:
      file_id: null  # (requis)
      received_at: null  # (requis) pivot temporel
      source_format: null
      payment_reference: null
      total_amount: null
      payment_date: null
      iban: null
      issuer_name: null

  # Optionnel — lignes des client files.
  client_file_line:
    source: null
    read_options: {}
    columns:
      file_id: null  # (requis)
      line_no: null
      invoice_reference: null  # (requis)
      amount: null
      gap_reason: null
''',
    'config/schema.synthetic.yaml': '''# Mapping du jeu synthétique (src/synthetic/generate.py). Ne concerne pas les données réelles.
amount_unit: units
decimal_separator: "."
source_timezone: Europe/Paris
date_format: "%Y-%m-%d"
timestamp_format: "%Y-%m-%d %H:%M:%S"

tables:
  payment:
    source: payment.csv
    columns:
      payment_id: payment_id
      value_date: value_date
      booking_date: booking_date
      amount: amount
      currency: currency
      iban_debtor: iban_debtor
      iban_creditor: iban_creditor
      label: label
      channel: channel
      payment_type: payment_type

  invoice:
    source: invoice.csv
    columns:
      invoice_id: invoice_id
      client_reference: client_reference
      internal_reference: internal_reference
      creation_date: creation_date
      due_date: due_date
      initial_amount: initial_amount
      current_amount: current_amount
      currency: currency
      debtor_id: debtor_id
      agreement_id: agreement_id

  imputation:
    source: imputation.csv
    columns:
      payment_id: payment_id
      invoice_id: invoice_id
      status: status
      updated_at: updated_at
      residual_amount: residual_amount
    value_maps:
      status: {FULL: TOTAL, PARTIAL: PARTIEL}

  assignor:
    source: assignor.csv
    columns:
      party_id: party_id
      bankroll_code: bankroll_code
      iban: iban
      name: name
      opened_at: opened_at
      closed_at: closed_at

  debtor:
    source: debtor.csv
    columns:
      party_id: party_id
      bankroll_code: bankroll_code
      iban: iban
      name: name
      opened_at: opened_at

  agreement:
    source: agreement.csv
    columns:
      agreement_id: agreement_id
      debtor_id: debtor_id
      client_id: client_id
      contract_number: contract_number
      created_at: created_at
      disabled_at: disabled_at
      market: market
      product: product
      recourse: recourse

  technical_account:
    source: technical_account.csv
    columns:
      iban: iban
      bankroll_code: bankroll_code
      description: description

  client_file:
    source: client_file.csv
    columns:
      file_id: file_id
      received_at: received_at
      source_format: source_format
      payment_reference: payment_reference
      total_amount: total_amount
      payment_date: payment_date
      iban: iban
      issuer_name: issuer_name

  client_file_line:
    source: client_file_line.csv
    columns:
      file_id: file_id
      line_no: line_no
      invoice_reference: invoice_reference
      amount: amount
      gap_reason: gap_reason
''',
    '.streamlit/config.toml': '''# Aucun appel réseau sortant (brief §10) : pas de télémétrie, écoute locale uniquement.
[browser]
gatherUsageStats = false

[server]
address = "localhost"
headless = true
runOnSave = false

[client]
toolbarMode = "minimal"
''',
}


def ensure_config(root: Path = REPO_ROOT) -> None:
    """Crée les fichiers de configuration manquants (jamais d'écrasement)."""
    for relative, content in _EMBEDDED_FILES.items():
        target = root / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")


ensure_config()
