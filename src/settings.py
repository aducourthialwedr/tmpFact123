"""Paramètres de la pipeline, par étape — source unique pour la pipeline et l'UI.

- `config/settings.yaml` : `Settings` (chemins, chargement, découpage,
  allocation, réconciliation ML, évaluation).
- `config/rules.yaml`    : `RulesConfig`, les règles déterministes de l'étape 4
  (déclaratives et versionnées, brief §6).

Les valeurs par défaut marquées « à calibrer » sont des points de départ, pas
des résultats : elles se fixent sur la période de validation.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import DEFAULT_SETTINGS_PATH, REPO_ROOT, read_yaml
from src.yaml_io import dump_model

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
Paramètres de la pipeline, par étape. Édité à la main, depuis le notebook ou par l'UI (streamlit run src/ui/app.py).
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
