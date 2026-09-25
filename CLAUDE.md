# POC — Moteur de rapprochement automatique paiements / factures

Ce fichier est le brief pour Claude Code. Lis-le en entier avant toute tâche et respecte l'ordre des étapes.

La spécification détaillée (features, formules, pseudo-code) est dans `spec_rapprochement_automatique.md`. Ce fichier fixe les choix du POC et la structure de la pipeline. En cas de contradiction, **ce fichier prime** et l'écart doit être signalé.

---

## 1. Contexte

Activité d'affacturage chez un client bancaire. Le cédant vend ses créances, le débiteur paie, les flux peuvent transiter par des comptes techniques. Un algorithme existant rapproche automatiquement une partie des paiements, le reste part en manuel.

Objectif : démontrer un gain de taux d'automatisation **à précision fixée (cible ≥ 99,5 %)**, comparé au taux actuel.

Données : 1 an d'historique réel. Chiffres de référence disponibles : volume total, taux d'automatisation actuel, volume manuel.

Infrastructure : API LLM **on-premise**. Aucun appel réseau vers l'extérieur.

---

## 2. Vue d'ensemble de la pipeline

```
1. Chargement des données
2. Découpage temporel quotidien          → boucle jour par jour
      pour chaque jour D :
3.       Allocation des paiements         → paiement ↔ débiteur (+ client file)
4.       Réconciliation algorithmique     → règles déterministes, somme exacte
5.       Réconciliation ML                → sur le résiduel de l'étape 4
6. Évaluation                             → par étape et globale
```

Les étapes 3, 4 et 5 s'exécutent **dans la boucle quotidienne**, sur l'état tel qu'il était au jour D. Chaque étape ne traite que ce que l'étape précédente n'a pas résolu, et chaque décision garde la trace de l'étape qui l'a produite.

Un seul code sert au rejeu de l'historique et à la production.

---

## 3. Étape 1 — Chargement des données (`src/load/`)

Conventions : montants en **entiers de centimes**, dates en `DATE` ou `TIMESTAMP` UTC. Les noms réels des tables et colonnes sont dans `config/schema.yaml`, à remplir par l'équipe. Ne jamais les deviner.

### 3.1 Tables sources (détail en §2 de la spec)

| Table | Champs principaux |
|---|---|
| `payment` | `payment_id`, `value_date`, `amount` (signé), `currency`, `iban_debtor`, `iban_creditor`, `label`, `channel`, `payment_type` |
| `invoice` | `invoice_id`, `client_reference`, référence interne, `creation_date`, `due_date`, `initial_amount`, `current_amount`, `currency`, `debtor_id`, `agreement_id` |
| `imputation` | `payment_id`, `invoice_id`, `status` (`FULL`/`PARTIAL`), `updated_at`, `residual_amount` — **source des labels** |
| `assignor`, `debtor` | `party_id`, `bankroll_code`, `iban`, `name`, `opened_at`, `closed_at` |
| `agreement` | `agreement_id`, `debtor_id`, `client_id`, `contract_number`, `created_at`, `disabled_at`, `market`, `product`, `recourse` |
| `client_file` | voir §3.2 |

### 3.2 Client files

Information transmise par le client pour un paiement, qui indique comment le rapprocher (liste des factures réglées et montants). **Format réel à confirmer** avant implémentation.

| Champ | Notes |
|---|---|
| `file_id` | PK |
| `received_at` | horodatage de réception — **pivot temporel**, pas la date du paiement |
| `source_format` | structuré (CSV, EDI, XML) ou non structuré (PDF, email) |
| clés de rattachement | référence de virement, montant total, date, IBAN, émetteur |
| lignes | référence de facture citée, montant, éventuel motif d'écart |

Si le format est non structuré, l'extraction des lignes se fait par LLM au chargement, avec sortie JSON stricte et cache.

### 3.3 Normalisation (`src/load/normalize.py`)

Appliquée au chargement, à tous les libellés, références et lignes de client files. Déterministe et versionnée.
- Majuscules, suppression accents (NFKD), non alphanumérique → espace, compression des espaces.
- `label_tokens` : tokens alphabétiques.
- `label_numbers` : tokens contenant des chiffres, plus variantes sans zéros de tête et sans préfixe (`FA0012345` → `FA0012345`, `0012345`, `12345`).
- Gestion de l'**inversion** bloc lettres / bloc chiffres (`123 FACT` ↔ `FACT123`) en reconstituant les paires de tokens adjacents dans les deux ordres.

### 3.4 Journal d'événements (`src/load/events.py`)

Les tables sont converties en un journal ordonné : `INVOICE_CREATED`, `PAYMENT_RECEIVED`, `CLIENT_FILE_RECEIVED`, `IMPUTATION_APPLIED`, `PARTY_OPENED/CLOSED`, `AGREEMENT_CREATED/DISABLED`. Départage déterministe des ex æquo (type d'événement, puis identifiant).

### 3.5 Écarts à vérifier sur les données réelles

- `debtor` peut ne pas avoir de `closed_at`.
- `payment` peut ne pas avoir de `bankroll_code` : jointure sur le débiteur, comptes techniques via un référentiel d'IBAN.
- `imputation` n'a pas de montant imputé : `residual_amount` est le solde **après** la ligne (`FULL` → 0).
- Existence éventuelle d'une date de comptabilisation distincte de `value_date`. Si oui, elle ordonne le journal.

Hors périmètre v1 : montants négatifs, reversements cédant, cross-currency.

**Fini quand** un script affiche volumes, plages de dates, champs manquants par table, et produit le journal.

---

## 4. Étape 2 — Découpage temporel quotidien (`src/timeline/`)

### 4.1 Boucle quotidienne
- `LedgerState` (`state.py`) reconstruit l'état à une date. Toute lecture exige un `as_of` : factures ouvertes, `current_amount` à date, activité des parties, agrégats comportementaux (incrémentaux), client files reçus.
- `DailyIterator` avance l'état **une fois par jour**. Pour chaque jour D, il fournit le lot de paiements à traiter (nouveaux du jour **+ reliquat** non résolu des jours précédents, rétention 60 jours) et l'état figé à la veille de D.
- Un client file n'est visible qu'à partir de `received_at`. Un paiement en reliquat est retraité le jour où son fichier arrive.
- **Jamais** `invoice.current_amount` brut, c'est une fuite directe.

### 4.2 Découpage des périodes
Découpage sur les jours, jamais aléatoire, avec purge de quelques jours entre blocs :
- **train** (≈ 8 mois) : sert à construire le dataset et entraîner le ML de l'étape 5
- **validation** (≈ 2 mois) : calibration et choix des seuils
- **test / backtest** (2 derniers mois) : jamais vu, exécution complète des étapes 3 à 5

Bornes paramétrables dans `config/settings.yaml`.

### 4.3 Tests obligatoires
- Supprimer les événements postérieurs à D ne change aucune sortie calculée à D.
- Deux exécutions sur le même journal produisent exactement le même résultat.

**Fini quand** un rapprocheur quelconque peut être branché dans la boucle et évalué sans fuite.

---

## 5. Étape 3 — Allocation des paiements (`src/allocation/`)

But : rattacher chaque paiement à **un débiteur** (et à son client file s'il existe), pour restreindre la réconciliation aux factures de ce débiteur. Ce n'est pas un rattachement au contrat.

Signaux, du plus fort au plus faible :
1. **Client file** rattaché au paiement sans ambiguïté (référence de virement, montant total, date, IBAN).
2. **Référence** de facture trouvée dans `label_numbers` → débiteur de cette facture.
3. **IBAN** via `resolve_iban` : `DEBTOR_DIRECT` / `ASSIGNOR` / `TECHNICAL_ACCOUNT` / `UNKNOWN`, arbitré par `bankroll_code`. Jamais de jointure IBAN naïve.
4. **Nom** : similarité entre libellé et nom du débiteur, via index inversé avec élagage des tokens trop fréquents.

Sortie par paiement : une **liste classée de débiteurs candidats** avec un score et le signal qui les a produits, plus le client file rattaché. Si un seul débiteur ressort avec un signal fort, l'allocation est ferme. Sinon, les étapes suivantes reçoivent plusieurs débiteurs possibles. Aucun débiteur plausible → file analyste.

Contrainte : aucune recherche sans index, jamais de boucle sur tous les débiteurs.

**Métrique clé : rappel d'allocation** (le vrai débiteur est dans la liste) **≥ 99 %**, et précision de l'allocation ferme. Un débiteur manqué ici est perdu pour la suite.

---

## 6. Étape 4 — Réconciliation algorithmique (`src/reconcile_rules/`)

But : résoudre tout ce qui peut l'être par des règles déterministes, auditables, sans apprentissage. C'est aussi la **baseline** à battre par l'étape 5.

Travaille sur les factures ouvertes à D des débiteurs alloués. Règles déclaratives en YAML, versionnées, appliquées par priorité :

1. **Client file concordant** : chaque ligne se résout vers une seule facture ouverte, montants concordants dans la tolérance → groupe entier validé.
2. **Référence exacte unique** : une seule facture ouverte correspond à une référence du libellé, montant compatible.
3. **Montant exact unique** : une seule facture ouverte du débiteur a exactement le montant.
4. **Somme exacte** : un sous-ensemble unique de factures ouvertes du débiteur somme exactement au paiement (DFS borné, max 5 factures, budget de nœuds).
5. **Paiement partiel sur facture référencée** : référence unique, montant inférieur au restant dû → imputation `PARTIAL`.

Principes :
- Validation **seulement si la solution est unique**. Ambiguïté → étape 5.
- Chaque règle est mesurée seule (précision, couverture). Une règle sous le seuil de précision est désactivée par config.
- Conflits arbitrés au niveau du lot du jour : une facture n'est bloquée que si elle est soldée totalement.
- Chaque décision garde l'identifiant et la version de la règle.
- Tests de non-régression sur un jeu figé de cas.

**Fini quand** la couverture et la précision de la baseline sont mesurées et comparées au taux actuel.

---

## 7. Étape 5 — Réconciliation ML (`src/reconcile_ml/`)

Ne traite que le **résiduel** de l'étape 4.

### 7.1 Candidats
Factures ouvertes à D des débiteurs candidats, plus les clés de la spec §4 : référence sans fenêtre, montant ± 90 j, factures citées dans le client file. Filtres durs : devise, facture ouverte, agreement actif. Rappel des candidats mesuré.

### 7.2 Scoring de paires
LightGBM binaire. Négatifs = tous les autres candidats du paiement, sans échantillonnage. Familles de features selon §5.3 de la spec : montant (dont escompte, frais, retenue de garantie), temporel, textuel, identité, comportemental (fenêtre strictement antérieure), contexte contrat. Plus :
- famille **allocation** : score et signal d'allocation du débiteur
- famille **client file** : `has_client_file`, `invoice_cited_in_client_file`, `client_file_line_amount_diff`, `client_file_total_matches_payment`, `debtor_client_file_rate`

Deux passes avec features de compétition (rang, marge au second, nombre de candidats), puis **calibration isotonique** sur la validation.

### 7.3 Entraînement
Le dataset est construit en rejouant la boucle quotidienne sur la période train. Features extraites sur l'état **avant** application des événements du jour. Labelling dans une **seconde passe séparée** à partir de `imputation`. Dataset partitionné par mois, colonne de date conservée. Hash du journal et de la featurisation versionné avec le modèle.

### 7.4 Ensembles
DFS borné sur les candidats les mieux scorés (max 5, 25 candidats, tolérance 5 € ou 3 %, préférence pour la cardinalité minimale). Passes dans cet ordre : somme proche, agrégats n↔n (paiements du même débiteur sur 72 h), candidat unique. Arbitrage des conflits au niveau du lot.

### 7.5 Décision
Auto-validation si `p ≥ τ_high` **et** marge au second ≥ δ. Revue entre `τ_low` et `τ_high`. Rejet sous `τ_low`. `τ_high` calibré pour ≥ 99,5 % de précision sur la validation. Seuils par segment (`market`, `bankroll_code`, tranche de montant, présence de client file) si le volume le permet.

### 7.6 LLM sur libellés (optionnel)
Extraction de références bruitées sur le résiduel, injectée comme candidats et features. Échantillon stratifié sur-représentant les cas manuels, cache, gardé seulement si le gain est mesurable.

---

## 8. Étape 6 — Évaluation (`src/evaluation/`)

Exécutée sur la période de test, en mode complet (étapes 3 à 5, seuils inclus), comparée aux imputations réellement prononcées.

Par étape :
- **Allocation** : rappel, précision de l'allocation ferme, part de paiements avec client file rattaché
- **Réconciliation algorithmique** : précision et couverture par règle et cumulées
- **Réconciliation ML** : rappel des candidats, precision@1, MRR, calibration, groupes reconstitués par type (1↔1, 1↔n, n↔1, n↔n)
- **Décision** : courbe taux d'automatisation / précision, volume en revue

Globales :
- **Taux d'automatisation à précision ≥ 99,5 %**, métrique métier principale
- Tableau en cascade : taux actuel, puis étape 4 seule, puis étapes 4 + 5, avec le gain de chaque brique
- Performance par mois pour mesurer la dégradation dans le temps
- Découpage avec et sans client file

Rapports reproductibles dans `reports/` (CSV + résumé markdown).

---

## 9. Ordre de développement

Chaque étape est validée avant de passer à la suivante. Chaque brique est mesurée et retirée si elle n'apporte rien.

1. Étape 1 — chargement, normalisation, journal
2. Étape 2 — état, boucle quotidienne, split, tests anti-fuite
3. Étape 6 minimale — harnais d'évaluation branché sur un rapprocheur vide
4. Étape 3 — allocation. **Ne pas avancer tant que le rappel est < 99 %.**
5. Étape 4 — réconciliation algorithmique, baseline mesurée
6. Étape 5 — candidats, puis scoring simple, puis familles avancées, puis deuxième passe, puis calibration, puis ensembles, puis décision
7. Étape 6 complète — backtest et tableau en cascade
8. LLM sur libellés, si le résiduel le justifie

---


---

## 10. Consignes pour Claude Code

- Travailler étape par étape, proposer un plan avant de coder.
- Ne jamais inventer un nom de table ou de colonne, ni un format de client file.
- Toute logique temporelle passe par `LedgerState` avec `as_of`.
- Aucune recherche sans index. Aucune boucle sur tous les débiteurs ou toutes les factures.
- Aucun appel réseau hors de l'endpoint LLM on-premise.
- Tests sur petits jeux construits à la main ou générés avec seed fixe, jamais sur les données réelles.
- Tous les tests passent avant de considérer une tâche terminée.
- Après modification de la normalisation, de l'allocation, de la featurisation ou du modèle, relancer la chaîne depuis l'étape concernée.