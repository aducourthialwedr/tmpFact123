# Guide du projet — rapprochement automatique paiements / factures

Ce guide explique comment le moteur fonctionne, pourquoi il est construit ainsi, et surtout les
**subtilités** qui ne se voient pas en lisant le code en diagonale. Il complète :

- `README.md` : installation et utilisation rapide ;
- `CLAUDE.md` : le brief (choix du POC, ordre des étapes) ;
- `spec_rapprochement_automatique.md` : la spécification détaillée (features, formules).

> **Version en 4 fichiers.** `reconciliation.py` regroupe tout le code de `src/` : chaque section
> y est précédée d'une bannière `src/<module>.py`, les chemins cités dans ce guide s'y retrouvent.
> L'interface est `reconciliation_ui.py` (`streamlit run reconciliation_ui.py`), le notebook
> `pipeline.ipynb`. Les fichiers de configuration sont créés au premier `import reconciliation`.
> Dépendances : `pip install pandas pyarrow numpy PyYAML pydantic scipy scikit-learn lightgbm`
> (+ `jupyterlab ipykernel altair` pour le notebook, `streamlit altair` pour l'interface).

---

## 1. Le problème en une page

Activité d'affacturage : le cédant vend ses créances (factures) au factor, le débiteur paie, parfois
via des comptes techniques. Chaque paiement reçu doit être **imputé** sur la ou les factures qu'il
règle. Aujourd'hui une partie est automatique, le reste part en traitement manuel.

**Objectif du POC** : augmenter le taux d'automatisation **à précision fixée ≥ 99,5 %**. Une
imputation erronée coûte bien plus cher qu'un passage en revue : on préfère ne pas décider que mal
décider.

Ce qui rend le problème difficile :

| Difficulté | Exemple |
|---|---|
| Références bruitées | `FA0012345` écrit `12345`, `123 45`, `0012345 FA`, avec une faute de frappe, ou la référence d'une autre facture |
| Montants décalés | escompte (0,5-3 %), frais bancaires (5-40 €), retenue de garantie BTP (5 %) |
| Groupes | 1↔n (un virement pour plusieurs factures), n↔1 (paiements partiels), n↔n |
| Identité floue | IBAN d'un compte technique, IBAN inconnu, nom tronqué au premier mot |
| Collisions | un même numéro de facture existe chez plusieurs cédants ; ~7 factures partagent chaque montant au centime près à 3 M de factures |
| Temps | une information n'existe qu'à partir d'une date ; l'utiliser avant, c'est tricher (fuite) |

---

## 2. Vue d'ensemble

```
 sources (CSV / parquet)            config/schema.yaml : mapping vers le modèle canonique
        │
 1 · chargement ── normalisation ── journal d'événements ── contrôles qualité
        │
 2 · état du grand livre à date (LedgerState) ── boucle jour par jour (DailyIterator)
        │                                             │
        │                      pour chaque jour D : lot = paiements du jour + reliquat
        │                                             │
 3 ·    │                                        allocation  → débiteurs candidats
 4 ·    │                                        règles      → décisions sûres (solution unique)
 5 ·    │                                        ML          → sur le résiduel des règles
        │                                             │
 6 · évaluation : décisions comparées aux imputations réellement prononcées
```

Point d'entrée : `src/api.py`, classe `Project`, une méthode par étape. Le notebook
`notebooks/pipeline.ipynb` et l'interface Streamlit appellent exactement ces méthodes.

**Un seul code pour le rejeu et la production.** Le backtest rejoue l'historique jour par jour avec
le même code que celui qui tournerait chaque nuit en production. Deux implémentations séparées
finissent toujours par diverger.

---

## 3. Concepts transverses (à comprendre avant tout le reste)

### 3.1 Le journal d'événements

Les tables sources sont converties en un journal ordonné d'événements datés :
`INVOICE_CREATED`, `PAYMENT_RECEIVED`, `CLIENT_FILE_RECEIVED`, `IMPUTATION_APPLIED`,
`PARTY_OPENED/CLOSED`, `AGREEMENT_CREATED/DISABLED` (`src/load/events.py`).

- **Départage des ex æquo** : beaucoup de dates sont sans heure. À horodatage égal, l'ordre est :
  type d'événement (ce qui ouvre avant ce qui l'utilise, ce qui ferme en dernier), puis
  identifiant. Sans cette règle, deux exécutions donneraient deux résultats différents.
- **Date de connaissance** : un paiement est daté par `booking_date` si elle existe, sinon par
  `value_date`. La date de valeur peut précéder de quelques jours le moment où le paiement est
  réellement visible dans le SI.
- **Empreinte** : le journal a une empreinte SHA-256 (`journal_sha256`). Rejeux, évaluations et
  modèles la mémorisent ; une incohérence (journal rechargé depuis) est signalée.

### 3.2 L'état à date et l'interdiction de lire le futur

`LedgerState` (`src/timeline/state.py`) rejoue le journal et répond aux questions du moteur
(factures ouvertes, restant dû, débiteurs connus, agrégats comportementaux…) **pour l'instant
`as_of` auquel il a été avancé**, et seulement pour celui-là :

- toute lecture exige `as_of` ; une lecture à une autre date lève `TemporalError` ;
- l'état ne recule jamais ;
- il reflète les événements **strictement antérieurs** à `as_of`.

**Le restant dû n'est jamais lu dans `invoice.current_amount`** : ce champ est le solde *final*, une
fuite directe (une facture à 0 est une facture déjà soldée). Il est retiré de la table au
chargement et ne sert qu'à un contrôle de cohérence. Le restant dû à date est reconstruit :
`initial_amount − Σ imputations antérieures`.

### 3.3 La journée D

- L'état est **figé à la veille** : minuit du jour D, événements < D.
- Conséquence voulue : une facture créée le jour D n'est visible qu'à partir de D+1. Un paiement
  qui la règle le jour même attend dans le reliquat et est retraité le lendemain.
- Le **lot** du jour = paiements arrivés le jour D + **reliquat** (paiements non résolus des jours
  précédents, pendant 60 jours au plus, `split.retention_days`).
- Un paiement quitte le reliquat quand : une imputation réelle est prononcée ; le moteur l'a
  auto-validé ; sa rétention expire.
- Tout paiement en reliquat est **retraité chaque jour**. C'est ainsi qu'il profite d'un client file
  arrivé après lui, ou d'une facture créée après lui.

### 3.4 Les garde-fous contre les fuites

- **Mécaniques** : lectures datées ; toute décision d'un rapprocheur est contrôlée par la boucle
  (`run_replay`) — citer une facture qui n'existe pas encore à D, ou un paiement absent du lot,
  lève `LeakError`.
- **Index sans information temporelle** : les index (références, noms, IBAN) sont construits une
  fois sur les attributs statiques, mais **tout ce qui dépend du temps est filtré à la requête**
  (facture créée à D ? débiteur connu à D ? combien de débiteurs partagent cette clé *à D* ?).
  Élaguer une clé « trop fréquente » en comptant les factures futures serait une fuite.
- **Test de troncature** (`tests/`) : on rejoue une fois avec toutes les données, une fois avec des
  données coupées au jour D ; toutes les sorties jusqu'à D doivent être identiques au bit près.
  Ce test a attrapé trois fuites subtiles pendant le développement :
  1. l'élagage de clés de référence sur un comptage incluant des factures futures ;
  2. les codes des catégories (canal, marché…) recalculés sur toutes les données — désormais figés
     dans le modèle à l'entraînement ;
  3. le nombre de « références du libellé » compté via un vocabulaire construit sur toutes les
     factures, futures comprises.

### 3.5 Rejeu : l'état suit la réalité, pas le moteur

En backtest, les soldes des factures suivent les **imputations réelles** du journal. Les décisions
du moteur ne modifient pas l'état : elles sont comparées à la réalité à l'évaluation.

Problème : une facture soldée par le moteur le jour D resterait ouverte le lendemain (tant que
l'imputation réelle n'est pas arrivée) et pourrait être attribuée une seconde fois. D'où le
**registre des réservations du moteur** (`RulesMatcher._claimed`) : les montants que le moteur a
imputés sont retirés du restant dû, **jusqu'à ce que l'imputation réelle du paiement apparaisse
dans l'état** (la réservation est alors levée, pour ne pas compter deux fois). Règles et ML
partagent ce registre.

### 3.6 Conventions de données

- Montants en **entiers de centimes**, jamais de flottant (conversion texte → centimes sans passer
  par un float ; `0.29` € donne exactement 29).
- Dates et horodatages en UTC naïf, `datetime64[us]`.
- `imputation.residual_amount` est interprété comme **le solde de la facture après la ligne**
  (brief §3.5). Le montant imputé en est déduit : `solde avant − solde après`. La spec disait
  « écart paiement / facture » : le brief prime. Le contrôle qualité vérifie
  `initial − Σ imputés = current_amount` final ; s'il échoue sur les données réelles,
  l'interprétation est à revoir.

---

## 4. Étape par étape : ce que ça fait et ce qu'il faut savoir

### Étape 1 — chargement (`src/load/`)

- **Mapping** : `config/schema.yaml` associe chaque champ canonique à une colonne réelle.
  **Rien n'est deviné** : un mapping requis manquant bloque le chargement avec la liste complète
  de ce qui reste à renseigner. Les champs optionnels absents sont signalés dans le profil.
- **Normalisation** (`normalize.py`, version `1.1.0`) : majuscules, sans accents, non
  alphanumérique → espace. Les « clés de référence » d'un libellé incluent :
  - les tokens contenant des chiffres, et leurs variantes sans préfixe alphabétique et sans zéros
    de tête (`FA0012345` → `FA0012345`, `0012345`, `12345`) ;
  - l'**inversion** bloc lettres / bloc chiffres (`123 FACT` ↔ `FACT123`) ;
  - la concaténation de **3 tokens adjacents** (extension du brief) : sans elle, `FA-2024-00123`
    écrit tel quel dans un libellé ne retrouve pas sa propre facture (la ponctuation le découpe).
  Un match de référence = intersection des clés du libellé et des clés de la facture.
- **Contrôles qualité** (`quality.py`) : rien n'est corrigé, tout est compté (doublons, orphelins,
  imputations antérieures au paiement ou à la facture, `FULL` avec résidu non nul…).
- **Subtilité performance** : sur les colonnes de chaînes pyarrow, `Series.isin`, `Series.map`,
  `str.extract` et `groupby().agg(tuple)` repassent par du Python ligne à ligne (10 à 100 fois plus
  lent). Le code utilise `src/arrow_ops.py` (`pyarrow.compute`) et des découpages de tableaux
  triés. À garder en tête pour tout nouveau code à 2 M de lignes.

### Étape 2 — temps (`src/timeline/`)

- **Découpage** (`split.py`) : ancré sur la fin de l'historique — test = 2 derniers mois,
  validation = 2 mois avant, entraînement = le reste ; 5 jours de purge entre blocs.
- **Application jour par jour** : même lors d'un grand saut, `advance_to` applique le journal jour
  par jour, pour que les agrégats (ex. « paiement partiel » évalué en fin de journée) ne dépendent
  pas de la taille des sauts.
- **Agrégats comportementaux** (délai moyen, taux de partiels, de groupés, de citation de la
  référence, encours) : maintenus incrémentalement sur une fenêtre glissante strictement
  antérieure (180 jours), par contributions journalières qui expirent.
- **Démarrage d'un rejeu** : le reliquat initial reprend les paiements des 60 jours précédant la
  période, pour se placer en régime établi.

### Étape 3 — allocation (`src/allocation/`)

Rattacher chaque paiement à une **liste classée de débiteurs candidats** (max 10), pour restreindre
la suite aux factures de ces débiteurs. Un débiteur manqué ici est perdu : la cible du brief est un
rappel ≥ 99 %, bloquant pour la suite.

| Signal | Poids | Détail |
|---|---|---|
| Client file | 1,0 | fichier reçu rattaché sans ambiguïté (montant total égal, corroboré par date, IBAN ou référence) ; débiteurs des factures citées |
| Référence | 0,95 | clé du libellé → factures **ouvertes à D** ; si l'une a un restant dû égal au paiement, elle seule compte ; spécificité = 1 / nombre de débiteurs **par paiement** |
| IBAN | 0,9 | routage `DEBTOR_DIRECT` / `ASSIGNOR` / `TECHNICAL_ACCOUNT` / `UNKNOWN` — jamais de jointure IBAN naïve |
| Nom | 0,7 | index inversé (mots + bigrammes), mots trop fréquents élagués à D ; score = max(couverture pondérée par la rareté, spécificité du meilleur mot) |
| Montant | 0,5 | facture ouverte de restant dû égal au paiement (extension du brief) |

- Les scores se combinent en « ou » probabiliste : `1 − Π(1 − sᵢ)`. Plusieurs signaux concordants
  se renforcent.
- **Allocation ferme** : un seul débiteur porte un signal fort (client file univoque, référence
  désignant un seul débiteur avec une clé d'au moins 6 caractères, IBAN direct unique).
- **Corroborations** (indispensables à l'échelle) :
  - un mot fréquent (« Nettoyage ») ne crée jamais de candidat, mais **confirme** un débiteur
    proposé par le montant exact quand jusqu'à 30 débiteurs partagent ce montant ;
  - un nom peu spécifique (homonymes) est gardé si un de ces débiteurs a une facture ouverte de
    restant dû **proche** du paiement (5 € ou 3 % : escompte, frais).
- **Références partagées entre cédants** : à 2 000 cédants numérotant chacun de leur côté, un même
  `FA0012345` existe chez une douzaine de débiteurs. D'où le filtrage « ouvertes à D » et la
  préférence au montant exact, calculés par paiement.
- **IBAN à la fois débiteur et cédant** : arbitrage par le `bankroll_code` du paiement s'il est
  connu, sinon paiement direct sans allocation ferme — **à confirmer avec le métier**.
- Résultat à 2 M : rappel 99,16 %, vrai débiteur en tête 97,2 %, précision de l'allocation ferme
  99,64 %. Les difficultés sont concentrées sur les comptes techniques et IBAN inconnus.

### Étape 4 — règles (`src/reconcile_rules/`, `config/rules.yaml`)

Règles déterministes, appliquées par priorité, **validées seulement si la solution est unique** :

| Règle | Portée | Principe |
|---|---|---|
| R1 client file concordant | tous les candidats | chaque ligne du fichier → une seule facture ouverte ; montants concordants ; somme = paiement |
| R2 référence exacte unique | tous les candidats | une seule facture ouverte citée ; restant dû = paiement |
| R3 montant exact unique | allocation ferme | une seule facture ouverte du débiteur a exactement le montant |
| R4 somme exacte | allocation ferme | un seul sous-ensemble (≤ 5 factures parmi ≤ 30) somme exactement ; arrêt dès la 2ᵉ solution |
| R5 partiel référencé | tous les candidats | référence unique, montant < restant dû → imputation partielle |

- **Portée** (`firm_only`) : les règles « montant » ne portent que sur une allocation ferme, sinon
  les collisions de montants entre débiteurs créeraient des erreurs.
- **Arbitrage au niveau du lot** : une facture n'est bloquée que si elle est soldée totalement ; deux
  paiements qui la soldent par la même règle sont tous deux rejetés (doublons de paiement).
- **Mesure** : chaque règle est mesurée « seule » (sur tous les paiements) et « en cascade » (ce
  qu'elle décide effectivement). R4, coûteuse, n'est mesurée seule que sur le résiduel de R1-R3.
  Une règle sous la précision minimale se désactive depuis l'interface, ce qui crée une nouvelle
  **version** du jeu de règles, tracée dans chaque décision.
- Tests de non-régression : un jeu figé de cas (un par règle, ambiguïtés, ex æquo, réservations).

### Étape 5 — ML (`src/reconcile_ml/`)

Ne traite que le **résiduel** des règles.

**Candidats** (`features.py`) : factures ouvertes des **3 premiers débiteurs** de l'allocation
(échéance entre −180 et +30 jours), plus les clés référence (sans fenêtre), montant exact (± 90 j)
et client file. Filtres durs : devise, facture ouverte, contrat actif. **Plafond de 60 par
paiement** :

- les clés précises passent d'abord ;
- puis l'union des factures les plus proches en montant **et** de celles dont l'échéance est la plus
  proche, parmi celles qui peuvent entrer dans le paiement.

> Piège rencontré : trier uniquement par proximité de montant élimine les vraies factures des
> paiements groupés (chacune bien plus petite que le paiement). 92 % des candidats manquants
> venaient de là.

**Scoring** (`model.py`) : LightGBM binaire sur les paires (paiement, facture), négatifs = tous les
autres candidats du paiement (pas d'échantillonnage des négatifs ; on échantillonne des paiements
entiers, 50 %).

- Familles de features du brief (montant, temporel, textuel, identité, comportemental, contrat,
  allocation, client file), activables dans `settings.yaml`.
- **Deuxième passe** avec features de compétition (rang, marge au meilleur concurrent). En
  entraînement, les scores de passe 1 sont calculés **hors échantillon** (3 plis par paiement) :
  sinon la passe 2 apprend sur des scores trop optimistes.
- **Calibration isotonique** sur la validation.
- Le dataset est construit en rejouant la boucle (features sur l'état avant les événements du
  jour), puis **étiqueté dans une seconde passe séparée** à partir des imputations. Il est
  partitionné par mois (`data/interim/…/ml/dataset`).

**Propositions** (`decision.py`) :

- **facture seule** : recevable seulement si elle *absorbe* le paiement (montant ≤ restant dû +
  tolérance). Si l'écart est dans la tolérance → « soldée » ; sinon → « partiel » ;
- **ensemble** : sous-ensembles des 25 meilleurs candidats dont la somme est à 5 € ou 3 % près,
  cardinalité minimale préférée, score = moyenne des scores de paire ;
- la meilleure proposition l'emporte ; **marge** = écart à la seconde ;
- **ambiguïté explicite** : une facture seule qui n'explique le paiement qu'à un escompte près,
  face à un ensemble qui l'explique exactement, reçoit une marge nulle → revue ;
- **n↔n** : paiements non résolus d'un même débiteur à 72 h d'intervalle agrégés ; ne produit que
  des propositions **en revue** (heuristique, comme le prévoit la spec).

**Décision** : auto-validation si score ≥ τ_high **et** marge ≥ δ (0,05) ; revue si score ≥ τ_low
(0,3) ; sinon rien (le paiement reste en attente et sera rescoré demain).

Les subtilités qui ont compté :

1. **Un seuil par type de proposition** (soldée, partiel, ensemble) : un « partiel » masque souvent
   un n↔n (le premier virement solde une facture et une partie de la suivante). En pratique les
   partiels ne passent jamais le seuil.
2. **Décision sur le score brut de passe 2**, pas sur la probabilité calibrée : la régression
   isotonique crée des plateaux (beaucoup de scores à exactement 1,0 mêlant justes et faux) où
   aucun seuil ne peut couper. Le score brut a le même ordre (la calibration est monotone), sans
   plateaux. La probabilité calibrée reste calculée pour les mesures de calibration.
3. **Calibration en ligne des seuils** : en production, un paiement en attente est rescoré chaque
   jour, donc tenté plusieurs fois contre le seuil. Calibrer sur le seul jour d'arrivée
   surestimait la précision (96,5 % observés pour 99,5 % visés). On rejoue donc la validation
   sans auto-validation ML, en enregistrant la proposition de chaque paiement chaque jour ; pour
   tout τ, la décision retenue est celle du **premier jour où le score franchit τ**. τ_high est le
   plus petit seuil tenant la précision cible (avec au moins 20 décisions).
   `project.calibrate()` refait cette étape sans réentraîner.
4. **Versionnement** : le modèle est sauvegardé (`models/<jeu>/pair_model/`) avec l'empreinte du
   journal, la version de la featurisation, la version des règles, les paramètres, les catégories,
   les seuils, les métriques de validation, la calibration et l'importance des features.

### Étape 6 — évaluation (`src/evaluation/`)

- **Unité** : le paiement. Une auto-validation est **correcte** si l'ensemble des factures décidées
  est exactement celui réellement imputé (toutes dates confondues). Auto-valider un paiement
  jamais imputé en réalité compte comme une erreur (choix conservateur).
- **Périmètre** : les paiements arrivés pendant la période ; le reliquat de démarrage est exclu.
- **Type de groupe** lu sur la composante connexe du graphe paiements–factures.
- **Métriques** : taux d'automatisation, précision, taux d'automatisation à précision cible
  (courbe : décisions des règles acquises, seuil ML variable), précision et rappel au niveau des
  paires, volume en revue, cascade (actuel → règles → règles + ML), découpages par étape, règle,
  groupe, mois, client file ; diagnostics ML (rappel des candidats, precision@1, MRR, calibration).
- Rapports reproductibles dans `reports/` (CSV, JSON, markdown).

---

## 5. Le jeu synthétique (`src/synthetic/generate.py`)

Sert aux tests et à la démonstration, **jamais à conclure sur la performance réelle**. Seed fixe,
génération vectorisée (~1 min pour 2 M de paiements).

Il reproduit : débiteurs de tailles très inégales (le plus gros a ~39 000 factures par an),
numérotation des factures par cédant (collisions), références citées de façon bruitée (sans
préfixe, sans zéros, inversées, fautes de frappe, mauvaise facture), comptes techniques et IBAN
inconnus, paiements groupés, partiels, retenues de garantie BTP, n↔n, escompte et frais, paiements
orphelins (3 %, jamais imputés) et doublons, client files.

Choix à connaître :

- **Orphelins** : ce sont des flux sans facture correspondante. Une première version laissait 3 %
  de paiements « jamais imputés » qui réglaient pourtant de vraies factures : les règles les
  rapprochaient à juste titre et la précision mesurée chutait artificiellement.
- **Noms** : patronymes synthétiques en tête de la plupart des raisons sociales. Une version avec
  40 mots pour 50 000 débiteurs rendait tout libellé tronqué indéterminable, ce qui n'est pas
  réaliste. Les homonymes restent nombreux (≈ 5-10 débiteurs par patronyme).
- Le bruit a fait passer la baseline par règles de 74 % à 63 % : un niveau plus crédible, qui
  laisse un vrai résiduel au ML.

---

## 6. Résultats actuels (synthétique, 2 M de paiements, période de test)

| | Automatisation | Précision |
|---|---|---|
| Règles seules | 62,75 % | 99,61 % |
| Règles + ML | **70,13 %** | **99,61 %** |
| dont ML | +7,4 pts | 99,64 % |
| À précision cible, seuil ML optimal a posteriori | 74,2 % | ≥ 99,5 % |

- Par type : 1↔1 83 %, n↔1 68 %, 1↔n 33 %, n↔n 23 % (précision 96,7 % sur les n↔n).
- ~58 000 paiements en revue ; stable d'un mois à l'autre.
- Allocation : rappel 99,16 %.
- Le taux d'automatisation actuel (chiffre client) est à saisir dans `settings.yaml`
  (`evaluation.current_automation_rate`) pour obtenir le gain en points dans la cascade.

---

## 7. Écarts au brief et à la spec (assumés, à valider)

| Sujet | Écart | Raison |
|---|---|---|
| `residual_amount` | solde après la ligne (brief) plutôt qu'écart paiement/facture (spec) | le brief prime ; contrôle de cohérence en place |
| Normalisation | concaténation de 3 tokens adjacents | références découpées par la ponctuation |
| Allocation | 5ᵉ signal « montant » et corroborations nom × montant | rappel 97 % → 99 % |
| Similarité de nom | couverture pondérée ou spécificité (seuil 0,5) au lieu de Jaro-Winkler 0,85 | plus robuste aux noms tronqués |
| R4 « mesurée seule » | sur le résiduel de R1-R3 | coût du DFS sur tous les paiements |
| Seuils ML | par type de proposition, calibrés en ligne, sur le score brut | voir §4 étape 5 |
| n↔n | revue uniquement | heuristique, précision insuffisante pour l'auto-validation |
| Étape 8 (LLM) | non faite | endpoint on-premise non disponible ; à évaluer si le résiduel le justifie |

---

## 8. Volumétrie et performances (2 M paiements, 3 M factures, 16 cœurs)

| Opération | Durée |
|---|---|
| Génération synthétique | ~1 min |
| Chargement + normalisation + journal + contrôles | ~2 min |
| Rejeu, rapprocheur vide, 2 mois | ~15 s |
| Mesure de l'allocation, 2 mois | ~3-4 min |
| Backtest règles, 2 mois | ~3 min |
| Apprentissage (jeu 10 mois + modèle + calibration en ligne) | ~1 h |
| Backtest règles + ML, 2 mois | ~30 min (~30 s/jour) |

Mémoire : ~6 Go au chargement, ~11 Go à l'apprentissage. La normalisation est parallélisée
(`load.workers`, sans effet sur le résultat).

Pour itérer vite : travailler à 50 000 paiements (quelques minutes de bout en bout), puis valider à
2 M.

---

## 9. Utilisation et configuration

- **Notebook** : `notebooks/pipeline.ipynb` — une cellule par étape.
- **API** : `from src.api import Project` — voir `README.md`.
- **Interface** : `streamlit run src/ui/app.py` — mêmes méthodes, lancées dans un processus séparé.
- **Paramètres** : `config/settings.yaml` (commenté, un bloc par étape), `config/rules.yaml`
  (règles versionnées), `config/schema.yaml` (mapping des sources réelles).
- **Données réelles** : remplir `config/schema.yaml`, puis `Project(dataset="real").load()` ; lire le
  profil de chargement (`reports/load_profile.md`) avant d'aller plus loin.

Après une modification de la normalisation, de l'allocation, de la featurisation ou du modèle,
**relancer la chaîne depuis l'étape concernée** (les empreintes signalent les incohérences).

---

## 10. Tests (dans le dépôt, hors du livrable)

`pytest` — ~150 tests, ~3 min. Ils couvrent notamment :

- les deux tests obligatoires du brief : troncature (aucune sortie à D ne dépend d'un événement
  postérieur) et déterminisme (deux rejeux identiques), appliqués à l'état, à l'allocation et au
  rapprocheur complet ;
- le restant dû reconstruit comparé à la formule de référence à plusieurs dates ;
- le jeu figé de non-régression des règles ;
- un rapprocheur « oracle » (précision 100 % attendue) et un volontairement faux pour valider
  l'évaluation ;
- l'interface (toutes les pages, boutons principaux) dans un environnement isolé.

Les tests ne sont pas dans `dist/reconciliation_poc.zip` (construit par `tools/build_bundle.py`).

---

## 11. Points ouverts et prochaines étapes

1. **Données réelles** : c'est la seule mesure qui compte. Points à confirmer au passage :
   interprétation de `residual_amount`, existence d'une date de comptabilisation, format réel des
   client files, sémantique de `bankroll_code`, IBAN partagés débiteur/cédant.
2. **Rappel des candidats ML (~80 %)** : premier levier de gain, surtout sur les 1↔n (factures d'un
   gros débiteur hors du plafond, débiteur au-delà du 3ᵉ rang de l'allocation).
3. **Paiements avec client file** : précision 98,0 %, sous la cible — revoir R1 et l'usage des
   lignes (écarts justifiés par un motif, lignes partielles), idéalement avec le format réel.
4. **Performance du rapprocheur complet** (~30 s/jour à 2 M) : génération de candidats et
   recherche d'ensembles en Python par paiement.
5. **Seuils par segment** (marché, bankroll, tranche de montant, client file) : implémentés,
   désactivés par défaut ; à activer si le volume réel le permet.
6. **Étape 8, LLM sur libellés** : à évaluer sur un échantillon du résiduel, avec cache, seulement si
   le gain est mesurable.

---

## Glossaire

| Terme | Sens |
|---|---|
| `as_of` | instant auquel l'état est lu ; il reflète les événements strictement antérieurs |
| Allocation ferme | un seul débiteur porte un signal fort |
| Candidat | facture proposée au modèle pour un paiement |
| Cascade | tableau taux actuel → règles → règles + ML |
| Journal | suite ordonnée des événements datés dérivée des tables |
| Marge | écart de score entre la meilleure proposition et la seconde |
| Reliquat | paiements non résolus, retraités chaque jour pendant 60 jours |
| Réservation | montant imputé par le moteur, retiré du restant dû jusqu'à l'imputation réelle |
| Résiduel | ce que les règles n'ont pas résolu, traité par le ML |
| τ_high / τ_low | seuils d'auto-validation / de mise en revue |
