# Spécification — Moteur de rapprochement automatique paiement / facture

Contexte : affacturage. Le cédant vend ses créances, le débiteur paie, les flux peuvent transiter par des comptes techniques (sous-participation, compte de liaison). Historique disponible : 1 an.

---

## 1. Architecture

Quatre étages, à implémenter dans cet ordre :

| Étage | Rôle | Sortie |
|---|---|---|
| **A. Génération de candidats** | Réduire l'espace de recherche par règles dures | Paires `(payment_id, invoice_id)` plausibles |
| **B. Scoring de paires** | Modèle ML supervisé, une probabilité par paire | `score ∈ [0,1]` |
| **C. Résolution d'ensembles** | Reconstituer les cas 1↔n, n↔1, n↔n | Groupes `{payment_ids} ↔ {invoice_ids}` |
| **D. Décision** | Seuils, auto-validation, file de revue | Imputation ou tâche humaine |

L'étage C est indispensable : un score de paire seul ne sait pas dire « ce virement de 12 480 € solde ces trois factures ».

---

## 2. Modèle de données d'entrée

Normalisation attendue avant tout traitement. Tous les montants en **entiers de centimes** (jamais de float), toutes les dates en `DATE` ou `TIMESTAMP` UTC.

### 2.1 `payment`

| Champ | Type | Notes |
|---|---|---|
| `payment_id` | PK | |
| `value_date` | date | date de valeur, pivot temporel |
| `amount` | int (centimes) | signé : négatif = reversement/rejet |
| `currency` | char(3) | ISO 4217 |
| `iban_debtor` | varchar | IBAN émetteur |
| `iban_creditor` | varchar | IBAN récepteur |
| `label` | text | libellé bancaire brut |
| `channel` | enum | SEPA, SWIFT, chèque, LCR… |
| `payment_type` | enum | |
| `bankroll_code` | enum | **à rapatrier ici si absent** — qualifie la nature du compte |

### 2.2 `invoice`

| Champ | Type | Notes |
|---|---|---|
| `invoice_id` | PK | |
| `client_reference` | varchar | référence côté cédant, souvent celle citée en libellé |
| `creation_date` | date | |
| `due_date` | date | échéance |
| `initial_amount` | int (centimes) | |
| `current_amount` | int (centimes) | restant dû — **piège de fuite, cf. §5.2** |
| `currency` | char(3) | |
| `debtor_id` | FK → `debtor` | |
| `agreement_id` | FK → `agreement` | |

### 2.3 `imputation` (table de liaison — source des labels)

| Champ | Type | Notes |
|---|---|---|
| `payment_id` | FK | |
| `invoice_id` | FK | |
| `status` | enum | `FULL` / `PARTIAL` |
| `updated_at` | timestamp | **clé pour reconstruire l'état passé** |
| `residual_amount` | int (centimes) | écart paiement / facture |

### 2.4 `assignor` (cédant) et `debtor`

| Champ | Type |
|---|---|
| `party_id` | PK |
| `bankroll_code` | enum |
| `iban` | varchar |
| `name` | varchar |
| `opened_at` / `closed_at` | date |

### 2.5 `agreement`

| Champ | Type | Notes |
|---|---|---|
| `agreement_id` | PK | |
| `debtor_id`, `client_id` | FK | |
| `contract_number` | varchar | |
| `created_at` / `disabled_at` | date | |
| `market` | enum | BTP, industrie, services… |
| `product` | enum | |
| `recourse` | bool/enum | avec ou sans recours |

---

## 3. Prétraitement

### 3.1 Normalisation du libellé bancaire

Pipeline déterministe, versionné (le modèle doit être reproductible) :

1. Passage en majuscules, suppression des accents (NFKD).
2. Remplacement de tout caractère non alphanumérique par un espace.
3. Compression des espaces multiples.
4. Extraction de deux vues :
   - `label_tokens` : liste de tokens alphabétiques (pour la similarité de nom).
   - `label_numbers` : liste de tokens contenant des chiffres, **plus leurs variantes sans zéros de tête et sans préfixe alphabétique** (`FA0012345` → `{FA0012345, 0012345, 12345}`).

C'est cette dernière normalisation qui fait la différence sur le taux de match par référence : les débiteurs tronquent, préfixent et zéro-paddent les références de manière très inconstante.

### 3.2 Résolution d'IBAN conditionnée au bankroll

Ne **jamais** faire un `iban_debtor = debtor.iban` naïf. Construire une fonction de routage :

```
resolve_iban(payment) -> {
    "DEBTOR_DIRECT",      # IBAN connu, appartient au débiteur
    "ASSIGNOR",           # IBAN du cédant → reversement probable
    "TECHNICAL_ACCOUNT",  # compte de liaison / sous-participation
    "UNKNOWN"
}
```

Le routage se fait par lookup sur `assignor.iban` / `debtor.iban`, arbitré par `bankroll_code`. Le résultat est une **variable catégorielle** exposée au modèle, pas un booléen.

---

## 4. Étage A — Génération de candidats

Objectif : pour chaque paiement, produire un ensemble réduit de factures plausibles, avec un **rappel proche de 100 %**. Un candidat manqué ici est définitivement perdu.

**Filtres durs :**
- Même devise (sauf activation du mode cross-currency, cf. §7.3).
- `invoice.current_amount_as_of(payment.value_date) > 0` (facture ouverte à la date du paiement).
- Facture non annulée, agreement actif à la date de facture.

**Clés de blocking (union des ensembles produits) :**

| Clé | Règle |
|---|---|
| `K1` — Débiteur | Factures du débiteur identifié par l'IBAN (si `DEBTOR_DIRECT`), fenêtre `[value_date − 180j, value_date + 30j]` |
| `K2` — Référence | Factures dont `client_reference` normalisée ∈ `label_numbers` (aucune contrainte temporelle) |
| `K3` — Montant exact | Factures dont `current_amount == payment.amount`, fenêtre ± 90 j |
| `K4` — Nom | Factures dont le nom du débiteur a une similarité ≥ 0.85 avec un n-gramme du libellé, fenêtre ± 90 j |

`K2` doit être sans fenêtre : une référence exacte trouvée dans un libellé est un signal fort même à 18 mois.

**Métrique de contrôle :** rappel du blocking mesuré sur l'historique (`% d'imputations réelles dont la paire figure dans les candidats`). Cible ≥ 99 %. Mesurer aussi le nombre médian de candidats par paiement — si > 200, resserrer les fenêtres.

---

## 5. Étage B — Scoring de paires

### 5.1 Formulation

**LightGBM**, objectif `binary` avec `scale_pos_weight` ajusté, ou `lambdarank` groupé par `payment_id`. Commencer par le binaire : plus simple à calibrer, et la calibration est nécessaire pour l'étage D.

- Un exemple = une paire candidate `(payment, invoice)`.
- Label positif : la paire existe dans `imputation`.
- Label négatif : tous les autres candidats générés par l'étage A pour ce paiement.

Ne pas échantillonner les négatifs au hasard dans le référentiel : ce sont les **négatifs difficiles issus du blocking** qui font apprendre quelque chose au modèle.

### 5.2 ⚠️ Fuite de données sur `current_amount`

`invoice.current_amount` est mis à jour **après** rapprochement. L'utiliser tel quel en entraînement, c'est donner la réponse au modèle : une facture à 0 est une facture déjà soldée.

Il faut reconstruire l'état à la date du paiement, à partir de `imputation.updated_at` :

```sql
current_amount_as_of(invoice, t) =
    invoice.initial_amount
  - SUM(imputation.imputed_amount
        WHERE imputation.invoice_id = invoice.invoice_id
          AND imputation.updated_at < t)
```

Cette fonction doit être utilisée partout : blocking, features, entraînement, inférence. C'est le point le plus susceptible de produire un modèle brillant en test et médiocre en production.

### 5.3 Features

#### Famille montant

| Feature | Calcul |
|---|---|
| `amount_diff_abs` | `payment.amount − invoice.current_amount_as_of` |
| `amount_diff_rel` | `amount_diff_abs / invoice.current_amount_as_of` |
| `amount_exact_match` | booléen |
| `payment_covers_invoice` | `payment.amount ≥ current_amount` |
| `amount_ratio` | `payment.amount / current_amount` — capte les paiements groupés (ratio > 1) |
| `is_typical_discount` | écart relatif ∈ [0.5 %, 3 %] → escompte |
| `is_bank_fee_gap` | écart absolu ∈ [5 €, 40 €] → frais SWIFT |
| `is_retention_gap` | écart relatif ∈ [4 %, 6 %] → retenue de garantie (BTP) |

Les trois derniers sont des features métier explicites. Un arbre peut les retrouver seul, mais les donner accélère beaucoup la convergence sur un an d'historique.

#### Famille temporelle

| Feature | Calcul |
|---|---|
| `days_to_due` | `value_date − due_date` (**signé** : les retards structurels sont informatifs) |
| `days_since_creation` | `value_date − creation_date` |
| `is_before_creation` | booléen — un paiement antérieur à la facture est presque toujours un faux candidat |
| `days_to_due_zscore` | écart normalisé par la distribution de délai historique du couple (débiteur, agreement) |

#### Famille textuelle / référence

| Feature | Calcul |
|---|---|
| `ref_exact_in_label` | `client_reference` normalisée ∈ `label_numbers` |
| `ref_partial_in_label` | correspondance sur suffixe ≥ 5 caractères |
| `invoice_id_in_label` | idem sur l'identifiant interne |
| `name_jaro_winkler` | max sur les n-grammes du libellé vs `debtor.name` |
| `name_token_set_ratio` | robuste aux formes sociales (SARL, SAS) et à l'ordre des tokens |
| `label_length`, `label_has_no_alpha` | qualité du libellé — un libellé vide change la fiabilité des autres features |

#### Famille identité / structure

| Feature | Calcul |
|---|---|
| `iban_route` | catégoriel : `DEBTOR_DIRECT` / `ASSIGNOR` / `TECHNICAL_ACCOUNT` / `UNKNOWN` |
| `iban_matches_invoice_debtor` | booléen, valide uniquement si `iban_route = DEBTOR_DIRECT` |
| `bankroll_code` | catégoriel |
| `channel`, `payment_type` | catégoriels |
| `same_agreement` | le paiement est-il rattachable au même contrat |
| `assignor_active_at_value_date` | `closed_at` postérieure à la date de valeur |
| `agreement_active_at_creation` | idem sur `disabled_at` |

#### Famille comportementale (agrégats historiques)

Calculés en fenêtre glissante **strictement antérieure** à `value_date`, jamais sur la totalité de l'historique (fuite).

| Feature | Calcul |
|---|---|
| `debtor_mean_payment_delay` | délai moyen observé, 6 mois glissants |
| `debtor_std_payment_delay` | régularité |
| `debtor_partial_payment_rate` | propension au paiement partiel |
| `debtor_grouping_rate` | % d'imputations appartenant à un groupe n↔1 |
| `debtor_open_invoice_count` | nombre de factures ouvertes au moment du paiement |
| `debtor_open_invoice_amount` | encours total |
| `debtor_ref_citation_rate` | ce débiteur cite-t-il habituellement la référence en libellé |
| `debtor_payment_count` | volume — pondère la confiance des agrégats ci-dessus |

`debtor_ref_citation_rate` est particulièrement utile : elle apprend au modèle que l'absence de référence est un signal négatif fort chez un débiteur rigoureux, et neutre chez un débiteur qui n'en met jamais.

#### Famille contexte contrat

`market` (BTP…), `product`, `recourse` en catégoriels. Le BTP a des distributions de délais et des retenues très spécifiques.

#### Famille compétition (calculée après un premier passage)

| Feature | Calcul |
|---|---|
| `rank_in_payment` | rang du score brut parmi les candidats du paiement |
| `score_margin_to_second` | écart au deuxième candidat |
| `n_candidates` | nombre de candidats du paiement |

Implémentation en deux passes : un modèle de base produit les scores bruts, ces trois features sont dérivées, un second modèle produit le score final. Gain typique important sur la précision, parce que l'ambiguïté devient une variable observable.

### 5.4 Entraînement et évaluation

- **Split temporel obligatoire.** Ex. : 8 mois train, 2 mois validation, 2 mois test. Jamais de split aléatoire : les groupes n↔n se retrouveraient des deux côtés.
- **Purge** de quelques jours entre les blocs pour éviter les chevauchements de groupes.
- Métriques :
  - `recall@blocking` (étage A)
  - `precision@1` et `MRR` sur les paiements 1↔1
  - **taux d'automatisation à précision fixée** : % de paiements auto-validés pour une précision ≥ 99,5 % — c'est la métrique métier, les autres sont des diagnostics
  - matrice de coût asymétrique : un faux positif (imputation erronée) coûte bien plus qu'un faux négatif (revue manuelle)

La construction concrète du jeu d'entraînement — rejeu de l'historique dans l'ordre d'arrivée des événements — est décrite au **§8**. Elle n'est pas optionnelle : c'est elle qui garantit que les features des §5.2 et §5.3 sont calculées sur un état de base réellement disponible au moment de la décision.

---

## 6. Étage C — Résolution d'ensembles

### 6.1 Cas 1↔n (un paiement, plusieurs factures)

Recherche de sous-ensemble sommant au montant du paiement, sur les factures ouvertes du même débiteur :

```python
def find_subsets(payment, candidates, max_k=5, tol_abs=500, tol_rel=0.03):
    # candidates triés par score de paire décroissant, tronqués à ~25
    # DFS avec élagage :
    #   - somme courante > payment.amount + tolérance → couper
    #   - profondeur > max_k → couper
    #   - somme des scores du sous-ensemble < seuil → couper
    # retourner les sous-ensembles valides, classés par
    #   (moyenne des scores de paire, −écart au montant, −cardinalité)
```

Paramètres de départ : `max_k = 5`, tolérance 5 € ou 3 %. Les sous-ensembles de cardinalité minimale sont préférés à égalité de score — la parcimonie évite les combinaisons fortuites.

Contrôle d'explosion : borner à 25 candidats en entrée du DFS et poser un budget de nœuds explorés. Au-delà, basculer directement en revue manuelle.

### 6.2 Cas n↔1 (plusieurs paiements, une facture)

Traité par l'état : chaque paiement s'impute sur `current_amount_as_of`. Un paiement qui solde partiellement laisse un reliquat, le suivant s'impute dessus. Aucune combinatoire nécessaire, à condition que `current_amount_as_of` soit correctement implémentée.

### 6.3 Cas n↔n

Traité en deux temps : d'abord regrouper les paiements du même débiteur dans une fenêtre courte (72 h) en un paiement virtuel agrégé, puis appliquer §6.1. C'est une heuristique, pas une résolution exacte — accepter un taux d'automatisation plus faible sur ce segment et privilégier la revue humaine.

### 6.4 Résolution des conflits

Une facture ne peut être imputée deux fois. Après scoring, résoudre l'affectation globale par un algorithme glouton sur les scores décroissants, ou par assignation hongroise sur les cas 1↔1. Le glouton suffit dans un premier temps.

---

## 7. Étage D — Décision et exploitation

### 7.1 Calibration et seuils

Le score LightGBM brut n'est pas une probabilité. Appliquer une **régression isotonique** sur le jeu de validation avant de poser des seuils.

| Zone | Condition | Action |
|---|---|---|
| Auto-validation | `p ≥ τ_high` **et** `margin_to_second ≥ δ` | Imputation automatique |
| Revue | `τ_low ≤ p < τ_high` | File de travail, proposition pré-remplie et classée |
| Rejet | `p < τ_low` | Paiement non affecté, en attente |

`τ_high` se calibre par la précision cible, pas à la main : choisir la valeur qui donne ≥ 99,5 % de précision sur la validation, puis la vérifier sur le test.

La condition sur la marge est importante : deux factures identiques du même débiteur peuvent toutes deux scorer 0.97, et il faut alors une intervention humaine même si la confiance absolue est haute.

Prévoir des seuils **différenciés par segment** (`market`, `bankroll_code`, montant) : le BTP tolérera probablement moins d'automatisation, et les gros montants méritent un `τ_high` plus élevé indépendamment du modèle.

### 7.2 Boucle d'apprentissage

- Toute décision humaine dans la file de revue est journalisée comme label, avec la raison du rejet quand elle est saisie.
- Réentraînement mensuel, avec comparaison systématique au modèle en production sur le mois écoulé avant bascule.
- **Active learning** : prioriser dans la file de revue les cas proches du seuil, qui apportent le plus d'information au modèle.
- Monitoring de dérive : distribution des scores, taux d'automatisation, taux de correction des auto-validations (métrique de sécurité principale — toute remontée doit déclencher une alerte).

### 7.3 Points ouverts à trancher

- **Cross-currency** : à activer seulement si le volume le justifie. Nécessite une table de taux et une tolérance élargie pour absorber le spread.
- **Reversements cédant** (`iban_route = ASSIGNOR`) : probablement à exclure du périmètre d'automatisation en v1 et à router vers un traitement dédié.
- **Montants négatifs** (rejets, impayés) : à sortir du modèle de rapprochement, logique métier distincte.
- **Écriture des imputations** : le moteur doit-il écrire directement en base ou produire des propositions validées par un batch ? Recommandation : propositions en v1, écriture directe une fois la précision mesurée sur trois mois de production.

---

## 8. Exécution de la pipeline

Tout ce qui précède décrit *ce que* calcule le moteur. Ce chapitre décrit *dans quel ordre* — et c'est le point qui conditionne la validité de l'ensemble.

Le principe directeur tient en une phrase : **le rapprochement est un problème événementiel, pas un problème sur table**. Une facture existe à partir d'une date, un paiement arrive à une date, une imputation est prononcée à une date. Toute feature calculée sans référence explicite à un instant `t` est suspecte.

Conséquence pratique : **un seul et même code d'exécution sert à l'entraînement et à la production**. En entraînement il rejoue le passé, en production il traite le présent. Deux implémentations séparées finissent toujours par diverger, et la divergence se manifeste par un modèle qui sous-performe en prod sans explication.

### 8.1 Le journal d'événements

Première étape d'implémentation : dériver des six tables un journal ordonné.

| Type d'événement | Horodatage | Source |
|---|---|---|
| `INVOICE_CREATED` | `invoice.creation_date` | `invoice` |
| `PAYMENT_RECEIVED` | `payment.value_date` | `payment` |
| `IMPUTATION_APPLIED` | `imputation.updated_at` | `imputation` |
| `PARTY_OPENED` / `PARTY_CLOSED` | `opened_at` / `closed_at` | `assignor`, `debtor` |
| `AGREEMENT_CREATED` / `AGREEMENT_DISABLED` | `created_at` / `disabled_at` | `agreement` |

Deux précautions sur l'ordonnancement :

- **Départager les ex æquo par une règle déterministe** (type d'événement, puis identifiant). Beaucoup de champs sont des dates sans heure : plusieurs dizaines d'événements partagent le même horodatage. Sans règle de départage, deux exécutions produisent deux jeux d'entraînement différents et le modèle n'est plus reproductible.
- **`value_date` n'est pas la date de connaissance.** La date de valeur peut être antérieure de quelques jours à la date à laquelle le paiement est réellement visible dans le système. Si une date de comptabilisation existe quelque part, c'est elle qui doit ordonner le journal, `value_date` restant une simple feature. À vérifier côté SI — c'est une source de fuite discrète mais réelle.

### 8.2 Le moteur d'état

Un objet unique qui répond aux questions du moteur pour un instant donné :

```python
class LedgerState:
    def open_invoices(self, debtor_id, as_of) -> list[Invoice]: ...
    def current_amount(self, invoice_id, as_of) -> int: ...
    def party_is_active(self, party_id, as_of) -> bool: ...
    def behavioral_stats(self, debtor_id, as_of, window_days=180) -> dict: ...

    def apply(self, event) -> None: ...   # avance l'état
```

Toutes les features des §5.2 et §5.3 se calculent **exclusivement** à travers cette interface, avec un `as_of` explicite. C'est la garantie mécanique d'absence de fuite : si une feature n'a pas besoin de `as_of`, c'est qu'elle est constante dans le temps, et il faut pouvoir le justifier.

Les agrégats comportementaux sont les plus coûteux. Les maintenir de façon incrémentale dans `LedgerState` (compteurs mis à jour à chaque `apply`) plutôt que par requête à chaque appel — sinon le replay d'un an devient inexploitable en temps de calcul.

### 8.3 Rejeu de l'historique (construction du jeu d'entraînement)

```
state = LedgerState(vide)

pour chaque événement e du journal, par ordre chronologique :

    si e est PAYMENT_RECEIVED :
        t          = e.timestamp
        candidats  = blocking(e.payment, state, as_of=t)      # étage A
        features   = [featurize(e.payment, inv, state, as_of=t)
                      for inv in candidats]
        écrire (features, payment_id, invoice_id, t) dans le dataset

    state.apply(e)      # ← APRÈS extraction, jamais avant
```

L'ordre des deux dernières lignes est l'essentiel du chapitre. Les features sont extraites sur l'état *avant* application de l'événement ; l'état n'intègre le paiement et ses imputations qu'ensuite.

Le labelling se fait dans une seconde passe, une fois le journal entièrement rejoué : pour chaque ligne du dataset, la paire est positive si elle figure dans `imputation`. Rejeu et labelling sont séparés pour que la première passe n'ait aucun accès à la vérité terrain — la séparation est structurelle, pas seulement disciplinaire.

Sortie : un dataset partitionné par mois, colonne `t` conservée. C'est elle qui sert au découpage temporel du §5.4 ; le split se fait sur `t`, pas sur un index de ligne.

Le rejeu est **rejouable à l'identique** : même journal, même code, même dataset. Versionner le hash du journal et de la fonction de featurisation avec chaque modèle entraîné.

### 8.4 Exécution en production

Même code, curseur avancé sur les événements nouveaux au lieu de l'historique.

Cadence : un batch par jour, après intégration du relevé bancaire.

```
1. Ingestion       → nouveaux paiements et factures du jour, ajoutés au journal
2. Avance d'état   → LedgerState.apply() jusqu'à la veille incluse
3. Pour chaque paiement non affecté (nouveau + reliquat des jours précédents) :
       a. blocking                                       (étage A)
       b. scoring de paires, passe 1 puis passe 2        (étage B)
       c. recherche de sous-ensembles                    (étage C)
4. Résolution des conflits sur l'ensemble du lot         (§6.4)
5. Application des seuils                                (étage D)
6. Écriture : imputations auto-validées / file de revue
7. Réinjection des décisions humaines de la veille dans le journal
```

Trois points d'attention :

- **L'étape 3 traite le reliquat, pas seulement le jour J.** Un paiement non affecté doit être rescoré chaque jour : la facture correspondante n'est peut-être pas encore créée au moment où le virement arrive. Prévoir une fenêtre de rétention (60 jours par exemple) au-delà de laquelle le paiement sort du cycle automatique.
- **L'étape 4 est globale au lot, pas par paiement.** Deux paiements du même débiteur traités le même jour peuvent revendiquer la même facture ; l'arbitrage doit voir les deux.
- **L'étape 7 ferme la boucle** décrite au §7.2. Les décisions humaines sont des événements du journal au même titre que les autres, donc immédiatement disponibles pour les agrégats comportementaux du lendemain.

### 8.5 Validation du dispositif

Avant toute mise en production, deux contrôles :

**Backtest en replay.** Rejouer les deux derniers mois avec le modèle entraîné sur les huit premiers, en mode production complet (étages A à D, seuils inclus), et comparer les imputations produites à celles réellement prononcées. C'est la seule mesure honnête du taux d'automatisation — les métriques du §5.4 mesurent le modèle, ce backtest mesure le système.

**Shadow run.** Faire tourner le batch quotidien en parallèle du processus manuel pendant quatre à six semaines, sans écriture. Comparer chaque jour. C'est là que se calibrent les seuils définitifs et que se révèlent les cas métier absents de l'historique.

---

## 9. Séquencement proposé

1. **Journal d'événements et `LedgerState`** (§8.1, §8.2). À faire avant tout le reste : c'est le socle sur lequel s'appuient le prétraitement, l'entraînement et la production.
2. Prétraitement + `current_amount_as_of` + résolution IBAN/bankroll.
3. Étage A seul, mesure du rappel de blocking. **Ne pas avancer tant que < 99 %.**
4. Baseline par règles (référence exacte + montant exact) → donne le plancher à battre.
5. Rejeu de l'historique, production du dataset (§8.3).
6. Étage B, features des familles montant / temporel / textuel uniquement.
7. Ajout des familles comportementale et contexte, mesure du gain.
8. Deuxième passe avec features de compétition.
9. Étage C.
10. Calibration, seuils, file de revue.
11. Backtest en replay, puis shadow run (§8.5).

Les étapes 3 et 4 sont les plus rentables : elles fixent le plafond de performance et le point de comparaison honnête. Un modèle qui ne bat pas nettement la baseline par règles ne mérite pas d'aller en production.

L'étape 1 est celle qu'on est le plus tenté de sauter — construire un dataset à plat en quelques requêtes SQL va beaucoup plus vite. C'est presque toujours une erreur : le dataset obtenu contient des fuites difficiles à localiser, et le code d'entraînement ne peut pas être réutilisé en production.
