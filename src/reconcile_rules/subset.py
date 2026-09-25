"""Recherche bornée de sous-ensembles à somme exacte (règle R4, ensembles de l'étape 5).

Parcours en profondeur sur les montants triés, avec élagage :
- la somme courante dépasse la cible → on coupe (montants croissants) ;
- même en ajoutant les plus grands montants restants, la cible est hors d'atteinte → on coupe ;
- profondeur > `max_size` → on coupe ;
- budget de nœuds épuisé → résultat « indéterminé ».

On s'arrête dès la deuxième solution : l'unicité est tout ce qui compte pour valider.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
