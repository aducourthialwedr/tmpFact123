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

from __future__ import annotations

import numpy as np
import pandas as pd

from src import memory
from src.allocation.allocator import Allocator
from src.allocation.indexes import _flatten
from src.reconcile_rules.subset import UNIQUE, exact_subset
from src.settings import RuleConfig, RulesConfig, Settings
from src.timeline.loop import AUTO, DECISION_COLUMNS, DayContext
from src.timeline.state import LedgerState

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

