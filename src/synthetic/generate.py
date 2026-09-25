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

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

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


def _days(start: np.datetime64, offsets: np.ndarray) -> np.ndarray:
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
        "opened_at": _days(start, -rng.integers(30, 900, n_a)),
        "closed_at": np.full(n_a, np.datetime64("NaT"), dtype="datetime64[D]"),
    })
    a_weight = rng.pareto(1.5, n_a) + 1
    a_style = np.arange(n_a) % _N_REF_STYLES

    # Débiteurs et comportements.
    n_d = max(5, round(cfg.n_payments * cfg.debtors_per_payment))
    d_iban = _ibans(rng, n_d)
    d_iban[rng.random(n_d) >= 0.9] = ""
    d_opened = _days(start, -rng.integers(-120, 900, n_d))
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
                            _days(start, rng.integers(-400, 90, n_ag)))
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
    o_day = _days(start, rng.integers(0, cfg.n_days, n_o))
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
