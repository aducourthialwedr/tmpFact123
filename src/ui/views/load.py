import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from src.config import resolve_path
from src.load.normalize import label_numbers, normalize_text, reference_keys
from src.synthetic.generate import SyntheticConfig
from src.ui.common import (
    SYNTHETIC, current_settings, dataset, interim_dir, load_meta, read_report, run_task, schema_path,
    settings_path, step_header,
)

step_header("load")
settings = current_settings()
ds = dataset()

# --- Lancement -----------------------------------------------------------------------------
st.subheader("Lancer le chargement")
kwargs = {}
if ds == SYNTHETIC:
    default = SyntheticConfig()
    c = st.columns([2, 1, 1])
    n_payments = c[0].number_input("Paiements visés", min_value=1_000, max_value=5_000_000,
                                   value=default.n_payments, step=100_000, key="load.n_payments",
                                   help="Volume réel : environ 2 millions de paiements sur un an.")
    seed = c[1].number_input("Seed", min_value=0, value=default.seed, step=1, key="load.seed")
    regenerate = c[2].checkbox("Régénérer", key="load.regenerate",
                               help="Sinon, le jeu déjà généré pour ce volume et cette seed est réutilisé.")
    variant_dir = resolve_path(settings.paths.synthetic_dir) / f"{int(n_payments)}_seed{int(seed)}"
    st.caption(f"Jeu : `{variant_dir}` — {'existe déjà' if (variant_dir / 'payment.csv').exists() else 'à générer'}. "
               "Comptez ≈ 1 min 30 de génération et ≈ 2 min de chargement pour 2 M de paiements.")
    kwargs = {"n_payments": int(n_payments), "seed": int(seed), "regenerate": bool(regenerate)}
else:
    st.caption(f"Sources décrites par `{schema_path()}` (page Configuration).")

if st.button("Lancer l'étape 1", type="primary", icon=":material/play_arrow:", key="run_load"):
    code, lines = run_task("load", "Étape 1 — chargement", **kwargs)
    if code != 0:
        st.error("Échec du chargement :\n\n" + "\n".join(lines[-8:]))
    else:
        st.rerun()

# --- Résultats -------------------------------------------------------------------------------
meta = load_meta(settings)
if meta is None:
    st.info(f"Aucun chargement pour le jeu **{ds}**.")
    st.stop()

st.subheader("Dernier chargement")
volumes = read_report("load_volumes.csv")
counts = dict(zip(volumes["table"], volumes["rows"])) if volumes is not None else {}
fmt = lambda n: f"{int(n):,}".replace(",", " ")  # noqa: E731
m = st.columns(5)
m[0].metric("Paiements", fmt(counts.get("payment", 0)))
m[1].metric("Factures", fmt(counts.get("invoice", 0)))
m[2].metric("Imputations", fmt(counts.get("imputation", 0)))
m[3].metric("Événements", fmt(meta["journal_events"]))
issues = read_report("load_issues.csv")
n_issues = 0 if issues is None else len(issues[issues["count"] > 0])
m[4].metric("Anomalies", n_issues)
st.caption(f"Source `{meta['source']}` · normalisation v{meta['normalization_version']} · "
           f"journal `{meta['journal_sha256'][:16]}…`")

tabs = st.tabs(["Anomalies", "Champs manquants", "Plages de dates", "Journal", "Temps d'exécution", "Tables"])
with tabs[0]:
    if issues is None or issues.empty or n_issues == 0:
        st.success("Aucune anomalie détectée.", icon=":material/check_circle:")
    else:
        st.dataframe(issues[issues["count"] > 0], hide_index=True, width="stretch")
with tabs[1]:
    missing = read_report("load_missing_fields.csv")
    if missing is not None:
        only = st.toggle("Seulement les champs non mappés ou incomplets", value=True, key="load.only_missing")
        view = missing[(~missing["mapped"]) | (missing["null"] > 0)] if only else missing
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={"null_pct": st.column_config.ProgressColumn(
                         "null %", min_value=0, max_value=100, format="%.1f %%")})
with tabs[2]:
    dates = read_report("load_date_ranges.csv")
    if dates is not None:
        st.dataframe(dates, hide_index=True, width="stretch")
with tabs[3]:
    events = read_report("load_events.csv")
    if events is not None:
        st.dataframe(events, hide_index=True, width="stretch")
with tabs[4]:
    timings = meta.get("timings_s", {})
    st.dataframe(pd.DataFrame({"étape": list(timings), "secondes": list(timings.values())}),
                 hide_index=True, width="content")
with tabs[5]:
    if volumes is not None:
        st.dataframe(volumes, hide_index=True, width="content")

# --- Données normalisées ------------------------------------------------------------------------
st.subheader("Libellés normalisés")
pay_path = interim_dir(settings) / "payment.parquet"
if pay_path.exists():
    pf = pq.ParquetFile(pay_path)
    sample = pf.read_row_group(0, columns=["payment_id", "label", "label_norm", "label_numbers"]).to_pandas()
    sample = sample.sample(min(200, len(sample)), random_state=0).sort_values("payment_id")
    sample["label_numbers"] = sample["label_numbers"].map(lambda v: " · ".join(v))
    st.caption(f"Échantillon de {len(sample)} paiements sur {pf.metadata.num_rows:,}.".replace(",", " "))
    st.dataframe(sample, hide_index=True, width="stretch", height=280)

st.subheader("Testeur de normalisation")
st.caption("Vérifie si une référence de facture est retrouvée dans un libellé, avec la normalisation courante.")
c = st.columns(2)
label = c[0].text_input("Libellé bancaire", "VIR SEPA DUPONT SARL REGLT 12345 FA", key="load.test_label")
ref = c[1].text_input("Référence de facture", "FA0012345", key="load.test_ref")
numbers = label_numbers(normalize_text(label).split())
keys = reference_keys(ref)
common = sorted(set(numbers) & set(keys))
c[0].markdown(f"Normalisé : `{normalize_text(label)}`  \nClés : " + " ".join(f"`{k}`" for k in numbers))
c[1].markdown("Clés : " + " ".join(f"`{k}`" for k in keys))
if common:
    st.success(f"Référence retrouvée via : {', '.join(common)}", icon=":material/check_circle:")
else:
    st.warning("Référence non retrouvée dans le libellé.", icon=":material/search_off:")
