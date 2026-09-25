import streamlit as st

from src.ui.common import STATUS_BADGE, STEPS, dataset, load_meta

st.title("Rapprochement automatique paiements / factures")
st.caption("POC — gain de taux d'automatisation à précision fixée (cible ≥ 99,5 %). "
           "Chaque étape ne traite que ce que la précédente n'a pas résolu.")

meta = load_meta()
cols = st.columns(3)
cols[0].metric("Jeu de données", dataset())
cols[1].metric("Événements du journal", f"{meta['journal_events']:,}".replace(",", " ") if meta else "—")
cols[2].metric("Normalisation", f"v{meta['normalization_version']}" if meta else "—")

st.subheader("Étapes")
for step in STEPS.values():
    label, color, icon = STATUS_BADGE[step.status]
    with st.container(border=True):
        left, right = st.columns([5, 1], vertical_alignment="center")
        left.markdown(f"**{step.title}**  \n{step.summary}")
        right.badge(label, icon=icon, color=color)
