import sys
from pathlib import Path

# Inside pages/ the project root is two levels up
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.auth.session import require_login
from app.components.forecast_charts import anomaly_scatter
from app.database.crud import get_anomaly_reviews, get_transactions, save_anomaly_review
from src.forecasting.anomalies import MIN_TRANSACTIONS, detect_anomalies

st.set_page_config(page_title="Anomalies - FinCast", layout="wide")

user_id = require_login()

st.title("Unusual transactions")


@st.cache_data(ttl=600, show_spinner="Checking your transactions...")
def flags_for(tx: pd.DataFrame, max_model_flags: int) -> pd.DataFrame:
    return detect_anomalies(tx, max_model_flags)


def review(transaction_id: int, status: str) -> None:
    save_anomaly_review(user_id, transaction_id, status)


def render() -> None:
    tx = get_transactions(user_id)
    if len(tx) < MIN_TRANSACTIONS:
        st.info(f"You have {len(tx)} transactions. FinCast needs at least {MIN_TRANSACTIONS} "
                "to tell what is unusual for you. Import more history first.")
        return

    st.write("FinCast looks for charges that repeat, amounts far outside your usual range, "
             "and combinations a statistical model finds odd. **Rule flags** are the more "
             "reliable ones; **model suggestions** are only hints and are often fine.")
    max_model = st.slider("Model suggestions to show (rule flags are always shown)", 0, 30, 15)
    try:
        flags = flags_for(tx, max_model)
    except FileNotFoundError:
        st.error("The anomaly model file is missing (models/anomaly.joblib).")
        return

    reviews = get_anomaly_reviews(user_id)
    flags = flags.assign(status=flags["id"].map(reviews).fillna("pending"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flagged", len(flags))
    m2.metric("To review", int((flags["status"] == "pending").sum()))
    m3.metric("Confirmed problems", int((flags["status"] == "confirmed").sum()))
    m4.metric("Dismissed", int((flags["status"] == "dismissed").sum()))

    st.plotly_chart(anomaly_scatter(tx, flags), use_container_width=True)

    show_reviewed = st.checkbox("Also show items I have already reviewed")
    shown = flags if show_reviewed else flags[flags["status"] == "pending"]
    if shown.empty:
        st.success("Nothing left to review.")
        return

    st.caption("**Confirm** = this is a real problem (for example a double charge). "
               "**Dismiss** = this is fine.")
    for row in shown.itertuples():
        with st.container(border=True):
            c1, c2, c3 = st.columns([5, 2, 2])
            c1.markdown(f"**{row.description}**  \n{row.date:%d.%m.%Y} · {row.category}")
            c1.caption(row.reason)
            c2.markdown(f"**{row.amount:,.2f}**")
            c2.caption("Rule flag" if row.source == "rule" else "Model suggestion")
            if row.status != "pending":
                c2.caption(f"You marked this: {row.status}")
            if row.status != "confirmed":
                c3.button("Confirm", key=f"confirm_{row.id}", on_click=review,
                          args=(row.id, "confirmed"), use_container_width=True)
            if row.status != "dismissed":
                c3.button("Dismiss", key=f"dismiss_{row.id}", on_click=review,
                          args=(row.id, "dismissed"), use_container_width=True)


render()

st.divider()
st.caption("For educational purposes, not financial advice.")