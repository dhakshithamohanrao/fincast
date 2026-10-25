import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.auth.session import require_login
from app.database.crud import get_transactions
from src.forecasting.predictor import forecast_user

st.set_page_config(page_title="What If - FinCast", layout="wide")

user_id = require_login()

st.title("What if...")


@st.cache_data(ttl=600, show_spinner="Calculating your forecast...")
def compute(tx: pd.DataFrame, day: str):
    return forecast_user(tx)


def render() -> None:
    tx = get_transactions(user_id)
    if tx.empty:
        st.info("Import some transactions first.")
        return
    try:
        fc = compute(tx, pd.Timestamp.today().strftime("%Y-%m-%d"))
    except (FileNotFoundError, RuntimeError):
        st.error("The forecast model is not available.")
        return
    if fc.status == "insufficient":
        st.info(fc.message)
        return

    table = fc.table
    cats = [s for s in table["series"] if s not in ("Income", "Total expenses")]
    st.write(f"See how a change in one spending category would affect "
             f"**{fc.target_month:%B %Y}**.")

    c1, c2 = st.columns(2)
    category = c1.selectbox("Category", cats)
    pct = c2.slider("Change in this category (%)", -50, 50, -10, step=5)

    base_cat = float(table.loc[table["series"] == category, "forecast"].iloc[0])
    income = fc.income["forecast"]
    base_expenses = fc.expenses["forecast"]
    new_expenses = base_expenses + base_cat * pct / 100
    base_saving = income - base_expenses
    new_saving = income - new_expenses
    diff = new_saving - base_saving

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{category} forecast", f"{base_cat * (1 + pct / 100):,.0f}",
              f"{base_cat * pct / 100:+,.0f}", delta_color="inverse")
    m2.metric("Monthly saving", f"{new_saving:,.0f}", f"{diff:+,.0f}")
    m3.metric("Over 12 months", f"{diff * 12:+,.0f}",
              help="Extra (or lost) savings if you keep this change for a year.")

    st.caption(f"Without any change you are forecast to save {base_saving:,.0f} "
               f"in {fc.target_month:%B %Y}. Other categories stay as forecast.")
    for note in fc.notes:
        st.caption(note)


render()

st.divider()
st.caption("For educational purposes, not financial advice.")