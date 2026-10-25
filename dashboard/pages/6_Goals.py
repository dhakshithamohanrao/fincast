import sys
from datetime import date, timedelta
from pathlib import Path

# Inside pages/ the project root is two levels up
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.auth.session import require_login
from app.database.crud import create_goal, delete_goal, get_goals, get_transactions
from src.forecasting.goals import project_goal
from src.forecasting.predictor import forecast_user

st.set_page_config(page_title="Goals - FinCast", layout="wide")

user_id = require_login()

st.title("Savings goals")


@st.cache_data(ttl=600, show_spinner="Calculating your forecast...")
def compute(tx: pd.DataFrame, day: str):
    return forecast_user(tx)


def when(d) -> str:
    return "not reachable at this pace" if d is None or pd.isna(d) else f"{d:%B %Y}"


def remove(goal_id: int) -> None:
    delete_goal(user_id, goal_id)


def render() -> None:
    tx = get_transactions(user_id)
    fc = None
    if not tx.empty:
        try:
            fc = compute(tx, pd.Timestamp.today().strftime("%Y-%m-%d"))
        except (FileNotFoundError, RuntimeError):
            fc = None
    usable = fc is not None and fc.status != "insufficient"

    st.subheader("New goal")
    with st.form("goal_form", clear_on_submit=True):
        name = st.text_input("Goal name", placeholder="Holiday, emergency fund, new laptop...")
        c1, c2 = st.columns(2)
        amount = c1.number_input("Target amount", min_value=0.0, step=100.0, format="%.2f")
        target = c2.date_input("Target date", value=date.today() + timedelta(days=365),
                               min_value=date.today())
        submitted = st.form_submit_button("Add goal")
    if submitted:
        if not name.strip():
            st.error("Please give the goal a name.")
        elif amount <= 0:
            st.error("The target amount must be greater than 0.")
        else:
            create_goal(user_id, name, amount, target.isoformat())
            st.success("Goal added.")

    goals = get_goals(user_id)
    st.subheader("Your goals")
    if goals.empty:
        st.info("No goals yet. Add one above.")
        return
    if not usable:
        st.info("Projections need a forecast: import at least 3 months of transactions first.")
    else:
        st.caption("Savings so far are estimated from the transactions you imported, because "
                   "your real balance is unknown. Each goal is calculated on its own, using "
                   "your whole estimated balance.")
        for note in fc.notes:
            st.caption(note)

    for g in goals.itertuples():
        with st.container(border=True):
            top, btn = st.columns([6, 1])
            top.markdown(f"**{g.name}**: {g.target_amount:,.0f} by {g.target_date:%d.%m.%Y}")
            btn.button("Delete", key=f"del_{g.id}", on_click=remove, args=(g.id,))
            if not usable:
                continue
            income, expenses = fc.income, fc.expenses
            p = project_goal(
                g.target_amount, fc.start_balance,
                income["forecast"] - expenses["forecast"], fc.target_month, g.target_date,
                saving_low=income["lower"] - expenses["upper"],
                saving_high=income["upper"] - expenses["lower"])
            st.progress(p.progress, text=f"{p.progress:.0%} of the goal "
                                         f"({max(fc.start_balance, 0):,.0f} of {g.target_amount:,.0f})")
            if p.status == "reached":
                st.success("You already have enough saved for this goal.")
            elif p.status == "no_savings":
                st.error(f"At the forecast pace ({p.monthly_saving:+,.0f} per month) you are "
                         "not saving anything, so this goal will not be reached. "
                         "Try the What-If page to see what would change that.")
            else:
                text = (f"Projected: **{when(p.projected_date)}** "
                        f"(best case {when(p.date_best)}, worst case {when(p.date_worst)}). "
                        f"Forecast saving: {p.monthly_saving:,.0f} per month.")
                if p.status == "on_track":
                    st.success("On track. " + text)
                else:
                    need = f" You would need about {p.required_monthly:,.0f} per month." \
                        if p.required_monthly else ""
                    st.warning("Behind. " + text + need)


render()

st.divider()
st.caption("For educational purposes, not financial advice.")