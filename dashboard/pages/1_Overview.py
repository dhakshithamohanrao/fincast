import sys
from pathlib import Path

# Inside pages/ the project root is two levels up
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st

from app.auth.session import current_username, require_login
from app.components.cards import format_money, kpi_row
from app.components.charts import category_donut, income_vs_spending_chart
from app.database.crud import get_transactions
from src.preprocessing.aggregate_monthly import aggregate_monthly

st.set_page_config(page_title="Overview - FinCast", layout="wide")

user_id = require_login()

st.title("Overview")
st.caption(f"Logged in as {current_username()}")

transactions = get_transactions(user_id)

if transactions.empty:
    st.info(
        "No transactions yet. Open **Import Transactions** in the sidebar to upload "
        "a bank CSV or add a transaction by hand."
    )
    st.stop()

monthly = aggregate_monthly(transactions)
latest = monthly.iloc[-1]

# --- KPI row for the latest month with data ---
st.subheader(latest["month"].strftime("%B %Y"))
income = float(latest["income"])
spending = float(latest["expenses"])
savings = float(latest["savings"])
savings_rate = f"{savings / income * 100:.0f} %" if income > 0 else "n/a"

kpi_row(
    [
        ("Income", format_money(income), None),
        ("Spending", format_money(spending), None),
        ("Savings", format_money(savings), None),
        ("Savings rate", savings_rate, None),
    ]
)

# --- Charts ---
left, right = st.columns([3, 2])
with left:
    st.plotly_chart(income_vs_spending_chart(monthly.tail(12)), width="stretch")
with right:
    this_month = transactions[transactions["date"] >= latest["month"]]
    st.plotly_chart(category_donut(this_month), width="stretch")

# --- Recent transactions ---
st.subheader("Recent transactions")
recent = (
    transactions.sort_values(["date", "id"], ascending=False)
    .head(10)[["date", "description", "amount", "category"]]
    .copy()
)
recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
st.dataframe(recent, width="stretch", hide_index=True)

st.divider()
st.caption("For educational purposes, not financial advice.")
