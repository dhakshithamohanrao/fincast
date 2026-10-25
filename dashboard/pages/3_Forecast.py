import sys
from pathlib import Path

# Inside pages/ the project root is two levels up
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.auth.session import require_login
from app.components.forecast_charts import forecast_donut, forecast_fan_chart
from app.database.crud import get_transactions, save_forecast
from src.forecasting.predictor import forecast_user
from src.preprocessing.aggregate_monthly import aggregate_monthly

st.set_page_config(page_title="Forecast - FinCast", layout="wide")

user_id = require_login()

st.title("Forecast")


@st.cache_data(ttl=600, show_spinner="Calculating your forecast...")
def compute(tx: pd.DataFrame, day: str):
    """Cached; `day` makes the cache refresh once per day."""
    return forecast_user(tx)


def render() -> None:
    tx = get_transactions(user_id)
    if tx.empty:
        st.info("You have no transactions yet. Open **Import Transactions** to upload a "
                "bank CSV or try the sample file.")
        return

    try:
        fc = compute(tx, pd.Timestamp.today().strftime("%Y-%m-%d"))
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(f"The forecasting model is not available: {exc}")
        return

    if fc.status == "insufficient":
        st.info(fc.message)
        return

    this_month = pd.Timestamp.today().to_period("M").to_timestamp()
    if fc.target_month < this_month:
        st.warning(f"Your latest complete month is before {fc.target_month:%B %Y}. Import "
                   "newer transactions to get a forecast for the coming month.")
    for note in fc.notes:
        st.info(note)

    st.subheader(f"Forecast for {fc.target_month:%B %Y}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected income", f"{fc.income['forecast']:,.0f}")
    c1.caption(f"Likely range {fc.income['lower']:,.0f} to {fc.income['upper']:,.0f}")
    c2.metric("Expected spending", f"{fc.expenses['forecast']:,.0f}")
    c2.caption(f"Likely range {fc.expenses['lower']:,.0f} to {fc.expenses['upper']:,.0f}")
    c3.metric("Expected end balance", f"{fc.end_balance:,.0f}")
    c3.caption(f"Starting from {fc.start_balance:,.0f} (savings in your data)")
    c4.metric("Chance of going below zero", f"{fc.p_negative:.0%}")
    c4.caption("Rough estimate")

    if fc.p_negative >= 0.5:
        st.error("Your balance is more likely than not to end the month below zero.")
    elif fc.p_negative >= 0.2:
        st.warning("There is a noticeable chance your balance ends the month below zero.")
    else:
        st.success("Your balance is unlikely to end the month below zero.")

    history = aggregate_monthly(tx)
    history = history[history["month"] < fc.target_month]
    left, right = st.columns([3, 2])
    left.plotly_chart(forecast_fan_chart(history, fc), use_container_width=True)
    right.plotly_chart(forecast_donut(fc.table), use_container_width=True)

    cat = fc.table[~fc.table["series"].isin(["Income", "Total expenses"])].copy()
    st.subheader("By category")
    shown = cat.assign(
        likely_range=cat["lower"].round(0).astype(int).astype(str) + " to "
        + cat["upper"].round(0).astype(int).astype(str)
    )[["series", "forecast", "likely_range", "last_month", "change"]]
    st.dataframe(
        shown.sort_values("forecast", ascending=False),
        hide_index=True, use_container_width=True,
        column_config={
            "series": "Category",
            "forecast": st.column_config.NumberColumn("Forecast", format="%.0f"),
            "likely_range": "Likely range",
            "last_month": st.column_config.NumberColumn("Last month", format="%.0f"),
            "change": st.column_config.NumberColumn("Change", format="%+.0f"),
        },
    )

    st.subheader("Biggest expected changes versus last month")
    for row in cat.loc[cat["change"].abs().nlargest(3).index].itertuples():
        st.write(f"- **{row.series}**: {row.change:+,.0f} compared with last month")

    # Store the forecast once per user and month so its accuracy can be checked later
    key = f"forecast_saved_{user_id}_{fc.target_month:%Y-%m}"
    if not st.session_state.get(key):
        save_forecast(user_id, fc.target_month, fc.table, fc.model_version)
        st.session_state[key] = True


render()

st.divider()
st.caption("For educational purposes, not financial advice.")