import sys
from pathlib import Path

# Inside pages/ the project root is two levels up
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st

from app.auth.session import require_login
from app.database.crud import add_transactions
from src.ingestion.bank_csv_parser import (
    STANDARD_COLUMNS,
    BankCSVError,
    detect_columns,
    parse_bank_csv,
    read_raw_csv,
)
from src.preprocessing.cleaning import clean_transactions

st.set_page_config(page_title="Import - FinCast", layout="wide")

user_id = require_login()

st.title("Import transactions")
st.write(
    "Upload a CSV export from your bank. FinCast detects the columns, cleans the "
    "data and skips transactions you have already imported."
)

uploaded = st.file_uploader("Bank CSV file", type=["csv", "txt"])

if uploaded is None:
    st.info("Choose a file to start. You can try the sample file in data/sample/.")
    st.stop()

# --- Step 1: read the file and let the user check the columns ---
try:
    raw = read_raw_csv(uploaded)
except BankCSVError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("1. Check the columns")
detected = detect_columns(raw.columns)
options = list(raw.columns)
labels = {
    "date": "Date column",
    "description": "Description column",
    "amount": "Amount column",
}

column_map = {}
for ui_col, key in zip(st.columns(3), STANDARD_COLUMNS):
    default = options.index(detected[key]) if key in detected else 0
    column_map[key] = ui_col.selectbox(labels[key], options, index=default)

if len(set(column_map.values())) < len(STANDARD_COLUMNS):
    st.error("Please choose three different columns for date, description and amount.")
    st.stop()

expenses_positive = st.checkbox("My file shows expenses as positive numbers")

# --- Step 2: parse, clean and preview ---
try:
    parsed = parse_bank_csv(uploaded, column_map)
    cleaned = clean_transactions(parsed, expenses_positive=expenses_positive)
except (BankCSVError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

if cleaned.empty:
    st.warning("No valid transactions were found in this file.")
    st.stop()

st.subheader("2. Preview")
income = cleaned.loc[cleaned["amount"] > 0, "amount"].sum()
expenses = -cleaned.loc[cleaned["amount"] < 0, "amount"].sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions", len(cleaned))
m2.metric("Income", f"{income:,.2f}")
m3.metric("Expenses", f"{expenses:,.2f}")
m4.metric(
    "Period",
    f"{cleaned['date'].min():%d.%m.%Y} - {cleaned['date'].max():%d.%m.%Y}",
)

st.dataframe(cleaned.head(20), use_container_width=True, hide_index=True)
if len(cleaned) > 20:
    st.caption(f"Showing the first 20 of {len(cleaned)} rows.")

# --- Step 3: save ---
st.subheader("3. Save")
if st.button("Confirm and save", type="primary"):
    added, skipped = add_transactions(user_id, cleaned, source="upload")
    if added:
        st.success(
            f"Saved {added} new transactions. {skipped} were already in your account and were skipped."
        )
    else:
        st.info(f"Nothing new to save: all {skipped} transactions were already imported.")

st.divider()
st.caption("For educational purposes, not financial advice.")