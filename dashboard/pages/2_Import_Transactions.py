import sys
from datetime import date
from pathlib import Path

# Inside pages/ the project root is two levels up
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.auth.session import require_login
from app.database.crud import add_transactions, save_corrections
from src.ingestion.bank_csv_parser import (
    STANDARD_COLUMNS,
    BankCSVError,
    detect_columns,
    parse_bank_csv,
    read_raw_csv,
)
from src.preprocessing.categorizer import categorize
from src.preprocessing.cleaning import clean_transactions

CATEGORIES = [
    "Groceries", "Rent", "Utilities", "Transport", "Dining", "Leisure",
    "Health", "Shopping", "Subscriptions", "Savings", "Income", "Other",
]

st.set_page_config(page_title="Import - FinCast", layout="wide")

user_id = require_login()

st.title("Import transactions")


@st.cache_data(show_spinner="Categorising transactions...")
def categorize_cached(df: pd.DataFrame) -> pd.DataFrame:
    """Cached so the model is not reloaded every time the page reruns."""
    return categorize(df)


def render_upload() -> None:
    st.write(
        "Upload a CSV export from your bank. FinCast detects the columns, cleans the "
        "data, assigns categories and skips transactions you have already imported."
    )
    uploaded = st.file_uploader("Bank CSV file", type=["csv", "txt"])
    if uploaded is None:
        st.info("Choose a file to start. You can try the sample file in data/sample/.")
        return

    # Step 1: read the file and let the user check the columns
    try:
        raw = read_raw_csv(uploaded)
    except BankCSVError as exc:
        st.error(str(exc))
        return

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
        return

    expenses_positive = st.checkbox("My file shows expenses as positive numbers")

    # Step 2: parse, clean, categorise and preview
    try:
        parsed = parse_bank_csv(uploaded, column_map)
        cleaned = clean_transactions(parsed, expenses_positive=expenses_positive)
    except (BankCSVError, ValueError) as exc:
        st.error(str(exc))
        return

    if cleaned.empty:
        st.warning("No valid transactions were found in this file.")
        return

    categorized = categorize_cached(cleaned)
    # Money coming in is always income
    categorized.loc[categorized["amount"] > 0, "category"] = "Income"

    st.subheader("2. Preview and fix categories")
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

    st.caption("Wrong category? Click a cell in the Category column and pick the right one.")
    edited = st.data_editor(
        categorized,
        column_config={
            "category": st.column_config.SelectboxColumn(
                "Category", options=CATEGORIES, required=True
            ),
            "category_source": st.column_config.TextColumn("Assigned by"),
        },
        disabled=["date", "description", "amount", "category_source"],
        hide_index=True,
        use_container_width=True,
        key="import_editor",
    )

    changed = edited.loc[edited["category"] != categorized["category"]]
    if len(changed):
        st.info(f"You changed {len(changed)} categories. They are saved when you confirm.")

    # Step 3: save
    st.subheader("3. Save")
    if st.button("Confirm and save", type="primary"):
        to_save = edited.drop(columns=["category_source"])
        added, skipped = add_transactions(user_id, to_save, source="upload")
        fixed = save_corrections(user_id, changed[["description", "category"]])
        if added:
            st.success(
                f"Saved {added} new transactions. "
                f"{skipped} were already in your account and were skipped."
            )
        else:
            st.info(f"Nothing new to save: all {skipped} transactions were already imported.")
        if fixed:
            st.caption(f"{fixed} category corrections stored for future retraining.")


def render_manual() -> None:
    st.write("Add a single transaction by hand.")
    with st.form("manual_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        tx_date = c1.date_input("Date", value=date.today())
        kind = c2.radio("Type", ["Expense", "Income"], horizontal=True)
        description = st.text_input("Description")
        c3, c4 = st.columns(2)
        amount = c3.number_input("Amount", min_value=0.0, step=1.0, format="%.2f")
        category = c4.selectbox("Category (income is always saved as 'Income')", CATEGORIES)
        submitted = st.form_submit_button("Add transaction")

    if submitted:
        if not description.strip():
            st.error("Please enter a description.")
        elif amount <= 0:
            st.error("The amount must be greater than 0.")
        else:
            signed = amount if kind == "Income" else -amount
            if kind == "Income":
                category = "Income"
            row = pd.DataFrame(
                {
                    "date": [pd.Timestamp(tx_date)],
                    "description": [description.strip()],
                    "amount": [signed],
                    "category": [category],
                }
            )
            added, _ = add_transactions(user_id, row, source="manual")
            if added:
                st.success("Transaction saved.")
            else:
                st.info("This exact transaction (same date, description and amount) already exists.")


tab_upload, tab_manual = st.tabs(["Upload CSV", "Add manually"])
with tab_upload:
    render_upload()
with tab_manual:
    render_manual()

st.divider()
st.caption("For educational purposes, not financial advice.")