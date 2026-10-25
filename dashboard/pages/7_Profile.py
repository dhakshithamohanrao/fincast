import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st

from app.auth.auth_service import change_password
from app.auth.session import current_username, logout_user, require_login
from app.database import crud

st.set_page_config(page_title="Profile - FinCast", layout="wide")

user_id = require_login()
username = current_username()

st.title("Profile")

user = crud.get_user_by_id(user_id)
st.write(f"Logged in as **{username}** ({user['email'] if user else 'unknown email'})")

INCOME_TYPES = ["salary", "freelance", "student", "other"]
CURRENCIES = ["EUR", "USD", "GBP", "CHF"]

st.subheader("Settings")
profile = crud.get_profile(user_id)
with st.form("profile_form"):
    income_type = st.selectbox(
        "Income type", INCOME_TYPES,
        index=INCOME_TYPES.index(profile["income_type"]) if profile["income_type"] in INCOME_TYPES else 0)
    currency = st.selectbox(
        "Currency", CURRENCIES,
        index=CURRENCIES.index(profile["currency"]) if profile["currency"] in CURRENCIES else 0)
    if st.form_submit_button("Save settings"):
        crud.save_profile(user_id, income_type, currency)
        st.success("Settings saved.")

st.subheader("Change password")
with st.form("password_form", clear_on_submit=True):
    current = st.text_input("Current password", type="password")
    new = st.text_input("New password", type="password")
    again = st.text_input("Repeat new password", type="password")
    if st.form_submit_button("Change password"):
        if new != again:
            st.error("The two new passwords do not match.")
        else:
            ok, message = change_password(user_id, current, new)
            (st.success if ok else st.error)(message)

st.subheader("Export your data")
tx = crud.get_transactions(user_id)
goals = crud.get_goals(user_id)
c1, c2 = st.columns(2)
c1.download_button("Download transactions (CSV)", tx.to_csv(index=False).encode("utf-8"),
                   file_name="fincast_transactions.csv", mime="text/csv",
                   disabled=tx.empty)
c2.download_button("Download goals (CSV)", goals.to_csv(index=False).encode("utf-8"),
                   file_name="fincast_goals.csv", mime="text/csv",
                   disabled=goals.empty)

st.subheader("Delete account")
st.warning("This permanently deletes your account, transactions, forecasts, goals and "
           "settings. It cannot be undone. Download your data first if you want a copy.")
typed = st.text_input(f"Type your username (**{username}**) to confirm")
if st.button("Delete my account and all data", type="primary", disabled=typed != username):
    if crud.delete_user_data(user_id):
        logout_user()
        st.success("Your account and data have been deleted. Go to Home to register again.")
        st.stop()
    st.error("Could not delete the account. Please try again.")

st.divider()
st.caption("For educational purposes, not financial advice.")
