import sys
from pathlib import Path

# Make the project root importable (Streamlit runs scripts from their own folder)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from app.auth.auth_service import login, register
from app.auth.session import (
    current_user_id,
    current_username,
    login_user,
    logout_user,
)
from app.database.db import init_db

st.set_page_config(page_title="FinCast", layout="centered")
init_db()

st.title("FinCast")
st.caption("Personal Finance Forecaster")

if current_user_id():
    st.success(f"Welcome, {current_username()}!")
    st.write("Use the sidebar to open the pages.")
    if st.sidebar.button("Log out"):
        logout_user()
        st.rerun()
else:
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            user = login(username, password)
            if user:
                login_user(user)
                st.rerun()
            else:
                st.error("Wrong username or password.")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Repeat the password", type="password")
            registered = st.form_submit_button("Create account")
        if registered:
            if new_password != confirm_password:
                st.error("The two passwords do not match.")
            else:
                ok, message = register(new_username, new_email, new_password)
                if ok:
                    st.success(message)
                else:
                    st.error(message)

st.divider()
st.caption("For educational purposes, not financial advice.")