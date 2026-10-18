import sys
from pathlib import Path

# Inside pages/ the project root is two levels up
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st

from app.auth.session import current_username, require_login

st.set_page_config(page_title="Overview - FinCast", layout="wide")

require_login()

st.title("Overview")
st.write(f"Logged in as **{current_username()}**.")
st.info("Your income and spending charts will appear here once you import transactions.")

st.divider()
st.caption("For educational purposes, not financial advice.")