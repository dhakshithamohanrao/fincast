"""Login state for the Streamlit app, stored in st.session_state."""
from typing import Optional

import streamlit as st


def login_user(user) -> None:
    """Remember the logged-in user for this browser session."""
    st.session_state["user_id"] = user["id"]
    st.session_state["username"] = user["username"]


def logout_user() -> None:
    """Forget the logged-in user."""
    st.session_state.pop("user_id", None)
    st.session_state.pop("username", None)


def current_user_id() -> Optional[int]:
    return st.session_state.get("user_id")


def current_username() -> Optional[str]:
    return st.session_state.get("username")


def require_login() -> int:
    """Stop the page with a warning if nobody is logged in.
    Returns the user id when someone is logged in."""
    user_id = current_user_id()
    if not user_id:
        st.warning("Please log in first.")
        st.stop()
    return user_id