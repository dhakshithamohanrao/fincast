"""Registration and login logic (passwords are only ever stored as bcrypt hashes)."""
import re
import sqlite3
from typing import Optional, Tuple

import bcrypt

from app.database import crud

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_USERNAME_LENGTH = 3
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72  # bcrypt cannot handle longer passwords


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def register(username: str, email: str, password: str) -> Tuple[bool, str]:
    """Validate input and create an account. Returns (success, message)."""
    username = username.strip()
    email = email.strip().lower()

    if len(username) < MIN_USERNAME_LENGTH:
        return False, f"Username must be at least {MIN_USERNAME_LENGTH} characters."
    if not EMAIL_PATTERN.match(email):
        return False, "Please enter a valid email address."
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return False, "Password is too long (maximum 72 bytes)."
    if crud.get_user_by_username(username):
        return False, "Username already taken."
    if crud.get_user_by_email(email):
        return False, "Email already registered."

    user_id = crud.create_user(username, email, hash_password(password))
    if user_id is None:
        return False, "Could not create the account. Please try again."
    return True, "Account created. You can log in now."


def login(username: str, password: str) -> Optional[sqlite3.Row]:
    """Return the user row if the credentials are correct, otherwise None."""
    user = crud.get_user_by_username(username.strip())
    if user is None:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user