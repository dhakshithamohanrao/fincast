"""All database queries. Every query on user data filters by user_id."""
import sqlite3
from contextlib import closing
from typing import Optional

from app.database.db import get_connection


def create_user(username: str, email: str, password_hash: str) -> Optional[int]:
    """Insert a user plus an empty profile. Returns the new id, or None if
    the username or email already exists."""
    with closing(get_connection()) as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash),
            )
            user_id = cur.lastrowid
            conn.execute("INSERT INTO profiles (user_id) VALUES (?)", (user_id,))
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            conn.rollback()
            return None


def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    with closing(get_connection()) as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with closing(get_connection()) as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()