"""All database queries. Every query on user data filters by user_id."""
import hashlib
import sqlite3
from contextlib import closing
from typing import Optional, Tuple

import pandas as pd

from app.database.db import get_connection


# ---------- users ----------

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


# ---------- transactions ----------

TRANSACTION_COLUMNS = ["id", "date", "description", "amount", "category", "is_income", "source"]


def make_dedupe_hash(date: str, description: str, amount: float) -> str:
    """Fingerprint of date + description + amount, used to detect duplicate uploads."""
    text = f"{date}|{description.strip().lower()}|{amount:.2f}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add_transactions(user_id: int, df: pd.DataFrame, source: str = "upload") -> Tuple[int, int]:
    """Insert cleaned transactions for one user and skip rows already stored.
    Returns (added, skipped)."""
    if df is None or df.empty:
        return 0, 0
    added = 0
    with closing(get_connection()) as conn:
        for row in df.itertuples(index=False):
            date = pd.Timestamp(row.date).strftime("%Y-%m-%d")
            description = str(row.description)
            amount = float(row.amount)
            category = getattr(row, "category", None)
            if category is not None and pd.isna(category):
                category = None
            cur = conn.execute(
                """INSERT OR IGNORE INTO transactions
                   (user_id, date, description, amount, category, is_income, source, dedupe_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id, date, description, amount, category,
                    int(amount > 0), source,
                    make_dedupe_hash(date, description, amount),
                ),
            )
            added += cur.rowcount
        conn.commit()
    return added, len(df) - added


def get_transactions(
    user_id: int, start: Optional[str] = None, end: Optional[str] = None
) -> pd.DataFrame:
    """All transactions of one user, optionally between two dates (inclusive)."""
    query = f"SELECT {', '.join(TRANSACTION_COLUMNS)} FROM transactions WHERE user_id = ?"
    params = [user_id]
    if start:
        query += " AND date >= ?"
        params.append(str(start)[:10])
    if end:
        query += " AND date <= ?"
        params.append(str(end)[:10])
    query += " ORDER BY date, id"
    with closing(get_connection()) as conn:
        rows = conn.execute(query, params).fetchall()
    df = pd.DataFrame([tuple(r) for r in rows], columns=TRANSACTION_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df

def save_corrections(user_id: int, corrections: pd.DataFrame) -> int:
    """Store (description, category) pairs the user fixed, for retraining later.
    Returns how many were saved."""
    if corrections is None or corrections.empty:
        return 0
    rows = [(user_id, d, c) for d, c in zip(corrections["description"], corrections["category"])]
    conn = get_connection()
    try:
        with conn:
            conn.executemany(
                "INSERT INTO category_corrections (user_id, description, category) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(user_id, description) DO UPDATE SET category = excluded.category",
                rows,
            )
    finally:
        conn.close()
    return len(rows)