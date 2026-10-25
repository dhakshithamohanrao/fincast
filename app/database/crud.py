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

def save_forecast(user_id: int, target_month, table: pd.DataFrame, model_version: str) -> int:
    """Store one forecast row per series for a target month ('YYYY-MM').

    Existing rows are kept (one forecast per user, month and series), so a past
    forecast is never overwritten and can later be compared with the real value.
    Returns how many rows were added.
    """
    month = pd.Timestamp(target_month).strftime("%Y-%m")
    added = 0
    with closing(get_connection()) as conn:
        for row in table.itertuples(index=False):
            cur = conn.execute(
                """INSERT OR IGNORE INTO forecasts
                   (user_id, target_month, metric, predicted, lower, upper, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, month, str(row.series), float(row.forecast),
                 float(row.lower), float(row.upper), model_version),
            )
            added += cur.rowcount
        conn.commit()
    return added

def save_anomaly_review(user_id: int, transaction_id: int, status: str) -> bool:
    """Mark one of the user's transactions as 'confirmed' (a real problem) or
    'dismissed' (fine). Returns False if the transaction is not this user's."""
    if status not in ("confirmed", "dismissed"):
        raise ValueError("status must be 'confirmed' or 'dismissed'")
    with closing(get_connection()) as conn:
        cur = conn.execute(
            """INSERT INTO anomaly_reviews (user_id, transaction_id, status)
               SELECT ?, id, ? FROM transactions WHERE id = ? AND user_id = ?
               ON CONFLICT(user_id, transaction_id)
               DO UPDATE SET status = excluded.status, reviewed_at = CURRENT_TIMESTAMP""",
            (user_id, status, int(transaction_id), user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def get_anomaly_reviews(user_id: int) -> dict:
    """{transaction_id: 'confirmed' | 'dismissed'} for one user."""
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT transaction_id, status FROM anomaly_reviews WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return {row["transaction_id"]: row["status"] for row in rows}

def create_goal(user_id: int, name: str, target_amount: float, target_date: str) -> int:
    with closing(get_connection()) as conn:
        cur = conn.execute(
            "INSERT INTO goals (user_id, name, target_amount, target_date) VALUES (?, ?, ?, ?)",
            (user_id, name.strip(), float(target_amount), str(target_date)[:10]),
        )
        conn.commit()
        return cur.lastrowid


def get_goals(user_id: int) -> pd.DataFrame:
    with closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT id, name, target_amount, target_date FROM goals "
            "WHERE user_id = ? ORDER BY target_date, id",
            (user_id,),
        ).fetchall()
    df = pd.DataFrame([tuple(r) for r in rows],
                      columns=["id", "name", "target_amount", "target_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def delete_goal(user_id: int, goal_id: int) -> bool:
    with closing(get_connection()) as conn:
        cur = conn.execute("DELETE FROM goals WHERE id = ? AND user_id = ?",
                           (int(goal_id), user_id))
        conn.commit()
        return cur.rowcount > 0


def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    with closing(get_connection()) as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_profile(user_id: int) -> dict:
    """Profile settings, with defaults when the user has no profile row yet."""
    with closing(get_connection()) as conn:
        row = conn.execute("SELECT income_type, currency FROM profiles WHERE user_id = ?",
                           (user_id,)).fetchone()
    if row is None:
        return {"income_type": "salary", "currency": "EUR"}
    return {"income_type": row["income_type"], "currency": row["currency"]}


def save_profile(user_id: int, income_type: str, currency: str) -> None:
    with closing(get_connection()) as conn:
        conn.execute(
            "INSERT INTO profiles (user_id, income_type, currency) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET income_type = excluded.income_type, "
            "currency = excluded.currency",
            (user_id, income_type, currency),
        )
        conn.commit()


def update_password_hash(user_id: int, password_hash: str) -> bool:
    with closing(get_connection()) as conn:
        cur = conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                           (password_hash, user_id))
        conn.commit()
        return cur.rowcount > 0


def delete_user_data(user_id: int) -> bool:
    """Delete the user and, through ON DELETE CASCADE, all of their data."""
    with closing(get_connection()) as conn:
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return cur.rowcount > 0
