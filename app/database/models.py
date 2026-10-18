"""SQLite table definitions for FinCast."""

CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_PROFILES = """
CREATE TABLE IF NOT EXISTS profiles (
    user_id INTEGER PRIMARY KEY,
    income_type TEXT NOT NULL DEFAULT 'salary',
    currency TEXT NOT NULL DEFAULT 'EUR',
    savings_goal REAL,
    goal_date TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

CREATE_TRANSACTIONS = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    amount REAL NOT NULL,
    category TEXT,
    is_income INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'upload',
    dedupe_hash TEXT NOT NULL,
    UNIQUE (user_id, dedupe_hash),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

CREATE_MONTHLY_SUMMARY = """
CREATE TABLE IF NOT EXISTS monthly_summary (
    user_id INTEGER NOT NULL,
    month TEXT NOT NULL,
    income REAL NOT NULL DEFAULT 0,
    expenses REAL NOT NULL DEFAULT 0,
    savings REAL NOT NULL DEFAULT 0,
    balance_end REAL,
    PRIMARY KEY (user_id, month),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

CREATE_FORECASTS = """
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    target_month TEXT NOT NULL,
    metric TEXT NOT NULL,
    predicted REAL NOT NULL,
    lower REAL,
    upper REAL,
    model_version TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actual REAL,
    UNIQUE (user_id, target_month, metric),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

CREATE_GOALS = """
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    target_amount REAL NOT NULL,
    target_date TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

ALL_TABLES = [
    CREATE_USERS,
    CREATE_PROFILES,
    CREATE_TRANSACTIONS,
    CREATE_MONTHLY_SUMMARY,
    CREATE_FORECASTS,
    CREATE_GOALS,
]