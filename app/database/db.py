"""SQLite connection and database setup."""
import sqlite3
from contextlib import closing

from app.database.models import ALL_TABLES
from src.utils.helpers import get_project_root, load_config


def get_db_path():
    """Database file path from config.yaml (creates the folder if needed)."""
    path = get_project_root() / load_config()["paths"]["database"]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection() -> sqlite3.Connection:
    """Open a connection with foreign keys switched on."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create all tables if they don't exist yet."""
    with closing(get_connection()) as conn:
        for statement in ALL_TABLES:
            conn.execute(statement)
        conn.commit()