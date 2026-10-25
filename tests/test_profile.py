import sqlite3
from contextlib import closing

import pytest

from app.auth import auth_service
from app.database import crud
from app.database.models import ALL_TABLES


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"

    def connect():
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    with closing(connect()) as conn:
        for statement in ALL_TABLES:
            conn.execute(statement)
        conn.commit()
    monkeypatch.setattr(crud, "get_connection", connect)


def _count(table, user_id):
    with closing(crud.get_connection()) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id = ?",
                            (user_id,)).fetchone()[0]


def test_change_password(db):
    ok, _ = auth_service.register("alice", "alice@example.com", "oldpassword1")
    assert ok
    user_id = auth_service.login("alice", "oldpassword1")["id"]

    assert not auth_service.change_password(user_id, "wrong-password", "newpassword1")[0]
    assert not auth_service.change_password(user_id, "oldpassword1", "short")[0]
    assert not auth_service.change_password(user_id, "oldpassword1", "oldpassword1")[0]
    assert auth_service.change_password(user_id, "oldpassword1", "newpassword1")[0]

    assert auth_service.login("alice", "newpassword1") is not None
    assert auth_service.login("alice", "oldpassword1") is None


def test_delete_removes_only_that_users_data(db):
    a = crud.create_user("alice", "a@example.com", "hash")
    b = crud.create_user("bob", "b@example.com", "hash")
    for uid in (a, b):
        crud.save_profile(uid, "salary", "EUR")
        crud.create_goal(uid, "Holiday", 1000, "2027-01-01")
        with closing(crud.get_connection()) as conn:
            conn.execute("INSERT INTO transactions (user_id, date, description, amount, "
                         "dedupe_hash) VALUES (?, '2026-01-01', 'x', -5, 'h')", (uid,))
            conn.commit()

    assert crud.delete_user_data(a)
    for table in ("profiles", "goals", "transactions"):
        assert _count(table, a) == 0
        assert _count(table, b) == 1
    assert crud.get_user_by_id(a) is None
    assert crud.get_user_by_id(b) is not None