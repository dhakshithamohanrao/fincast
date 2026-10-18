import pytest

from app.auth import auth_service
from app.database import db


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    """Give every test its own fresh, empty database file."""
    monkeypatch.setattr(db, "get_db_path", lambda: tmp_path / "test.db")
    db.init_db()


def test_register_and_login():
    ok, _ = auth_service.register("anna", "anna@example.com", "secret123")
    assert ok
    assert auth_service.login("anna", "secret123") is not None


def test_duplicate_username_rejected():
    auth_service.register("anna", "anna@example.com", "secret123")
    ok, message = auth_service.register("anna", "other@example.com", "secret123")
    assert not ok
    assert "taken" in message.lower()


def test_wrong_password_rejected():
    auth_service.register("anna", "anna@example.com", "secret123")
    assert auth_service.login("anna", "wrongpass") is None


def test_hash_is_not_plain_password():
    hashed = auth_service.hash_password("secret123")
    assert hashed != "secret123"
    assert hashed.startswith("$2")
    assert auth_service.verify_password("secret123", hashed)