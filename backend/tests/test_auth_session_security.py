"""Integration coverage (real HTTP, via TestClient) for the two session-security
additions: refresh-token issuance/rotation/revocation, and per-IP rate limiting on
the login endpoint. Everything else in tests/test_auth_api.py calls the route
functions directly as plain Python, which is enough for their logic but can't
exercise slowapi (it needs a real starlette.Request) or a full login -> refresh ->
logout round trip through the actual dependency-injected session.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.db import get_db
from app.core.rate_limit import limiter
from app.core.security import hash_password
from app.main import app
from app.models.clinic import Clinic
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.services.refresh_tokens import InvalidRefreshToken, issue_refresh_token, revoke_refresh_token, rotate_refresh_token


def _clinic(db, name="Quickcheck Clinic"):
    c = Clinic(name=name, slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _verified_user(db, clinic, password="CorrectPass123"):
    u = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password=hash_password(password), full_name="Test User", email_verified=True,
    )
    db.add(u)
    db.flush()
    db.refresh(u)
    return u


def _client(db):
    """Routes the app's `get_db` dependency to this test's own transactional
    session (see conftest.py's `db` fixture) so anything committed during a
    TestClient request rolls back with the rest of the test, same as every other
    test in this suite — rather than hitting SessionLocal's real, separate
    connection and leaving rows behind.
    """
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


# --- refresh tokens ---


def test_login_response_includes_a_usable_refresh_token(db):
    clinic = _clinic(db)
    user = _verified_user(db, clinic)
    client = _client(db)
    try:
        resp = client.post("/auth/login", json={"email": user.email, "password": "CorrectPass123"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["refresh_token"]

        refreshed = client.post("/auth/refresh", json={"refresh_token": body["refresh_token"]})
        assert refreshed.status_code == 200
        assert refreshed.json()["access_token"]
        assert refreshed.json()["refresh_token"] != body["refresh_token"]
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_refresh_rejects_an_already_rotated_token(db):
    clinic = _clinic(db)
    user = _verified_user(db, clinic)
    plaintext = issue_refresh_token(db, user)

    rotate_refresh_token(db, plaintext)

    with pytest.raises(InvalidRefreshToken):
        rotate_refresh_token(db, plaintext)


def test_refresh_rejects_an_unknown_token(db):
    with pytest.raises(InvalidRefreshToken):
        rotate_refresh_token(db, "not-a-real-token")


def test_logout_revokes_the_token_so_it_can_no_longer_be_refreshed(db):
    clinic = _clinic(db)
    user = _verified_user(db, clinic)
    plaintext = issue_refresh_token(db, user)

    revoke_refresh_token(db, plaintext)

    with pytest.raises(InvalidRefreshToken):
        rotate_refresh_token(db, plaintext)


def test_logout_endpoint_revokes_only_the_one_session(db):
    clinic = _clinic(db)
    user = _verified_user(db, clinic)
    client = _client(db)
    try:
        login_resp = client.post("/auth/login", json={"email": user.email, "password": "CorrectPass123"})
        session_a = login_resp.json()["refresh_token"]
        session_b = issue_refresh_token(db, user)  # a second "device"

        logout_resp = client.post("/auth/logout", json={"refresh_token": session_a})
        assert logout_resp.status_code == 200

        # session_a is dead...
        dead = client.post("/auth/refresh", json={"refresh_token": session_a})
        assert dead.status_code == 401

        # ...but session_b (a different device) is untouched.
        alive = client.post("/auth/refresh", json={"refresh_token": session_b})
        assert alive.status_code == 200
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_expired_refresh_token_is_rejected(db):
    clinic = _clinic(db)
    user = _verified_user(db, clinic)
    plaintext = issue_refresh_token(db, user)
    row = db.query(RefreshToken).filter(RefreshToken.user_id == user.id).one()
    row.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    db.commit()

    with pytest.raises(InvalidRefreshToken):
        rotate_refresh_token(db, plaintext)


# --- rate limiting ---


def test_login_endpoint_is_rate_limited_per_ip(db, monkeypatch):
    """Rate limiting is disabled process-wide under pytest (see
    app/core/rate_limit.py) so the ~1100 other tests that call route functions
    directly don't need a real Request. Re-enabled just for this one test, which
    goes through TestClient and therefore has one.
    """
    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()

    clinic = _clinic(db)
    user = _verified_user(db, clinic)
    client = _client(db)
    try:
        # login is capped at 10/minute — the first 10 (even though the password is
        # wrong, and so 401) should go through untouched by the limiter.
        for _ in range(10):
            resp = client.post("/auth/login", json={"email": user.email, "password": "wrong-password"})
            assert resp.status_code == 401

        blocked = client.post("/auth/login", json={"email": user.email, "password": "wrong-password"})
        assert blocked.status_code == 429
    finally:
        app.dependency_overrides.pop(get_db, None)
        limiter.reset()
