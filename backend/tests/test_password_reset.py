import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.core.security import hash_password, verify_password
from app.models.clinic import Clinic
from app.models.password_reset_otp import PasswordResetOtp
from app.models.user import User
from app.services.password_reset import (
    InvalidOrExpiredOtp,
    OtpCooldownActive,
    apply_password_reset,
    request_password_reset,
    verify_otp,
)


def _clinic(db):
    c = Clinic(name="Quickcheck Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _user(db, clinic, email=None, password="OldPass123"):
    u = User(
        clinic_id=clinic.id,
        role="patient",
        email=email or f"{uuid.uuid4().hex}@example.com",
        hashed_password=hash_password(password),
        full_name="Test Patient",
    )
    db.add(u)
    db.flush()
    db.refresh(u)
    return u


def test_request_password_reset_returns_user_and_code_for_known_email(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="known@example.com")

    result = request_password_reset(db, "known@example.com")

    assert result is not None
    returned_user, code = result
    assert returned_user.id == user.id
    assert len(code) == 6 and code.isdigit()

    row = db.query(PasswordResetOtp).filter_by(user_id=user.id).one()
    assert row.used is False
    assert row.attempts == 0
    assert verify_password(code, row.otp_hash)


def test_request_password_reset_returns_none_for_unknown_email_without_leaking(db):
    result = request_password_reset(db, "nobody@example.com")
    assert result is None


def test_request_password_reset_enforces_cooldown(db):
    clinic = _clinic(db)
    _user(db, clinic, email="cooldown@example.com")

    request_password_reset(db, "cooldown@example.com")
    with pytest.raises(OtpCooldownActive):
        request_password_reset(db, "cooldown@example.com")


def test_apply_password_reset_succeeds_with_correct_code(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="reset@example.com", password="OldPass123")

    _, code = request_password_reset(db, "reset@example.com")
    apply_password_reset(db, "reset@example.com", code, "NewPass456")

    db.refresh(user)
    assert verify_password("NewPass456", user.hashed_password)
    assert user.password_changed_at is not None

    row = db.query(PasswordResetOtp).filter_by(user_id=user.id).one()
    assert row.used is True


def test_apply_password_reset_rejects_wrong_code_and_counts_attempt(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="wrongcode@example.com")

    request_password_reset(db, "wrongcode@example.com")
    with pytest.raises(InvalidOrExpiredOtp):
        apply_password_reset(db, "wrongcode@example.com", "000000", "NewPass456")

    row = db.query(PasswordResetOtp).filter_by(user_id=user.id).one()
    assert row.attempts == 1
    assert row.used is False


def test_apply_password_reset_rejects_expired_code(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="expired@example.com")

    _, code = request_password_reset(db, "expired@example.com")
    row = db.query(PasswordResetOtp).filter_by(user_id=user.id).one()
    row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()

    with pytest.raises(InvalidOrExpiredOtp):
        apply_password_reset(db, "expired@example.com", code, "NewPass456")


def test_apply_password_reset_rejects_reused_code(db):
    clinic = _clinic(db)
    _user(db, clinic, email="reused@example.com")

    _, code = request_password_reset(db, "reused@example.com")
    apply_password_reset(db, "reused@example.com", code, "NewPass456")

    with pytest.raises(InvalidOrExpiredOtp):
        apply_password_reset(db, "reused@example.com", code, "AnotherPass789")


def test_apply_password_reset_locks_out_after_max_attempts(db):
    clinic = _clinic(db)
    _user(db, clinic, email="lockout@example.com")

    _, code = request_password_reset(db, "lockout@example.com")
    for _ in range(5):
        with pytest.raises(InvalidOrExpiredOtp):
            apply_password_reset(db, "lockout@example.com", "999999", "NewPass456")

    # Even the correct code is now rejected — attempts have hit the cap.
    with pytest.raises(InvalidOrExpiredOtp):
        apply_password_reset(db, "lockout@example.com", code, "NewPass456")


def test_apply_password_reset_rejects_unknown_email(db):
    with pytest.raises(InvalidOrExpiredOtp):
        apply_password_reset(db, "nobody@example.com", "123456", "NewPass456")


def test_verify_otp_succeeds_without_consuming_the_code(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="verifyonly@example.com")

    _, code = request_password_reset(db, "verifyonly@example.com")
    verify_otp(db, "verifyonly@example.com", code)

    # Not marked used, and still usable afterward for the real reset.
    row = db.query(PasswordResetOtp).filter_by(user_id=user.id).one()
    assert row.used is False

    apply_password_reset(db, "verifyonly@example.com", code, "NewPass456")
    db.refresh(row)
    assert row.used is True


def test_verify_otp_rejects_wrong_code(db):
    clinic = _clinic(db)
    _user(db, clinic, email="verifywrong@example.com")

    request_password_reset(db, "verifywrong@example.com")
    with pytest.raises(InvalidOrExpiredOtp):
        verify_otp(db, "verifywrong@example.com", "000000")
