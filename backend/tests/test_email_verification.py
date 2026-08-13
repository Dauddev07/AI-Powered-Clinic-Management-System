import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.core.security import hash_password, verify_password
from app.models.clinic import Clinic
from app.models.email_verification_otp import EmailVerificationOtp
from app.models.user import User
from app.services.email_verification import (
    InvalidOrExpiredOtp,
    OtpCooldownActive,
    request_email_verification,
    send_verification_for_new_registration,
    verify_email,
)


def _clinic(db):
    c = Clinic(name="Quickcheck Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _user(db, clinic, email=None, verified=False):
    u = User(
        clinic_id=clinic.id,
        role="patient",
        email=email or f"{uuid.uuid4().hex}@example.com",
        hashed_password=hash_password("SomePass123"),
        full_name="Test Patient",
        email_verified=verified,
    )
    db.add(u)
    db.flush()
    db.refresh(u)
    return u


def test_send_verification_for_new_registration_returns_a_code(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="fresh@example.com")

    code = send_verification_for_new_registration(db, user)

    assert len(code) == 6 and code.isdigit()
    row = db.query(EmailVerificationOtp).filter_by(user_id=user.id).one()
    assert row.used is False
    assert verify_password(code, row.otp_hash)


def test_verify_email_succeeds_and_marks_user_verified(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="verifyme@example.com")
    code = send_verification_for_new_registration(db, user)

    result = verify_email(db, "verifyme@example.com", code)

    assert result.email_verified is True
    db.refresh(user)
    assert user.email_verified is True
    row = db.query(EmailVerificationOtp).filter_by(user_id=user.id).one()
    assert row.used is True


def test_verify_email_rejects_wrong_code_and_counts_attempt(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="wrongcode@example.com")
    send_verification_for_new_registration(db, user)

    with pytest.raises(InvalidOrExpiredOtp):
        verify_email(db, "wrongcode@example.com", "000000")

    db.refresh(user)
    assert user.email_verified is False
    row = db.query(EmailVerificationOtp).filter_by(user_id=user.id).one()
    assert row.attempts == 1


def test_verify_email_rejects_expired_code(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="expired@example.com")
    code = send_verification_for_new_registration(db, user)
    row = db.query(EmailVerificationOtp).filter_by(user_id=user.id).one()
    row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()

    with pytest.raises(InvalidOrExpiredOtp):
        verify_email(db, "expired@example.com", code)


def test_verify_email_rejects_reused_code(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="reused@example.com")
    code = send_verification_for_new_registration(db, user)
    verify_email(db, "reused@example.com", code)

    with pytest.raises(InvalidOrExpiredOtp):
        verify_email(db, "reused@example.com", code)


def test_verify_email_locks_out_after_max_attempts(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="lockout@example.com")
    code = send_verification_for_new_registration(db, user)

    for _ in range(5):
        with pytest.raises(InvalidOrExpiredOtp):
            verify_email(db, "lockout@example.com", "999999")

    with pytest.raises(InvalidOrExpiredOtp):
        verify_email(db, "lockout@example.com", code)


def test_verify_email_rejects_unknown_email(db):
    with pytest.raises(InvalidOrExpiredOtp):
        verify_email(db, "nobody@example.com", "123456")


def test_request_email_verification_returns_none_for_unknown_email(db):
    assert request_email_verification(db, "nobody@example.com") is None


def test_request_email_verification_returns_none_for_already_verified_account(db):
    clinic = _clinic(db)
    _user(db, clinic, email="already@example.com", verified=True)

    assert request_email_verification(db, "already@example.com") is None


def test_request_email_verification_returns_user_and_code_for_unverified_account(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="pending@example.com", verified=False)

    result = request_email_verification(db, "pending@example.com")

    assert result is not None
    returned_user, code = result
    assert returned_user.id == user.id
    assert len(code) == 6


def test_request_email_verification_enforces_cooldown(db):
    clinic = _clinic(db)
    user = _user(db, clinic, email="cooldown@example.com", verified=False)
    send_verification_for_new_registration(db, user)

    with pytest.raises(OtpCooldownActive):
        request_email_verification(db, "cooldown@example.com")
