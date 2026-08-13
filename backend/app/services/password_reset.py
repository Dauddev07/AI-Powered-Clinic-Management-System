"""Forgot-password OTP flow: request_password_reset emails a 6-digit code, apply_password_reset
verifies it and updates the password. Both are written to never reveal whether a given
email has an account (same response either way) — the only signal an attacker gets is
"invalid or expired code", identical for "no such user" and "wrong code".
"""
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password, pwd_context
from app.models.password_reset_otp import PasswordResetOtp
from app.models.user import User
from app.services.email import send_password_reset_otp_email


class OtpCooldownActive(Exception):
    """A code was already requested for this user too recently — caller should show a
    "please wait" message. Never raised for an unknown email (see request_password_reset)."""


class InvalidOrExpiredOtp(Exception):
    pass


def request_password_reset(db: Session, email: str) -> tuple[User, str] | None:
    """Returns (user, plaintext_otp) to email, or None if there's no active account
    with this email — callers must respond identically to the patient either way, only
    branching on the return value to decide whether to actually send the email.
    """
    user = db.execute(
        select(User).where(User.email == email.lower(), User.is_active.is_(True))
    ).scalars().first()
    if user is None:
        return None

    cooldown_start = datetime.now(timezone.utc) - timedelta(
        seconds=settings.PASSWORD_RESET_OTP_RESEND_COOLDOWN_SECONDS
    )
    recent = db.execute(
        select(PasswordResetOtp)
        .where(PasswordResetOtp.user_id == user.id, PasswordResetOtp.created_at > cooldown_start)
        .order_by(PasswordResetOtp.created_at.desc())
    ).scalars().first()
    if recent is not None:
        raise OtpCooldownActive()

    code = f"{secrets.randbelow(1_000_000):06d}"
    otp = PasswordResetOtp(
        user_id=user.id,
        otp_hash=pwd_context.hash(code),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=settings.PASSWORD_RESET_OTP_TTL_MINUTES),
    )
    db.add(otp)
    db.commit()

    return user, code


def apply_password_reset(db: Session, email: str, code: str, new_password: str) -> None:
    user = db.execute(
        select(User).where(User.email == email.lower(), User.is_active.is_(True))
    ).scalars().first()
    if user is None:
        raise InvalidOrExpiredOtp()

    otp = db.execute(
        select(PasswordResetOtp)
        .where(PasswordResetOtp.user_id == user.id)
        .order_by(PasswordResetOtp.created_at.desc())
    ).scalars().first()

    if (
        otp is None
        or otp.used
        or otp.attempts >= settings.PASSWORD_RESET_OTP_MAX_ATTEMPTS
        or otp.expires_at < datetime.now(timezone.utc)
    ):
        raise InvalidOrExpiredOtp()

    if not pwd_context.verify(code, otp.otp_hash):
        otp.attempts += 1
        db.commit()
        raise InvalidOrExpiredOtp()

    user.hashed_password = hash_password(new_password)
    # Same invalidation as change-password: any token issued before now is rejected
    # by get_current_user, forcing re-login on every device.
    user.password_changed_at = datetime.now(timezone.utc).replace(microsecond=0)
    otp.used = True
    db.commit()


def send_otp_email(user: User, code: str) -> None:
    send_password_reset_otp_email(
        to=user.email,
        full_name=user.full_name,
        otp_code=code,
        ttl_minutes=settings.PASSWORD_RESET_OTP_TTL_MINUTES,
    )
