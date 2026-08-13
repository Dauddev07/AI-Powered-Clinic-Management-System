"""Registration email-verification OTP flow — same shape as
app/services/password_reset.py (generate/hash/email a 6-digit code, check it without
leaking which failure case it was), kept as its own module/table rather than sharing
code with password reset since the two represent different consequences if
compromised and shouldn't share a lifecycle.
"""
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import pwd_context
from app.models.email_verification_otp import EmailVerificationOtp
from app.models.user import User
from app.services.email import send_email_verification_otp_email


class OtpCooldownActive(Exception):
    """A code was already requested for this user too recently."""


class InvalidOrExpiredOtp(Exception):
    pass


def _generate_and_store_otp(db: Session, user: User) -> str:
    cooldown_start = datetime.now(timezone.utc) - timedelta(
        seconds=settings.EMAIL_VERIFICATION_OTP_RESEND_COOLDOWN_SECONDS
    )
    recent = db.execute(
        select(EmailVerificationOtp)
        .where(EmailVerificationOtp.user_id == user.id, EmailVerificationOtp.created_at > cooldown_start)
        .order_by(EmailVerificationOtp.created_at.desc())
    ).scalars().first()
    if recent is not None:
        raise OtpCooldownActive()

    code = f"{secrets.randbelow(1_000_000):06d}"
    otp = EmailVerificationOtp(
        user_id=user.id,
        otp_hash=pwd_context.hash(code),
        expires_at=datetime.now(timezone.utc)
        + timedelta(minutes=settings.EMAIL_VERIFICATION_OTP_TTL_MINUTES),
    )
    db.add(otp)
    db.commit()
    return code


def send_verification_for_new_registration(db: Session, user: User) -> str:
    """Called once, right after a fresh registration commits — no cooldown is ever
    realistically hit here (there's no prior OTP row for a brand new user), but goes
    through the same shared path as a resend for consistency.
    """
    return _generate_and_store_otp(db, user)


def request_email_verification(db: Session, email: str) -> tuple[User, str] | None:
    """Backs the "resend verification code" action. Returns None (no email actually
    sent) both when the email has no account AND when the account is already
    verified — callers must respond identically for both cases, same
    no-enumeration principle as request_password_reset.
    """
    user = db.execute(
        select(User).where(User.email == email.lower(), User.is_active.is_(True))
    ).scalars().first()
    if user is None or user.email_verified:
        return None

    code = _generate_and_store_otp(db, user)
    return user, code


def verify_email(db: Session, email: str, code: str) -> User:
    user = db.execute(
        select(User).where(User.email == email.lower(), User.is_active.is_(True))
    ).scalars().first()
    if user is None:
        raise InvalidOrExpiredOtp()

    otp = db.execute(
        select(EmailVerificationOtp)
        .where(EmailVerificationOtp.user_id == user.id)
        .order_by(EmailVerificationOtp.created_at.desc())
    ).scalars().first()

    if (
        otp is None
        or otp.used
        or otp.attempts >= settings.EMAIL_VERIFICATION_OTP_MAX_ATTEMPTS
        or otp.expires_at < datetime.now(timezone.utc)
    ):
        raise InvalidOrExpiredOtp()

    if not pwd_context.verify(code, otp.otp_hash):
        otp.attempts += 1
        db.commit()
        raise InvalidOrExpiredOtp()

    user.email_verified = True
    otp.used = True
    db.commit()
    return user


def send_verification_otp_email(user: User, code: str) -> None:
    send_email_verification_otp_email(
        to=user.email,
        full_name=user.full_name,
        otp_code=code,
        ttl_minutes=settings.EMAIL_VERIFICATION_OTP_TTL_MINUTES,
    )
