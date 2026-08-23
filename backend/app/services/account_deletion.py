"""Patient self-service account deletion: request_delete_otp emails a 6-digit code to
the already-authenticated patient, confirm_and_delete_account verifies it and then
permanently deletes the account and every row tied to it. No FK on users.id has
ondelete=CASCADE anywhere in the schema, so every dependent table is cleared here,
in FK-safe order, inside one transaction, before the user row itself is deleted.
"""
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import pwd_context
from app.models.account_delete_otp import AccountDeleteOtp
from app.models.appointment import Appointment
from app.models.appointment_department_day_reschedule_use import AppointmentDepartmentDayRescheduleUse
from app.models.appointment_department_day_use import AppointmentDepartmentDayUse
from app.models.appointment_feedback import AppointmentFeedback
from app.models.audit_log import AuditLog
from app.models.conversation_memory import ConversationMemory
from app.models.email_verification_otp import EmailVerificationOtp
from app.models.ingestion_log import IngestionLog
from app.models.notification import Notification
from app.models.password_reset_otp import PasswordResetOtp
from app.models.patient_memory_profile import PatientMemoryProfile
from app.models.refresh_token import RefreshToken
from app.models.slot import Slot
from app.models.user import User
from app.services.email import send_account_delete_otp_email


class OtpCooldownActive(Exception):
    """A code was already requested for this user too recently — caller should show a
    "please wait" message."""


class InvalidOrExpiredOtp(Exception):
    pass


def request_delete_otp(db: Session, user: User) -> str:
    """Returns the plaintext OTP to email. Unlike password-reset, the caller is
    already authenticated as `user`, so there's no need to hide whether the account
    exists — only the resend cooldown needs enforcing here.
    """
    cooldown_start = datetime.now(timezone.utc) - timedelta(
        seconds=settings.ACCOUNT_DELETE_OTP_RESEND_COOLDOWN_SECONDS
    )
    recent = db.execute(
        select(AccountDeleteOtp)
        .where(AccountDeleteOtp.user_id == user.id, AccountDeleteOtp.created_at > cooldown_start)
        .order_by(AccountDeleteOtp.created_at.desc())
    ).scalars().first()
    if recent is not None:
        raise OtpCooldownActive()

    code = f"{secrets.randbelow(1_000_000):06d}"
    otp = AccountDeleteOtp(
        user_id=user.id,
        otp_hash=pwd_context.hash(code),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=settings.ACCOUNT_DELETE_OTP_TTL_MINUTES),
    )
    db.add(otp)
    db.commit()

    return code


def send_otp_email(user: User, code: str) -> None:
    send_account_delete_otp_email(
        to=user.email,
        full_name=user.full_name,
        otp_code=code,
        ttl_minutes=settings.ACCOUNT_DELETE_OTP_TTL_MINUTES,
    )


def _checked_otp(db: Session, user: User, code: str) -> AccountDeleteOtp:
    otp = db.execute(
        select(AccountDeleteOtp)
        .where(AccountDeleteOtp.user_id == user.id)
        .order_by(AccountDeleteOtp.created_at.desc())
    ).scalars().first()

    if (
        otp is None
        or otp.used
        or otp.attempts >= settings.ACCOUNT_DELETE_OTP_MAX_ATTEMPTS
        or otp.expires_at < datetime.now(timezone.utc)
    ):
        raise InvalidOrExpiredOtp()

    if not pwd_context.verify(code, otp.otp_hash):
        otp.attempts += 1
        db.commit()
        raise InvalidOrExpiredOtp()

    return otp


def confirm_and_delete_account(db: Session, user: User, code: str) -> None:
    otp = _checked_otp(db, user, code)
    otp.used = True

    # Release any still-active bookings' slots back to "open" before their
    # appointment rows are gone, so they don't stay stuck "booked" with nothing
    # booking them (same as booking_engine.cancel_appointment's own release).
    confirmed_appointments = db.execute(
        select(Appointment).where(Appointment.patient_id == user.id, Appointment.status == "confirmed")
    ).scalars().all()
    for appointment in confirmed_appointments:
        slot = db.get(Slot, appointment.slot_id)
        if slot is not None:
            slot.status = "open"

    # Delete every row that actually belongs to this patient, in FK-safe order.
    # Flushed after each model rather than once at the end: with no ORM
    # relationship() configured between these tables (only raw FK columns),
    # SQLAlchemy's unit-of-work can't infer the dependency graph itself and won't
    # necessarily honor this loop's ordering within a single flush — e.g. Appointment
    # could be issued before AppointmentFeedback and violate its FK.
    for model, column in (
        (AppointmentFeedback, AppointmentFeedback.patient_id),
        (AppointmentDepartmentDayUse, AppointmentDepartmentDayUse.patient_id),
        (AppointmentDepartmentDayRescheduleUse, AppointmentDepartmentDayRescheduleUse.patient_id),
        (Appointment, Appointment.patient_id),
        (Notification, Notification.user_id),
        (RefreshToken, RefreshToken.user_id),
        (EmailVerificationOtp, EmailVerificationOtp.user_id),
        (PasswordResetOtp, PasswordResetOtp.user_id),
        (AccountDeleteOtp, AccountDeleteOtp.user_id),
        (ConversationMemory, ConversationMemory.user_id),
        (PatientMemoryProfile, PatientMemoryProfile.user_id),
    ):
        for row in db.execute(select(model).where(column == user.id)).scalars().all():
            db.delete(row)
        db.flush()

    # audit_logs/ingestion_logs are clinic-side operational records, not the
    # patient's own data — their FK is nullable specifically so the log entry can
    # outlive its actor, so these are anonymized rather than deleted.
    for row in db.execute(select(AuditLog).where(AuditLog.actor_user_id == user.id)).scalars().all():
        row.actor_user_id = None
    for row in db.execute(select(IngestionLog).where(IngestionLog.uploaded_by_user_id == user.id)).scalars().all():
        row.uploaded_by_user_id = None

    db.delete(user)
    db.commit()
