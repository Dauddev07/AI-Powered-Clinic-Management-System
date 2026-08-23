import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.core.security import hash_password, verify_password
from app.models.account_delete_otp import AccountDeleteOtp
from app.models.appointment import Appointment
from app.models.appointment_feedback import AppointmentFeedback
from app.models.audit_log import AuditLog
from app.models.clinic import Clinic
from app.models.conversation_memory import ConversationMemory
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.notification import Notification
from app.models.refresh_token import RefreshToken
from app.models.slot import Slot
from app.models.user import User
from app.services.account_deletion import (
    InvalidOrExpiredOtp,
    OtpCooldownActive,
    confirm_and_delete_account,
    request_delete_otp,
)


def _clinic(db):
    c = Clinic(name="Quickcheck Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _user(db, clinic, email=None):
    u = User(
        clinic_id=clinic.id,
        role="patient",
        email=email or f"{uuid.uuid4().hex}@example.com",
        hashed_password=hash_password("SomePass123"),
        full_name="Test Patient",
    )
    db.add(u)
    db.flush()
    db.refresh(u)
    return u


def _confirmed_appointment(db, clinic, patient):
    dept = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(dept)
    db.flush()

    doctor = Doctor(
        clinic_id=clinic.id, department_id=dept.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
        full_name="Dr. Jane Example", is_active=True,
    )
    db.add(doctor)
    db.flush()

    start = datetime.now(timezone.utc) + timedelta(days=1)
    slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start, end_utc=start + timedelta(minutes=30),
        status="booked",
    )
    db.add(slot)
    db.flush()

    appointment = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="confirmed",
    )
    db.add(appointment)
    db.flush()
    return appointment, slot, doctor


def test_request_delete_otp_returns_code(db):
    clinic = _clinic(db)
    user = _user(db, clinic)

    code = request_delete_otp(db, user)

    assert len(code) == 6 and code.isdigit()
    row = db.query(AccountDeleteOtp).filter_by(user_id=user.id).one()
    assert row.used is False
    assert row.attempts == 0
    assert verify_password(code, row.otp_hash)


def test_request_delete_otp_enforces_cooldown(db):
    clinic = _clinic(db)
    user = _user(db, clinic)

    request_delete_otp(db, user)
    with pytest.raises(OtpCooldownActive):
        request_delete_otp(db, user)


def test_confirm_and_delete_account_rejects_wrong_code_and_counts_attempt(db):
    clinic = _clinic(db)
    user = _user(db, clinic)

    request_delete_otp(db, user)
    with pytest.raises(InvalidOrExpiredOtp):
        confirm_and_delete_account(db, user, "000000")

    row = db.query(AccountDeleteOtp).filter_by(user_id=user.id).one()
    assert row.attempts == 1

    # The account itself must still exist after a failed attempt.
    assert db.get(User, user.id) is not None


def test_confirm_and_delete_account_rejects_expired_code(db):
    clinic = _clinic(db)
    user = _user(db, clinic)

    code = request_delete_otp(db, user)
    row = db.query(AccountDeleteOtp).filter_by(user_id=user.id).one()
    row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()

    with pytest.raises(InvalidOrExpiredOtp):
        confirm_and_delete_account(db, user, code)
    assert db.get(User, user.id) is not None


def test_confirm_and_delete_account_locks_out_after_max_attempts(db):
    clinic = _clinic(db)
    user = _user(db, clinic)

    code = request_delete_otp(db, user)
    for _ in range(5):
        with pytest.raises(InvalidOrExpiredOtp):
            confirm_and_delete_account(db, user, "999999")

    with pytest.raises(InvalidOrExpiredOtp):
        confirm_and_delete_account(db, user, code)
    assert db.get(User, user.id) is not None


def test_confirm_and_delete_account_removes_user_and_all_dependent_data(db):
    clinic = _clinic(db)
    user = _user(db, clinic)
    user_id = user.id
    appointment, slot, doctor = _confirmed_appointment(db, clinic, user)

    db.add(
        AppointmentFeedback(
            clinic_id=clinic.id, patient_id=user.id, appointment_id=appointment.id, doctor_id=doctor.id, rating=5,
        )
    )
    db.add(
        Notification(
            clinic_id=clinic.id, user_id=user.id, type="appointment_booked", message="hi",
            related_appointment_id=appointment.id,
        )
    )
    db.add(RefreshToken(user_id=user.id, token_hash="x" * 64, expires_at=datetime.now(timezone.utc) + timedelta(days=1)))
    db.add(ConversationMemory(clinic_id=clinic.id, user_id=user.id, session_id=uuid.uuid4(), role="user", content="hi"))
    audit_row = AuditLog(clinic_id=clinic.id, actor_user_id=user.id, action="test_action", entity_type="test")
    db.add(audit_row)
    db.flush()
    audit_row_id = audit_row.id

    code = request_delete_otp(db, user)
    confirm_and_delete_account(db, user, code)

    assert db.get(User, user_id) is None
    assert db.query(Appointment).filter_by(patient_id=user_id).count() == 0
    assert db.query(AppointmentFeedback).filter_by(patient_id=user_id).count() == 0
    assert db.query(Notification).filter_by(user_id=user_id).count() == 0
    assert db.query(RefreshToken).filter_by(user_id=user_id).count() == 0
    assert db.query(ConversationMemory).filter_by(user_id=user_id).count() == 0
    assert db.query(AccountDeleteOtp).filter_by(user_id=user_id).count() == 0

    db.refresh(slot)
    assert slot.status == "open"

    # Audit log row survives, anonymized rather than deleted.
    db.refresh(audit_row)
    assert audit_row.id == audit_row_id
    assert audit_row.actor_user_id is None
