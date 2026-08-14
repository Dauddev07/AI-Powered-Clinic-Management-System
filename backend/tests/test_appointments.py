import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User
from app.services import booking_engine
from app.services.appointments import get_pending_visit_confirmations


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _appointment(db, clinic, start_utc, end_utc, status="confirmed"):
    dept = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(dept)
    db.flush()

    doctor = Doctor(
        clinic_id=clinic.id, department_id=dept.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
        full_name="Dr. Jane Example", is_active=True,
    )
    db.add(doctor)
    db.flush()

    patient = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(patient)
    db.flush()

    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start_utc, end_utc=end_utc)
    db.add(slot)
    db.flush()

    appointment = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status=status
    )
    db.add(appointment)
    db.flush()
    return appointment


def test_appointment_not_pending_mid_slot_after_start_but_before_end(db, clinic):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=15)  # 17:00 if now is 17:15
    end = now + timedelta(minutes=15)  # 17:30
    appointment = _appointment(db, clinic, start, end)

    pending = get_pending_visit_confirmations(db, clinic.id, appointment.patient_id)

    assert pending == []


def test_appointment_is_pending_once_slot_end_has_passed(db, clinic):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=45)
    end = now - timedelta(minutes=15)  # already ended
    appointment = _appointment(db, clinic, start, end)

    pending = get_pending_visit_confirmations(db, clinic.id, appointment.patient_id)

    assert [a.id for a in pending] == [appointment.id]
    assert appointment.status == "confirmed"  # still confirmed — nothing auto-flips it


def test_cancelled_appointment_never_shows_up_as_pending(db, clinic):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=2)
    end = now - timedelta(hours=1)
    appointment = _appointment(db, clinic, start, end, status="cancelled")

    pending = get_pending_visit_confirmations(db, clinic.id, appointment.patient_id)

    assert pending == []


def test_confirm_visit_completed_marks_appointment_completed(db, clinic):
    now = datetime.now(timezone.utc)
    appointment = _appointment(db, clinic, now - timedelta(hours=2), now - timedelta(hours=1))

    result = booking_engine.confirm_visit(
        db, clinic_id=clinic.id, patient_id=appointment.patient_id, appointment_id=appointment.id, completed=True
    )

    assert result.status == "completed"


def test_confirm_visit_missed_marks_appointment_no_show(db, clinic):
    now = datetime.now(timezone.utc)
    appointment = _appointment(db, clinic, now - timedelta(hours=2), now - timedelta(hours=1))

    result = booking_engine.confirm_visit(
        db, clinic_id=clinic.id, patient_id=appointment.patient_id, appointment_id=appointment.id, completed=False
    )

    assert result.status == "no_show"


def test_confirm_visit_rejects_appointment_whose_slot_has_not_ended(db, clinic):
    now = datetime.now(timezone.utc)
    appointment = _appointment(db, clinic, now - timedelta(minutes=5), now + timedelta(minutes=25))

    with pytest.raises(HTTPException) as exc_info:
        booking_engine.confirm_visit(
            db, clinic_id=clinic.id, patient_id=appointment.patient_id, appointment_id=appointment.id, completed=True
        )

    assert exc_info.value.status_code == 409
    assert appointment.status == "confirmed"


def test_confirm_visit_rejects_appointment_not_in_confirmed_status(db, clinic):
    now = datetime.now(timezone.utc)
    appointment = _appointment(db, clinic, now - timedelta(hours=2), now - timedelta(hours=1), status="cancelled")

    with pytest.raises(HTTPException) as exc_info:
        booking_engine.confirm_visit(
            db, clinic_id=clinic.id, patient_id=appointment.patient_id, appointment_id=appointment.id, completed=True
        )

    assert exc_info.value.status_code == 409
