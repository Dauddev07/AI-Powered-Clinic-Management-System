import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User
from app.services.appointments import auto_complete_past_appointments


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


def test_appointment_stays_confirmed_mid_slot_after_start_but_before_end(db, clinic):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=15)  # 17:00 if now is 17:15
    end = now + timedelta(minutes=15)  # 17:30
    appointment = _appointment(db, clinic, start, end)

    auto_complete_past_appointments(db, clinic.id)
    db.flush()
    db.refresh(appointment)

    assert appointment.status == "confirmed"


def test_appointment_completes_once_slot_end_has_passed(db, clinic):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=45)
    end = now - timedelta(minutes=15)  # already ended
    appointment = _appointment(db, clinic, start, end)

    auto_complete_past_appointments(db, clinic.id)
    db.flush()
    db.refresh(appointment)

    assert appointment.status == "completed"


def test_cancelled_appointment_is_never_overridden_even_after_slot_ends(db, clinic):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=2)
    end = now - timedelta(hours=1)
    appointment = _appointment(db, clinic, start, end, status="cancelled")

    auto_complete_past_appointments(db, clinic.id)
    db.flush()
    db.refresh(appointment)

    assert appointment.status == "cancelled"
