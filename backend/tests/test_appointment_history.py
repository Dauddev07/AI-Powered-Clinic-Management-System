import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.api.appointments import list_my_appointment_history
from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


@pytest.fixture
def patient(db, clinic):
    p = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(p)
    db.flush()
    return p


@pytest.fixture
def other_patient(db, clinic):
    p = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Other Patient",
    )
    db.add(p)
    db.flush()
    return p


def _appointment(db, clinic, patient, start_utc, status):
    dept = Department(clinic_id=clinic.id, name=f"Cardiology-{uuid.uuid4().hex[:8]}")
    db.add(dept)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id, department_id=dept.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
        full_name="Dr. Jane Example", is_active=True,
    )
    db.add(doctor)
    db.flush()
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start_utc, end_utc=start_utc + timedelta(minutes=30))
    db.add(slot)
    db.flush()
    appointment = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status=status,
    )
    db.add(appointment)
    db.flush()
    return appointment


def test_history_excludes_confirmed_upcoming_appointments(db, clinic, patient):
    now = datetime.now(timezone.utc)
    _appointment(db, clinic, patient, now + timedelta(days=1), status="confirmed")
    completed = _appointment(db, clinic, patient, now - timedelta(days=1), status="completed")

    result = list_my_appointment_history(current_user=patient, db=db)

    ids = [a.id for a in result.appointments]
    assert ids == [completed.id]


def test_history_includes_cancelled_no_show_and_expired(db, clinic, patient):
    now = datetime.now(timezone.utc)
    cancelled = _appointment(db, clinic, patient, now - timedelta(days=1), status="cancelled")
    no_show = _appointment(db, clinic, patient, now - timedelta(days=2), status="no_show")
    expired = _appointment(db, clinic, patient, now - timedelta(days=3), status="expired")

    result = list_my_appointment_history(current_user=patient, db=db)

    ids = {a.id for a in result.appointments}
    assert ids == {cancelled.id, no_show.id, expired.id}


def test_history_is_newest_first(db, clinic, patient):
    now = datetime.now(timezone.utc)
    older = _appointment(db, clinic, patient, now - timedelta(days=5), status="completed")
    newer = _appointment(db, clinic, patient, now - timedelta(days=1), status="completed")

    result = list_my_appointment_history(current_user=patient, db=db)

    assert [a.id for a in result.appointments] == [newer.id, older.id]


def test_history_never_returns_another_patients_appointments(db, clinic, patient, other_patient):
    now = datetime.now(timezone.utc)
    _appointment(db, clinic, other_patient, now - timedelta(days=1), status="completed")

    result = list_my_appointment_history(current_user=patient, db=db)

    assert result.appointments == []
