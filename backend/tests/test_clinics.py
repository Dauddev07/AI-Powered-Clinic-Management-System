import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.api import clinics as clinics_module
from app.api.clinics import public_top_rated_doctors
from app.models.appointment import Appointment
from app.models.appointment_feedback import AppointmentFeedback
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User

# This endpoint is intentionally cross-clinic and unscoped (the landing page is
# reached before a visitor has picked a clinic), so it also picks up whatever
# real ratings already exist in the dev DB's seed data alongside anything a test
# creates. Tests here never assert an exact/empty result set for that reason —
# instead they monkeypatch the LIMIT way up so a test's own doctors are never
# pushed out of the slice by seed data, then look only at the rows for the
# doctor names the test itself created.


@pytest.fixture(autouse=True)
def _uncapped_limit(monkeypatch):
    monkeypatch.setattr(clinics_module, "TOP_RATED_DOCTORS_LIMIT", 10_000)


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


@pytest.fixture
def department(db, clinic):
    dept = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(dept)
    db.flush()
    return dept


def _doctor(db, clinic, department, name):
    d = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
        full_name=name, is_active=True,
    )
    db.add(d)
    db.flush()
    return d


def _completed_appointment(db, clinic, doctor):
    patient = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(patient)
    db.flush()

    start = datetime.now(timezone.utc) - timedelta(days=1)
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start, end_utc=start + timedelta(minutes=30))
    db.add(slot)
    db.flush()

    appointment = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="completed",
    )
    db.add(appointment)
    db.flush()
    return appointment


def _rate(db, clinic, doctor, rating):
    appointment = _completed_appointment(db, clinic, doctor)
    feedback = AppointmentFeedback(
        clinic_id=clinic.id, patient_id=appointment.patient_id, appointment_id=appointment.id,
        doctor_id=doctor.id, rating=rating,
    )
    db.add(feedback)
    db.flush()
    return appointment


def _by_name(result, name):
    return next(r for r in result if r.doctor_name == name)


def test_public_top_rated_doctors_orders_by_average_rating_descending(db, clinic, department):
    high = _doctor(db, clinic, department, f"Dr. High {uuid.uuid4().hex[:6]}")
    low = _doctor(db, clinic, department, f"Dr. Low {uuid.uuid4().hex[:6]}")
    _rate(db, clinic, high, 5)
    _rate(db, clinic, low, 3)

    result = public_top_rated_doctors(db=db)
    names = [r.doctor_name for r in result]

    assert names.index(high.full_name) < names.index(low.full_name)


def test_public_top_rated_doctors_limits_to_three(db, clinic, department):
    clinics_module.TOP_RATED_DOCTORS_LIMIT = 3
    for i in range(5):
        doctor = _doctor(db, clinic, department, f"Dr. Limit {i} {uuid.uuid4().hex[:6]}")
        _rate(db, clinic, doctor, 5)

    result = public_top_rated_doctors(db=db)

    assert len(result) == 3


def test_public_top_rated_doctors_excludes_inactive_doctors(db, clinic, department):
    inactive = _doctor(db, clinic, department, f"Dr. Inactive {uuid.uuid4().hex[:6]}")
    inactive.is_active = False
    db.flush()
    _rate(db, clinic, inactive, 5)

    result = public_top_rated_doctors(db=db)

    assert inactive.full_name not in [r.doctor_name for r in result]


def test_public_top_rated_doctors_spans_every_active_clinic(db, clinic, department):
    other_clinic = Clinic(name=f"Other Clinic {uuid.uuid4().hex[:6]}", slug=f"other-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(other_clinic)
    db.flush()
    other_department = Department(clinic_id=other_clinic.id, name="Neurology")
    db.add(other_department)
    db.flush()
    other_doctor = _doctor(db, other_clinic, other_department, f"Dr. Other Clinic {uuid.uuid4().hex[:6]}")
    _rate(db, other_clinic, other_doctor, 5)

    result = public_top_rated_doctors(db=db)

    row = _by_name(result, other_doctor.full_name)
    assert row.clinic_name == other_clinic.name


def test_public_top_rated_doctors_excludes_inactive_clinics(db, clinic, department):
    inactive_clinic = Clinic(
        name=f"Closed Clinic {uuid.uuid4().hex[:6]}", slug=f"closed-{uuid.uuid4().hex[:8]}", timezone="UTC", is_active=False,
    )
    db.add(inactive_clinic)
    db.flush()
    inactive_department = Department(clinic_id=inactive_clinic.id, name="Neurology")
    db.add(inactive_department)
    db.flush()
    doctor = _doctor(db, inactive_clinic, inactive_department, f"Dr. Closed Clinic {uuid.uuid4().hex[:6]}")
    _rate(db, inactive_clinic, doctor, 5)

    result = public_top_rated_doctors(db=db)

    assert doctor.full_name not in [r.doctor_name for r in result]


def test_public_top_rated_doctors_includes_department_and_clinic_name(db, clinic, department):
    doctor = _doctor(db, clinic, department, f"Dr. Named {uuid.uuid4().hex[:6]}")
    _rate(db, clinic, doctor, 5)

    result = public_top_rated_doctors(db=db)

    row = _by_name(result, doctor.full_name)
    assert row.department_name == "Cardiology"
    assert row.clinic_name == clinic.name


def test_public_top_rated_doctors_visit_count_is_all_completed_appointments(db, clinic, department):
    doctor = _doctor(db, clinic, department, f"Dr. Visits {uuid.uuid4().hex[:6]}")
    _rate(db, clinic, doctor, 5)
    _completed_appointment(db, clinic, doctor)

    result = public_top_rated_doctors(db=db)

    row = _by_name(result, doctor.full_name)
    assert row.visit_count == 2
    assert row.rating_count == 1
