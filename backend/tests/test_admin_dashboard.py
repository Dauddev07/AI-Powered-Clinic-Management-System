import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.api.admin_dashboard import get_admin_dashboard_stats
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
def admin(db, clinic):
    a = User(
        clinic_id=clinic.id, role="admin", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Ad Min",
    )
    db.add(a)
    db.flush()
    return a


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


def _appointment_today(db, clinic, doctor, status="confirmed", hours_from_now=1):
    patient = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(patient)
    db.flush()

    now = datetime.now(timezone.utc)
    start = now + timedelta(hours=hours_from_now)
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start, end_utc=start + timedelta(minutes=30))
    db.add(slot)
    db.flush()

    appointment = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status=status,
    )
    db.add(appointment)
    db.flush()
    return appointment


def test_busiest_doctors_today_is_empty_when_nothing_booked(db, clinic, admin):
    stats = get_admin_dashboard_stats(current_user=admin, db=db)
    assert stats.busiest_doctors_today == []


def test_busiest_doctors_today_counts_confirmed_and_completed_busiest_first(db, clinic, admin, department):
    busy_doctor = _doctor(db, clinic, department, "Dr. Busy")
    quiet_doctor = _doctor(db, clinic, department, "Dr. Quiet")

    _appointment_today(db, clinic, busy_doctor, hours_from_now=1, status="confirmed")
    _appointment_today(db, clinic, busy_doctor, hours_from_now=2, status="completed")
    _appointment_today(db, clinic, quiet_doctor, hours_from_now=3, status="confirmed")

    stats = get_admin_dashboard_stats(current_user=admin, db=db)

    breakdown = {row.doctor_name: row.count for row in stats.busiest_doctors_today}
    assert breakdown == {"Dr. Busy": 2, "Dr. Quiet": 1}
    # Busiest first.
    assert stats.busiest_doctors_today[0].doctor_name == "Dr. Busy"


def test_busiest_doctors_today_excludes_cancelled_appointments(db, clinic, admin, department):
    doctor = _doctor(db, clinic, department, "Dr. Cancelled Only")
    _appointment_today(db, clinic, doctor, hours_from_now=1, status="cancelled")

    stats = get_admin_dashboard_stats(current_user=admin, db=db)

    assert stats.busiest_doctors_today == []


def test_busiest_doctors_today_excludes_cancelled_but_counts_the_rest(db, clinic, admin, department):
    doctor = _doctor(db, clinic, department, "Dr. Mixed")
    _appointment_today(db, clinic, doctor, hours_from_now=1, status="confirmed")
    _appointment_today(db, clinic, doctor, hours_from_now=2, status="cancelled")
    _appointment_today(db, clinic, doctor, hours_from_now=3, status="cancelled")

    stats = get_admin_dashboard_stats(current_user=admin, db=db)

    assert len(stats.busiest_doctors_today) == 1
    assert stats.busiest_doctors_today[0].count == 1


def test_busiest_doctors_today_excludes_appointments_outside_today(db, clinic, admin, department):
    doctor = _doctor(db, clinic, department, "Dr. Future")
    _appointment_today(db, clinic, doctor, hours_from_now=48)

    stats = get_admin_dashboard_stats(current_user=admin, db=db)

    assert stats.busiest_doctors_today == []


def test_busiest_doctors_today_is_clinic_scoped(db, clinic, admin, department):
    other_clinic = Clinic(name="Other Clinic", slug=f"other-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(other_clinic)
    db.flush()
    other_department = Department(clinic_id=other_clinic.id, name="Neurology")
    db.add(other_department)
    db.flush()
    other_doctor = _doctor(db, other_clinic, other_department, "Dr. Other Clinic")
    _appointment_today(db, other_clinic, other_doctor, hours_from_now=1)

    stats = get_admin_dashboard_stats(current_user=admin, db=db)

    assert stats.busiest_doctors_today == []
