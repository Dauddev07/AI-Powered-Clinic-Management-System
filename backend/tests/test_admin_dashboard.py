import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.api.admin_dashboard import get_admin_dashboard_stats, get_top_rated_doctors
from app.models.appointment import Appointment
from app.models.appointment_feedback import AppointmentFeedback
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
    """Creates a completed appointment and rates it — the unit this feature
    actually cares about (one rated visit), not a bare feedback row."""
    appointment = _completed_appointment(db, clinic, doctor)
    feedback = AppointmentFeedback(
        clinic_id=clinic.id, patient_id=appointment.patient_id, appointment_id=appointment.id,
        doctor_id=doctor.id, rating=rating,
    )
    db.add(feedback)
    db.flush()
    return appointment


def test_top_rated_doctors_is_empty_when_no_ratings_exist(db, clinic, admin, department):
    _doctor(db, clinic, department, "Dr. Unrated")

    result = get_top_rated_doctors(current_user=admin, db=db)

    assert result == []


def test_top_rated_doctors_orders_by_average_rating_descending(db, clinic, admin, department):
    high = _doctor(db, clinic, department, "Dr. High")
    low = _doctor(db, clinic, department, "Dr. Low")
    _rate(db, clinic, high, 5)
    _rate(db, clinic, low, 3)

    result = get_top_rated_doctors(current_user=admin, db=db)

    assert [r.doctor_name for r in result] == ["Dr. High", "Dr. Low"]
    assert result[0].average_rating == 5.0
    assert result[1].average_rating == 3.0


def test_top_rated_doctors_averages_multiple_ratings_for_the_same_doctor(db, clinic, admin, department):
    doctor = _doctor(db, clinic, department, "Dr. Mixed Ratings")
    _rate(db, clinic, doctor, 5)
    _rate(db, clinic, doctor, 4)

    result = get_top_rated_doctors(current_user=admin, db=db)

    assert result[0].average_rating == 4.5
    assert result[0].rating_count == 2


def test_top_rated_doctors_limits_to_three(db, clinic, admin, department):
    for i in range(5):
        doctor = _doctor(db, clinic, department, f"Dr. {i}")
        _rate(db, clinic, doctor, 5)

    result = get_top_rated_doctors(current_user=admin, db=db)

    assert len(result) == 3


def test_top_rated_doctors_breaks_ties_by_rating_count(db, clinic, admin, department):
    more_ratings = _doctor(db, clinic, department, "Dr. More Ratings")
    fewer_ratings = _doctor(db, clinic, department, "Dr. Fewer Ratings")
    _rate(db, clinic, more_ratings, 5)
    _rate(db, clinic, more_ratings, 5)
    _rate(db, clinic, fewer_ratings, 5)

    result = get_top_rated_doctors(current_user=admin, db=db)

    assert result[0].doctor_name == "Dr. More Ratings"
    assert result[0].rating_count == 2
    assert result[1].rating_count == 1


def test_top_rated_doctors_visit_count_is_all_completed_appointments_not_just_rated_ones(
    db, clinic, admin, department
):
    doctor = _doctor(db, clinic, department, "Dr. Popular")
    _rate(db, clinic, doctor, 5)  # 1 rated (and therefore completed) visit
    _completed_appointment(db, clinic, doctor)  # 1 more completed visit, never rated

    result = get_top_rated_doctors(current_user=admin, db=db)

    assert result[0].rating_count == 1
    assert result[0].visit_count == 2


def test_top_rated_doctors_excludes_inactive_doctors(db, clinic, admin, department):
    inactive = _doctor(db, clinic, department, "Dr. Inactive")
    inactive.is_active = False
    db.flush()
    _rate(db, clinic, inactive, 5)

    result = get_top_rated_doctors(current_user=admin, db=db)

    assert result == []


def test_top_rated_doctors_is_clinic_scoped(db, clinic, admin, department):
    other_clinic = Clinic(name="Other Clinic", slug=f"other-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(other_clinic)
    db.flush()
    other_department = Department(clinic_id=other_clinic.id, name="Neurology")
    db.add(other_department)
    db.flush()
    other_doctor = _doctor(db, other_clinic, other_department, "Dr. Other Clinic")
    _rate(db, other_clinic, other_doctor, 5)

    result = get_top_rated_doctors(current_user=admin, db=db)

    assert result == []


def test_top_rated_doctors_includes_department_name(db, clinic, admin, department):
    doctor = _doctor(db, clinic, department, "Dr. Cardio")
    _rate(db, clinic, doctor, 5)

    result = get_top_rated_doctors(current_user=admin, db=db)

    assert result[0].department_name == "Cardiology"
