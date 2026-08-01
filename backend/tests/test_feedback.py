import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.api.admin_feedback import list_feedback
from app.core.tenancy import ClinicContext
from app.models.appointment import Appointment
from app.models.appointment_feedback import AppointmentFeedback
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User
from app.services import booking_engine
from app.services.feedback import (
    BAD_RATING_ACK,
    GOOD_RATING_ACK,
    NEUTRAL_RATING_ACK,
    build_feedback_prompt,
    get_pending_feedback_appointments,
    submit_feedback,
)


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


@pytest.fixture
def admin(db, clinic):
    a = User(
        clinic_id=clinic.id, role="admin", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Admin User",
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


@pytest.fixture
def doctor(db, clinic, department):
    d = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
        full_name="Dr. Jane Example", is_active=True,
    )
    db.add(d)
    db.flush()
    return d


@pytest.fixture
def other_doctor(db, clinic, department):
    d = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
        full_name="Dr. John Second", is_active=True,
    )
    db.add(d)
    db.flush()
    return d


def _completed_appointment(db, clinic, patient, doctor, hours_ago=24):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_ago)
    slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start, end_utc=start + timedelta(minutes=30),
        status="booked",
    )
    db.add(slot)
    db.flush()
    appointment = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="completed",
    )
    db.add(appointment)
    db.flush()
    return appointment


def _confirmed_appointment(db, clinic, patient, doctor, hours_from_now=48):
    now = datetime.now(timezone.utc)
    start = now + timedelta(hours=hours_from_now)
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start, end_utc=start + timedelta(minutes=30))
    db.add(slot)
    db.flush()
    return booking_engine.book_appointment(db, clinic_id=clinic.id, patient_id=patient.id, slot_id=slot.id)


def _ctx(clinic, patient):
    return ClinicContext(clinic_id=clinic.id, user_id=patient.id, role="patient")


# --- get_pending_feedback_appointments --------------------------------------------

def test_pending_feedback_includes_completed_unrated_appointment(db, clinic, patient, doctor):
    appointment = _completed_appointment(db, clinic, patient, doctor)
    db.commit()

    pending = get_pending_feedback_appointments(db, _ctx(clinic, patient))

    assert len(pending) == 1
    assert pending[0]["appointment_id"] == appointment.id
    assert pending[0]["doctor_name"] == "Dr. Jane Example"


def test_pending_feedback_excludes_confirmed_appointments(db, clinic, patient, doctor):
    _confirmed_appointment(db, clinic, patient, doctor)
    db.commit()

    pending = get_pending_feedback_appointments(db, _ctx(clinic, patient))

    assert pending == []


def test_pending_feedback_excludes_already_rated_appointments(db, clinic, patient, doctor):
    appointment = _completed_appointment(db, clinic, patient, doctor)
    db.commit()
    submit_feedback(db, _ctx(clinic, patient), [appointment.id], rating=5, reason=None)

    pending = get_pending_feedback_appointments(db, _ctx(clinic, patient))

    assert pending == []


def test_pending_feedback_surfaces_only_the_single_latest_unrated_appointment(db, clinic, patient, doctor, other_doctor):
    _completed_appointment(db, clinic, patient, doctor, hours_ago=72)
    _completed_appointment(db, clinic, patient, doctor, hours_ago=48)
    latest = _completed_appointment(db, clinic, patient, other_doctor, hours_ago=24)
    db.commit()

    pending = get_pending_feedback_appointments(db, _ctx(clinic, patient))

    assert len(pending) == 1
    assert pending[0]["appointment_id"] == latest.id
    assert pending[0]["doctor_name"] == "Dr. John Second"


def test_pending_feedback_is_own_and_clinic_scoped_only(db, clinic, patient, other_patient, doctor):
    _completed_appointment(db, clinic, other_patient, doctor)
    db.commit()

    pending = get_pending_feedback_appointments(db, _ctx(clinic, patient))

    assert pending == []


# --- build_feedback_prompt ----------------------------------------------------------

def test_build_feedback_prompt_single_appointment_names_doctor_and_time():
    prompt = build_feedback_prompt([{"appointment_id": uuid.uuid4(), "doctor_name": "Dr. Jane Example", "when": "5 Aug at 10:00 AM"}])
    assert "Dr. Jane Example" in prompt
    assert "5 Aug at 10:00 AM" in prompt


def test_build_feedback_prompt_returns_none_when_no_pending_appointments():
    assert build_feedback_prompt([]) is None


# --- submit_feedback ------------------------------------------------------------

def test_submit_feedback_low_rating_saves_reason_and_returns_bad_ack(db, clinic, patient, doctor):
    appointment = _completed_appointment(db, clinic, patient, doctor)
    db.commit()

    message = submit_feedback(db, _ctx(clinic, patient), [appointment.id], rating=1, reason="Long wait time")

    assert message == BAD_RATING_ACK
    row = db.query(AppointmentFeedback).filter(AppointmentFeedback.appointment_id == appointment.id).one()
    assert row.rating == 1
    assert row.reason == "Long wait time"


def test_submit_feedback_high_rating_discards_reason_and_returns_good_ack(db, clinic, patient, doctor):
    appointment = _completed_appointment(db, clinic, patient, doctor)
    db.commit()

    message = submit_feedback(db, _ctx(clinic, patient), [appointment.id], rating=5, reason="should be ignored")

    assert message == GOOD_RATING_ACK
    row = db.query(AppointmentFeedback).filter(AppointmentFeedback.appointment_id == appointment.id).one()
    assert row.rating == 5
    assert row.reason is None


def test_submit_feedback_middle_rating_returns_neutral_ack(db, clinic, patient, doctor):
    appointment = _completed_appointment(db, clinic, patient, doctor)
    db.commit()

    message = submit_feedback(db, _ctx(clinic, patient), [appointment.id], rating=3, reason=None)

    assert message == NEUTRAL_RATING_ACK




def test_submit_feedback_ignores_appointment_belonging_to_another_patient(db, clinic, patient, other_patient, doctor):
    appointment = _completed_appointment(db, clinic, other_patient, doctor)
    db.commit()

    submit_feedback(db, _ctx(clinic, patient), [appointment.id], rating=5, reason=None)

    # Scoped to this test's own appointment, not a bare table-wide count() — the
    # dev DB this suite runs against can carry real feedback rows from actual app
    # usage alongside the test's isolated transaction, so a global count is not a
    # safe assertion here.
    assert db.query(AppointmentFeedback).filter(AppointmentFeedback.appointment_id == appointment.id).count() == 0


def test_submit_feedback_ignores_already_rated_appointment_without_duplicating(db, clinic, patient, doctor):
    appointment = _completed_appointment(db, clinic, patient, doctor)
    db.commit()
    submit_feedback(db, _ctx(clinic, patient), [appointment.id], rating=2, reason="first")

    submit_feedback(db, _ctx(clinic, patient), [appointment.id], rating=5, reason=None)

    rows = db.query(AppointmentFeedback).filter(AppointmentFeedback.appointment_id == appointment.id).all()
    assert len(rows) == 1
    assert rows[0].rating == 2


# --- admin /admin/feedback endpoint ------------------------------------------------

def test_admin_list_feedback_is_clinic_scoped_and_newest_first(db, clinic, patient, doctor, admin):
    appt1 = _completed_appointment(db, clinic, patient, doctor, hours_ago=48)
    appt2 = _completed_appointment(db, clinic, patient, doctor, hours_ago=24)
    db.commit()
    submit_feedback(db, _ctx(clinic, patient), [appt1.id], rating=3, reason=None)
    submit_feedback(db, _ctx(clinic, patient), [appt2.id], rating=1, reason="Rude staff")

    result = list_feedback(limit=20, offset=0, tone=None, current_user=admin, db=db)

    assert result.total == 2
    assert result.items[0].reason == "Rude staff"
    assert result.items[0].patient_name == "Pat Ient"
    assert result.items[0].doctor_name == "Dr. Jane Example"


def test_admin_list_feedback_computes_clinic_wide_average_and_tone_counts(db, clinic, patient, doctor, admin):
    appt1 = _completed_appointment(db, clinic, patient, doctor, hours_ago=72)
    appt2 = _completed_appointment(db, clinic, patient, doctor, hours_ago=48)
    appt3 = _completed_appointment(db, clinic, patient, doctor, hours_ago=24)
    db.commit()
    submit_feedback(db, _ctx(clinic, patient), [appt1.id], rating=5, reason=None)
    submit_feedback(db, _ctx(clinic, patient), [appt2.id], rating=3, reason=None)
    submit_feedback(db, _ctx(clinic, patient), [appt3.id], rating=1, reason="Long wait")

    result = list_feedback(limit=20, offset=0, tone=None, current_user=admin, db=db)

    assert result.average_rating == 3.0
    assert result.tone_counts.good == 1
    assert result.tone_counts.neutral == 1
    assert result.tone_counts.bad == 1


def test_admin_list_feedback_tone_filter_narrows_items_but_not_summary(db, clinic, patient, doctor, admin):
    appt1 = _completed_appointment(db, clinic, patient, doctor, hours_ago=72)
    appt2 = _completed_appointment(db, clinic, patient, doctor, hours_ago=24)
    db.commit()
    submit_feedback(db, _ctx(clinic, patient), [appt1.id], rating=5, reason=None)
    submit_feedback(db, _ctx(clinic, patient), [appt2.id], rating=1, reason="Rude staff")

    result = list_feedback(limit=20, offset=0, tone="bad", current_user=admin, db=db)

    assert result.total == 1
    assert len(result.items) == 1
    assert result.items[0].rating == 1
    # Summary stays clinic-wide even though the filtered item list only has one row.
    assert result.tone_counts.good == 1
    assert result.tone_counts.bad == 1
