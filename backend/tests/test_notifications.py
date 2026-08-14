import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from app.api.notifications import list_my_notifications, mark_all_notifications_read, mark_notification_read
from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.notification import Notification
from app.models.slot import Slot
from app.models.user import User
from app.services import booking_engine
from app.services.notifications import create_notification, format_appointment_datetime


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
        full_name="Dr. John Other", is_active=True,
    )
    db.add(d)
    db.flush()
    return d


def _future_slot(db, clinic, doctor, hours_from_now=48):
    now = datetime.now(timezone.utc)
    start = now + timedelta(hours=hours_from_now)
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start, end_utc=start + timedelta(minutes=30))
    db.add(slot)
    db.flush()
    return slot


# --- format_appointment_datetime -------------------------------------------------

def test_format_appointment_datetime_matches_clinic_timezone():
    start_utc = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)
    assert format_appointment_datetime(start_utc, "UTC") == "5 Aug at 10:00 AM"


# --- create_notification ----------------------------------------------------------

def test_create_notification_defaults_to_unread(db, clinic, patient):
    notification = create_notification(
        db, clinic_id=clinic.id, user_id=patient.id, notif_type="appointment_booked", message="Test message",
    )
    db.flush()
    assert notification.read_at is None
    assert notification.message == "Test message"
    assert notification.type == "appointment_booked"


# --- booking_engine triggers -------------------------------------------------------

def test_booking_appointment_creates_appointment_booked_notification(db, clinic, patient, doctor):
    slot = _future_slot(db, clinic, doctor)

    booking_engine.book_appointment(db, clinic_id=clinic.id, patient_id=patient.id, slot_id=slot.id)

    notifications = db.query(Notification).filter(Notification.user_id == patient.id).all()
    assert len(notifications) == 1
    assert notifications[0].type == "appointment_booked"
    assert doctor.full_name in notifications[0].message


def test_cancelling_appointment_creates_appointment_cancelled_notification(db, clinic, patient, doctor):
    slot = _future_slot(db, clinic, doctor)
    appointment = booking_engine.book_appointment(db, clinic_id=clinic.id, patient_id=patient.id, slot_id=slot.id)

    booking_engine.cancel_appointment(db, clinic_id=clinic.id, patient_id=patient.id, appointment_id=appointment.id)

    notifications = (
        db.query(Notification)
        .filter(Notification.user_id == patient.id, Notification.type == "appointment_cancelled")
        .all()
    )
    assert len(notifications) == 1
    assert "cancelled" in notifications[0].message


def test_rescheduling_appointment_creates_appointment_rescheduled_notification(db, clinic, patient, doctor):
    slot = _future_slot(db, clinic, doctor, hours_from_now=72)
    new_slot = _future_slot(db, clinic, doctor, hours_from_now=96)
    appointment = booking_engine.book_appointment(db, clinic_id=clinic.id, patient_id=patient.id, slot_id=slot.id)

    booking_engine.reschedule_appointment(
        db, clinic_id=clinic.id, patient_id=patient.id, appointment_id=appointment.id, new_slot_id=new_slot.id,
    )

    notifications = (
        db.query(Notification)
        .filter(Notification.user_id == patient.id, Notification.type == "appointment_rescheduled")
        .all()
    )
    assert len(notifications) == 1
    assert "rescheduled" in notifications[0].message
    assert "from" not in notifications[0].message


def test_rescheduling_appointment_to_a_different_doctor_names_both_doctors(db, clinic, patient, doctor, other_doctor):
    slot = _future_slot(db, clinic, doctor, hours_from_now=72)
    new_slot = _future_slot(db, clinic, other_doctor, hours_from_now=96)
    appointment = booking_engine.book_appointment(db, clinic_id=clinic.id, patient_id=patient.id, slot_id=slot.id)

    booking_engine.reschedule_appointment(
        db, clinic_id=clinic.id, patient_id=patient.id, appointment_id=appointment.id, new_slot_id=new_slot.id,
    )

    notifications = (
        db.query(Notification)
        .filter(Notification.user_id == patient.id, Notification.type == "appointment_rescheduled")
        .all()
    )
    assert len(notifications) == 1
    message = notifications[0].message
    assert "rescheduled from" in message
    assert doctor.full_name in message
    assert other_doctor.full_name in message


# --- booking_engine email triggers --------------------------------------------------
# Emails are fired off a background thread (see app.services.email._send_in_background)
# rather than via FastAPI's BackgroundTasks — booking_engine is shared by the REST API
# AND the chatbot's tool-calling path, only one of which has BackgroundTasks available.
# monkeypatching the send_* wrapper itself (rather than the low-level send_email) lets
# these tests assert on the exact recipient/content without waiting on/joining a thread.

def test_booking_appointment_sends_a_confirmation_email(db, clinic, patient, doctor, monkeypatch):
    sent = []
    monkeypatch.setattr(booking_engine, "send_appointment_booked_email", lambda **kwargs: sent.append(kwargs))
    slot = _future_slot(db, clinic, doctor)

    booking_engine.book_appointment(db, clinic_id=clinic.id, patient_id=patient.id, slot_id=slot.id)

    assert len(sent) == 1
    assert sent[0]["to"] == patient.email
    assert sent[0]["doctor_name"] == doctor.full_name


def test_cancelling_appointment_sends_a_cancellation_email(db, clinic, patient, doctor, monkeypatch):
    sent = []
    monkeypatch.setattr(booking_engine, "send_appointment_cancelled_email", lambda **kwargs: sent.append(kwargs))
    slot = _future_slot(db, clinic, doctor)
    appointment = booking_engine.book_appointment(db, clinic_id=clinic.id, patient_id=patient.id, slot_id=slot.id)

    booking_engine.cancel_appointment(db, clinic_id=clinic.id, patient_id=patient.id, appointment_id=appointment.id)

    assert len(sent) == 1
    assert sent[0]["to"] == patient.email


def test_rescheduling_appointment_sends_a_reschedule_email(db, clinic, patient, doctor, monkeypatch):
    sent = []
    monkeypatch.setattr(booking_engine, "send_appointment_rescheduled_email", lambda **kwargs: sent.append(kwargs))
    slot = _future_slot(db, clinic, doctor, hours_from_now=72)
    new_slot = _future_slot(db, clinic, doctor, hours_from_now=96)
    appointment = booking_engine.book_appointment(db, clinic_id=clinic.id, patient_id=patient.id, slot_id=slot.id)

    booking_engine.reschedule_appointment(
        db, clinic_id=clinic.id, patient_id=patient.id, appointment_id=appointment.id, new_slot_id=new_slot.id,
    )

    assert len(sent) == 1
    assert sent[0]["to"] == patient.email
    assert sent[0]["doctor_changed"] is False


def test_rescheduling_to_a_different_doctor_flags_doctor_changed_in_the_email(
    db, clinic, patient, doctor, other_doctor, monkeypatch
):
    sent = []
    monkeypatch.setattr(booking_engine, "send_appointment_rescheduled_email", lambda **kwargs: sent.append(kwargs))
    slot = _future_slot(db, clinic, doctor, hours_from_now=72)
    new_slot = _future_slot(db, clinic, other_doctor, hours_from_now=96)
    appointment = booking_engine.book_appointment(db, clinic_id=clinic.id, patient_id=patient.id, slot_id=slot.id)

    booking_engine.reschedule_appointment(
        db, clinic_id=clinic.id, patient_id=patient.id, appointment_id=appointment.id, new_slot_id=new_slot.id,
    )

    assert len(sent) == 1
    assert sent[0]["doctor_changed"] is True
    assert sent[0]["old_doctor_name"] == doctor.full_name
    assert sent[0]["new_doctor_name"] == other_doctor.full_name


# --- API endpoints -------------------------------------------------------------

def test_list_my_notifications_is_own_and_clinic_scoped_only(db, clinic, patient, other_patient):
    create_notification(db, clinic_id=clinic.id, user_id=patient.id, notif_type="appointment_booked", message="Mine")
    create_notification(
        db, clinic_id=clinic.id, user_id=other_patient.id, notif_type="appointment_booked", message="Not mine",
    )
    db.commit()

    result = list_my_notifications(current_user=patient, db=db)

    assert len(result.notifications) == 1
    assert result.notifications[0].message == "Mine"
    assert result.unread_count == 1


def test_list_my_notifications_newest_first(db, clinic, patient):
    create_notification(db, clinic_id=clinic.id, user_id=patient.id, notif_type="appointment_booked", message="First")
    create_notification(db, clinic_id=clinic.id, user_id=patient.id, notif_type="appointment_booked", message="Second")
    db.commit()

    result = list_my_notifications(current_user=patient, db=db)

    assert [n.message for n in result.notifications] == ["Second", "First"]


def test_mark_notification_read_sets_read_at(db, clinic, patient):
    notification = create_notification(
        db, clinic_id=clinic.id, user_id=patient.id, notif_type="appointment_booked", message="Hello",
    )
    db.commit()

    result = mark_notification_read(notification_id=notification.id, current_user=patient, db=db)

    assert result.read_at is not None


def test_mark_notification_read_rejects_another_patients_notification(db, clinic, patient, other_patient):
    notification = create_notification(
        db, clinic_id=clinic.id, user_id=other_patient.id, notif_type="appointment_booked", message="Not yours",
    )
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        mark_notification_read(notification_id=notification.id, current_user=patient, db=db)

    assert exc_info.value.status_code == 404


def test_mark_all_notifications_read_clears_unread_count(db, clinic, patient):
    create_notification(db, clinic_id=clinic.id, user_id=patient.id, notif_type="appointment_booked", message="A")
    create_notification(db, clinic_id=clinic.id, user_id=patient.id, notif_type="appointment_booked", message="B")
    db.commit()

    result = mark_all_notifications_read(current_user=patient, db=db)

    assert result.unread_count == 0
    assert all(n.read_at is not None for n in result.notifications)
