import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from app.api.notifications import (
    get_push_public_key,
    list_my_notifications,
    mark_all_notifications_read,
    mark_notification_read,
    subscribe_to_push,
    unsubscribe_from_push,
)
from app.core.config import settings
from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.notification import Notification
from app.models.push_subscription import PushSubscription
from app.models.slot import Slot
from app.models.user import User
from app.schemas.notification import PushSubscriptionIn, PushSubscriptionKeysIn, PushUnsubscribeIn
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


def test_lazy_auto_complete_creates_notification_without_the_scheduler_tick(db, clinic, patient, doctor):
    # Reproduces a reported gap: an 'appointment_auto_completed' notification was only
    # ever created by the scheduler's backstop tick (app/services/scheduler.py). In
    # practice the lazy auto_complete_past_appointments call at the top of every
    # relevant endpoint/tool almost always flips the appointment to 'completed' first,
    # so by the time the tick ran there was nothing left for it to catch — most patients
    # never got a completion notification at all. Now auto_complete_past_appointments
    # itself creates the notification, so it fires on the lazy path too.
    from app.services.appointments import auto_complete_past_appointments

    slot = _future_slot(db, clinic, doctor, hours_from_now=-1)
    slot.status = "booked"
    appointment = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id,
        status="confirmed", booked_via="chatbot",
    )
    db.add(appointment)
    db.flush()

    auto_complete_past_appointments(db, clinic.id)

    assert appointment.status == "completed"
    notifications = (
        db.query(Notification)
        .filter(Notification.user_id == patient.id, Notification.type == "appointment_auto_completed")
        .all()
    )
    assert len(notifications) == 1
    assert doctor.full_name in notifications[0].message


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


def test_get_push_public_key_returns_configured_key(monkeypatch):
    monkeypatch.setattr(settings, "VAPID_PUBLIC_KEY", "some-public-key")
    assert get_push_public_key().public_key == "some-public-key"


def test_get_push_public_key_empty_when_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "VAPID_PUBLIC_KEY", "")
    assert get_push_public_key().public_key == ""


def _push_payload(endpoint: str) -> PushSubscriptionIn:
    return PushSubscriptionIn(
        endpoint=endpoint, keys=PushSubscriptionKeysIn(p256dh="fake-p256dh", auth="fake-auth")
    )


def test_subscribe_to_push_creates_a_row(db, clinic, patient):
    subscribe_to_push(
        payload=_push_payload("https://fcm.googleapis.com/fcm/send/new-endpoint"),
        current_user=patient, db=db,
    )
    row = db.query(PushSubscription).filter(PushSubscription.user_id == patient.id).one()
    assert row.endpoint == "https://fcm.googleapis.com/fcm/send/new-endpoint"
    assert row.clinic_id == clinic.id


def test_subscribe_to_push_is_idempotent_by_endpoint(db, patient):
    endpoint = "https://fcm.googleapis.com/fcm/send/same-endpoint"
    subscribe_to_push(payload=_push_payload(endpoint), current_user=patient, db=db)
    subscribe_to_push(payload=_push_payload(endpoint), current_user=patient, db=db)

    rows = db.query(PushSubscription).filter(PushSubscription.endpoint == endpoint).all()
    assert len(rows) == 1


def test_unsubscribe_from_push_removes_the_row(db, patient):
    endpoint = "https://fcm.googleapis.com/fcm/send/to-remove"
    subscribe_to_push(payload=_push_payload(endpoint), current_user=patient, db=db)

    unsubscribe_from_push(payload=PushUnsubscribeIn(endpoint=endpoint), current_user=patient, db=db)

    remaining = db.query(PushSubscription).filter(PushSubscription.endpoint == endpoint).one_or_none()
    assert remaining is None


def test_unsubscribe_from_push_does_not_remove_another_patients_subscription(db, patient, other_patient):
    endpoint = "https://fcm.googleapis.com/fcm/send/belongs-to-other"
    subscribe_to_push(payload=_push_payload(endpoint), current_user=other_patient, db=db)

    unsubscribe_from_push(payload=PushUnsubscribeIn(endpoint=endpoint), current_user=patient, db=db)

    remaining = db.query(PushSubscription).filter(PushSubscription.endpoint == endpoint).one_or_none()
    assert remaining is not None
