import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.notification import Notification
from app.models.slot import Slot
from app.models.user import User
from app.services.appointment_reminders import send_due_reminders_for_clinic


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _confirmed_appointment(db, clinic, *, starts_in: timedelta, booked_at: datetime | None = None):
    """`booked_at` defaults to well in the past (2 hours before "now") so every
    reminder window has genuine lead time — isolates "has this appointment now
    crossed threshold X" tests from the separate booked-too-late-for-window-X
    concern covered by its own dedicated tests below. Pass `booked_at` explicitly
    to simulate a late/near-term booking."""
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

    now = datetime.now(timezone.utc)
    start = now + starts_in
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start, end_utc=start + timedelta(minutes=30))
    db.add(slot)
    db.flush()

    appointment = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="confirmed",
        created_at=booked_at if booked_at is not None else now - timedelta(hours=2),
    )
    db.add(appointment)
    db.flush()
    return appointment


def _reminder_types(db, clinic, appointment):
    rows = db.execute(
        select(Notification).where(
            Notification.clinic_id == clinic.id,
            Notification.related_appointment_id == appointment.id,
        )
    ).scalars().all()
    return sorted(r.type for r in rows)


def test_far_future_appointment_gets_no_reminders_yet(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(hours=2))

    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == []


def test_appointment_59_minutes_out_gets_the_60m_reminder_only(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=59))

    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == ["appointment_reminder_60m"]


def test_appointment_29_minutes_out_gets_60m_and_30m_reminders(db, clinic):
    # Crossed both the 60m and 30m thresholds already (e.g. a tick was delayed) —
    # both are independent and both must fire, not just the nearest one.
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=29))

    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == ["appointment_reminder_30m", "appointment_reminder_60m"]


def test_appointment_4_minutes_out_gets_60m_30m_and_5m_reminders(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=4))

    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == [
        "appointment_reminder_30m", "appointment_reminder_5m", "appointment_reminder_60m",
    ]


def test_appointment_starting_now_gets_all_four_reminders(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=0))

    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == [
        "appointment_reminder_30m", "appointment_reminder_5m", "appointment_reminder_60m", "appointment_starting",
    ]


def test_a_second_tick_never_sends_a_duplicate_of_an_already_sent_reminder(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=59))

    send_due_reminders_for_clinic(db, clinic.id)
    send_due_reminders_for_clinic(db, clinic.id)
    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == ["appointment_reminder_60m"]


def test_reminders_accumulate_as_the_appointment_gets_closer_across_multiple_ticks(db, clinic, monkeypatch):
    # Simulates real ticks over time: the appointment starts 61 minutes out (nothing
    # due yet), then "time passes" and it's re-checked at 59, then 29, then 4 minutes
    # out — each tick must only add the newly-crossed reminder(s), never re-send an
    # earlier one.
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=61))

    send_due_reminders_for_clinic(db, clinic.id)
    assert _reminder_types(db, clinic, appointment) == []

    slot = db.get(Slot, appointment.slot_id)
    slot.start_utc = datetime.now(timezone.utc) + timedelta(minutes=59)
    db.flush()
    send_due_reminders_for_clinic(db, clinic.id)
    assert _reminder_types(db, clinic, appointment) == ["appointment_reminder_60m"]

    slot.start_utc = datetime.now(timezone.utc) + timedelta(minutes=29)
    db.flush()
    send_due_reminders_for_clinic(db, clinic.id)
    assert _reminder_types(db, clinic, appointment) == ["appointment_reminder_30m", "appointment_reminder_60m"]


def test_cancelled_appointment_gets_no_reminders(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=4))
    appointment.status = "cancelled"
    db.flush()

    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == []


def test_reminder_message_names_the_doctor_and_time_remaining(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=59))

    send_due_reminders_for_clinic(db, clinic.id)

    notification = db.execute(
        select(Notification).where(Notification.related_appointment_id == appointment.id)
    ).scalar_one()
    assert "Dr. Jane Example" in notification.message
    assert "1 hour" in notification.message
    assert notification.read_at is None


def test_starting_reminder_message_says_starting_now(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=0))

    send_due_reminders_for_clinic(db, clinic.id)

    notification = db.execute(
        select(Notification).where(
            Notification.related_appointment_id == appointment.id,
            Notification.type == "appointment_starting",
        )
    ).scalar_one()
    assert "starting now" in notification.message


def test_appointment_booked_5_minutes_before_start_gets_only_the_5m_reminder(db, clinic):
    # Reported live: booking an appointment at 12:55 for a 1:00pm start fired the
    # 60m, 30m, AND 5m reminders all at once, right at booking — every window's
    # threshold was already in the past the instant the appointment existed. A "in
    # 1 hour"/"in 30 minutes" reminder is nonsensical for an appointment that was
    # never actually that far out at any point after being booked; only the window
    # that genuinely had lead time (5m here) should fire.
    now = datetime.now(timezone.utc)
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=5), booked_at=now)

    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == ["appointment_reminder_5m"]


def test_appointment_booked_at_the_exact_start_time_still_gets_the_starting_reminder(db, clinic):
    # The "appointment_starting" (0-minute) window is exempt from the lead-time
    # gate — an appointment starting right now is always a real, meaningful event
    # to notify about, no matter how late it was booked.
    now = datetime.now(timezone.utc)
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=0), booked_at=now)

    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == ["appointment_starting"]
