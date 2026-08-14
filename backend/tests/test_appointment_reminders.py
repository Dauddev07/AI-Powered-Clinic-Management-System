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
from app.services import appointment_reminders as reminders_module
from app.services.appointment_reminders import send_due_reminders_for_clinic


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _confirmed_appointment(db, clinic, *, starts_in: timedelta, booked_at: datetime | None = None, email: str | None = None):
    """`booked_at` defaults to well in the past (2 hours before "now") so the reminder
    window has genuine lead time — isolates "has this appointment now crossed the
    60-minute threshold" tests from the separate booked-too-late-for-the-window concern
    covered by its own dedicated test below. Pass `booked_at` explicitly to simulate a
    late/near-term booking."""
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
        clinic_id=clinic.id, role="patient", email=email or f"{uuid.uuid4().hex}@example.com",
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


def test_far_future_appointment_gets_no_reminder_yet(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(hours=2))

    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == []


def test_appointment_59_minutes_out_gets_the_60m_reminder(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=59))

    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == ["appointment_reminder_60m"]


def test_a_second_tick_never_sends_a_duplicate_of_an_already_sent_reminder(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=59))

    send_due_reminders_for_clinic(db, clinic.id)
    send_due_reminders_for_clinic(db, clinic.id)
    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == ["appointment_reminder_60m"]


def test_cancelled_appointment_gets_no_reminder(db, clinic):
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=59))
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


def test_appointment_booked_30_minutes_before_start_gets_no_reminder(db, clinic):
    # Reported live (back when there were multiple reminder windows): booking an
    # appointment less than 60 minutes out fired the 60m reminder immediately, right
    # at booking — the threshold was already in the past the instant the appointment
    # existed. A "in 1 hour" reminder is nonsensical for an appointment that was never
    # actually that far out at any point after being booked.
    now = datetime.now(timezone.utc)
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=30), booked_at=now)

    send_due_reminders_for_clinic(db, clinic.id)

    assert _reminder_types(db, clinic, appointment) == []


def test_reminder_sends_an_email_to_the_patient(db, clinic, monkeypatch):
    sent = []
    monkeypatch.setattr(
        reminders_module,
        "send_appointment_reminder_email",
        lambda **kwargs: sent.append(kwargs),
    )
    appointment = _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=59), email="patient@example.com")

    send_due_reminders_for_clinic(db, clinic.id)

    assert len(sent) == 1
    assert sent[0]["to"] == "patient@example.com"
    assert sent[0]["doctor_name"] == "Dr. Jane Example"


def test_reminder_does_not_resend_the_email_on_a_second_tick(db, clinic, monkeypatch):
    sent = []
    monkeypatch.setattr(
        reminders_module,
        "send_appointment_reminder_email",
        lambda **kwargs: sent.append(kwargs),
    )
    _confirmed_appointment(db, clinic, starts_in=timedelta(minutes=59))

    send_due_reminders_for_clinic(db, clinic.id)
    send_due_reminders_for_clinic(db, clinic.id)

    assert len(sent) == 1
