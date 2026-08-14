import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User
from app.services import scheduler as scheduler_module


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


@pytest.fixture
def use_test_session_for_scheduler(db, monkeypatch):
    """The tick under test opens its own SessionLocal() (a fresh connection), which
    would never see this test's flushed-but-uncommitted rows. Pointing SessionLocal at
    the test's own connection-bound session instead means the tick's internal
    db.commit() calls only commit the fixture's SAVEPOINT (auto-restarted per
    conftest's after_transaction_end listener) — the outer transaction still rolls back
    everything at teardown, real seeded-clinic data included."""
    monkeypatch.setattr(scheduler_module, "SessionLocal", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)


def _upcoming_appointment(db, clinic, *, starts_in):
    # Booked well in the past (2 hours before "now") so every reminder window has
    # genuine lead time — see test_appointment_reminders.py's
    # test_appointment_booked_5_minutes_before_start_gets_only_the_5m_reminder for
    # the dedicated coverage of a genuinely-late booking instead.
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
        created_at=now - timedelta(hours=2),
    )
    db.add(appointment)
    db.flush()
    return appointment


def test_reminder_tick_sends_the_due_reminder_for_the_patient(db, clinic, use_test_session_for_scheduler):
    from app.models.notification import Notification

    appointment = _upcoming_appointment(db, clinic, starts_in=timedelta(minutes=4))

    scheduler_module.run_appointment_reminder_tick()

    types = sorted(
        n.type
        for n in db.execute(
            select(Notification).where(
                Notification.clinic_id == clinic.id, Notification.user_id == appointment.patient_id
            )
        ).scalars().all()
    )
    assert types == ["appointment_reminder_60m"]


def test_reminder_tick_skips_inactive_clinics(db, clinic, use_test_session_for_scheduler):
    from app.models.notification import Notification

    clinic.is_active = False
    db.flush()
    _upcoming_appointment(db, clinic, starts_in=timedelta(minutes=4))

    scheduler_module.run_appointment_reminder_tick()

    notifications = db.execute(select(Notification).where(Notification.clinic_id == clinic.id)).scalars().all()
    assert notifications == []
