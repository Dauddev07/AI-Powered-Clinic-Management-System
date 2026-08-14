"""Sends an in-app notification AND an email exactly 60 minutes before a confirmed
appointment's start — see app.services.scheduler.run_appointment_reminder_tick for
the job that calls this on a tick.

EXACTLY-ONCE, NOT "AT MOST ONE EVER": the reminder is keyed by (type,
related_appointment_id) — an appointment gets at most one 60-minute reminder over its
lifetime. A tick is idempotent: it only ever creates a reminder that doesn't already
exist for that appointment, so a delayed or re-run tick can never double-send. This
mirrors app.services.appointments.get_pending_visit_confirmations' own idempotent,
self-healing pattern — a missed tick (process restart, deploy) just gets caught by the
very next one instead of silently skipping the reminder, since the underlying condition
("this appointment has now crossed the 60-minute threshold and hasn't been reminded
yet") stays true until it's acted on.
"""
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.doctor import Doctor
from app.models.notification import Notification
from app.models.slot import Slot
from app.models.user import User
from app.services.email import send_appointment_reminder_email
from app.services.notifications import create_notification, format_appointment_datetime

REMINDER_TYPE = "appointment_reminder_60m"
REMINDER_OFFSET_MINUTES = 60


def send_due_reminders_for_clinic(db: Session, clinic_id: uuid.UUID) -> list[Notification]:
    """Finds every 'confirmed' appointment that has now crossed the 60-minute-before
    threshold and doesn't already have a reminder sent for it, and sends one (in-app
    notification + email). Returns every notification just created (not merely a
    count) so callers can log/audit off the same list, same convention as
    get_pending_visit_confirmations.
    """
    now = datetime.now(timezone.utc)
    clinic = db.get(Clinic, clinic_id)
    if clinic is None:
        return []
    clinic_timezone = clinic.timezone

    threshold = now + timedelta(minutes=REMINDER_OFFSET_MINUTES)
    already_reminded = select(Notification.related_appointment_id).where(
        Notification.clinic_id == clinic_id,
        Notification.type == REMINDER_TYPE,
        Notification.related_appointment_id.isnot(None),
    )
    query = (
        select(Appointment)
        .join(Slot, Slot.id == Appointment.slot_id)
        .where(
            Appointment.clinic_id == clinic_id,
            Appointment.status == "confirmed",
            Slot.start_utc <= threshold,
            # Reported live (back when there were multiple windows): booking an
            # appointment less than 60 minutes out fired the 60m reminder
            # immediately, right at booking — the threshold was already in the past
            # the moment the appointment existed. A "60 minutes before" reminder is
            # meaningless for an appointment that was never 60 minutes away at any
            # point after being booked, so only send it if there was actually a real
            # point in time, after booking, when the appointment was still that far out.
            Slot.start_utc >= Appointment.created_at + timedelta(minutes=REMINDER_OFFSET_MINUTES),
            Appointment.id.notin_(already_reminded),
        )
    )
    due = db.execute(query).scalars().all()
    created: list[Notification] = []
    for appointment in due:
        slot = db.get(Slot, appointment.slot_id)
        doctor = db.get(Doctor, appointment.doctor_id)
        doctor_name = doctor.full_name if doctor else "your doctor"
        when_text = format_appointment_datetime(slot.start_utc, clinic_timezone)
        notification = create_notification(
            db,
            clinic_id=clinic_id,
            user_id=appointment.patient_id,
            notif_type=REMINDER_TYPE,
            message=f"Your appointment with {doctor_name} is in 1 hour ({when_text}).",
            related_appointment_id=appointment.id,
        )
        created.append(notification)

        patient = db.get(User, appointment.patient_id)
        if patient is not None:
            send_appointment_reminder_email(
                to=patient.email, full_name=patient.full_name, doctor_name=doctor_name, when_text=when_text
            )
    if created:
        db.flush()
    return created
