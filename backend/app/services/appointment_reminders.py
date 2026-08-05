"""Sends an in-app reminder notification at exactly 60 minutes, 30 minutes, and 5
minutes before a confirmed appointment's start, plus one when the appointment's own
start time arrives — see app.services.scheduler.run_appointment_reminder_tick for
the job that calls this on a 1-minute interval.

EXACTLY-ONCE PER WINDOW, NOT "AT MOST ONE EVER": each of the four reminder types is
its own independent notification, keyed by (type, related_appointment_id) — an
appointment gets up to 4 reminders total over its lifetime (60m, 30m, 5m, starting),
never more than one of each. A tick is idempotent: it only ever creates a
reminder that doesn't already exist for that (type, appointment) pair, so a delayed
or re-run tick can never double-send. This mirrors
app.services.appointments.auto_complete_past_appointments' own idempotent,
self-healing pattern — a missed tick (process restart, deploy) just gets caught by
the very next one instead of silently skipping that reminder, since the underlying
condition ("this appointment has now crossed this threshold and hasn't been
reminded yet") stays true until it's acted on.

WHY A 1-MINUTE TICK: the reported requirement is "exactly 1h/30m/5m/start", not
"roughly around" — a coarser tick (the 10-30 minute intervals used elsewhere in this
codebase) would let an appointment slip past a threshold between ticks and either
send the reminder late or, worse, skip straight past a 5-minute window that's
already closed by the time the next tick runs. A 1-minute tick keeps the worst-case
lateness to under a minute, which is as close to "exact" as a polling-based
scheduler (rather than one job scheduled per appointment) can get.
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
from app.services.notifications import create_notification, format_appointment_datetime

# (notification type, minutes-before-start it fires at). Order doesn't matter for
# correctness (each is independent), listed furthest-out first for readability.
REMINDER_WINDOWS: tuple[tuple[str, int], ...] = (
    ("appointment_reminder_60m", 60),
    ("appointment_reminder_30m", 30),
    ("appointment_reminder_5m", 5),
    ("appointment_starting", 0),
)


def _reminder_message(notif_type: str, doctor_name: str, when_text: str) -> str:
    if notif_type == "appointment_starting":
        return f"Your appointment with {doctor_name} is starting now ({when_text})."
    minutes = {"appointment_reminder_60m": "1 hour", "appointment_reminder_30m": "30 minutes",
               "appointment_reminder_5m": "5 minutes"}[notif_type]
    return f"Your appointment with {doctor_name} is in {minutes} ({when_text})."


def send_due_reminders_for_clinic(db: Session, clinic_id: uuid.UUID) -> list[Notification]:
    """For each of the 4 reminder windows, finds every 'confirmed' appointment that
    has now crossed that window's threshold (start_utc <= now + offset) and doesn't
    already have that reminder type sent for it, and sends one. Returns every
    notification just created (not merely a count) so callers can log/audit off the
    same list, same convention as auto_complete_past_appointments.
    """
    now = datetime.now(timezone.utc)
    clinic = db.get(Clinic, clinic_id)
    if clinic is None:
        return []
    clinic_timezone = clinic.timezone

    created: list[Notification] = []
    for notif_type, offset_minutes in REMINDER_WINDOWS:
        threshold = now + timedelta(minutes=offset_minutes)
        already_reminded = select(Notification.related_appointment_id).where(
            Notification.clinic_id == clinic_id,
            Notification.type == notif_type,
            Notification.related_appointment_id.isnot(None),
        )
        due = db.execute(
            select(Appointment)
            .join(Slot, Slot.id == Appointment.slot_id)
            .where(
                Appointment.clinic_id == clinic_id,
                Appointment.status == "confirmed",
                Slot.start_utc <= threshold,
                Appointment.id.notin_(already_reminded),
            )
        ).scalars().all()
        for appointment in due:
            slot = db.get(Slot, appointment.slot_id)
            doctor = db.get(Doctor, appointment.doctor_id)
            doctor_name = doctor.full_name if doctor else "your doctor"
            when_text = format_appointment_datetime(slot.start_utc, clinic_timezone)
            notification = create_notification(
                db,
                clinic_id=clinic_id,
                user_id=appointment.patient_id,
                notif_type=notif_type,
                message=_reminder_message(notif_type, doctor_name, when_text),
                related_appointment_id=appointment.id,
            )
            created.append(notification)
    if created:
        db.flush()
    return created
