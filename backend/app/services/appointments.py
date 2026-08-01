from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.services.notifications import create_notification, format_appointment_datetime


def auto_complete_past_appointments(db: Session, clinic_id) -> list[Appointment]:
    """Moves 'confirmed' appointments whose slot has already ended straight to
    'completed' — no manual admin marking step. Only ever touches rows still in
    'confirmed' status: an appointment already 'cancelled' is never overridden by this,
    regardless of how much time has passed since its slot's end_utc.

    Called lazily at the top of the relevant endpoints AND from the scheduler's backstop
    tick (app.services.scheduler.run_appointment_auto_complete_tick) — an
    'appointment_auto_completed' notification is created here, once per completed
    appointment, so every call site gets one regardless of which one actually catches a
    given appointment first. Returns the appointments that were just completed (not
    merely a count) so callers can log/audit off the same list.
    """
    now = datetime.now(timezone.utc)
    past_slot_ids = select(Slot.id).where(Slot.clinic_id == clinic_id, Slot.end_utc < now)
    to_complete = db.execute(
        select(Appointment).where(
            Appointment.clinic_id == clinic_id,
            Appointment.status == "confirmed",
            Appointment.slot_id.in_(past_slot_ids),
        )
    ).scalars().all()
    if not to_complete:
        return []

    clinic = db.get(Clinic, clinic_id)
    clinic_timezone = clinic.timezone if clinic else "UTC"
    for appointment in to_complete:
        appointment.status = "completed"
        slot = db.get(Slot, appointment.slot_id)
        doctor = db.get(Doctor, appointment.doctor_id)
        create_notification(
            db,
            clinic_id=clinic_id,
            user_id=appointment.patient_id,
            notif_type="appointment_auto_completed",
            message=(
                f"Your appointment with {doctor.full_name if doctor else 'your doctor'} on "
                f"{format_appointment_datetime(slot.start_utc, clinic_timezone)} has been marked as completed."
            ),
            related_appointment_id=appointment.id,
        )
    db.flush()
    return list(to_complete)
