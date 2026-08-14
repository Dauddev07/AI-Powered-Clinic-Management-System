from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.appointment import Appointment
from app.models.slot import Slot


def get_pending_visit_confirmations(db: Session, clinic_id, patient_id) -> list[Appointment]:
    """Appointments still sitting in 'confirmed' whose slot has already ended — the
    patient hasn't yet said whether the visit actually happened. These are no longer
    auto-flipped to 'completed' by the passage of time alone (see
    app/services/booking_engine.py's confirm_visit); a patient must explicitly confirm
    or say they missed it. Oldest-ended first, so a patient who's been away for a
    while works through them in the order they happened.
    """
    now = datetime.now(timezone.utc)
    stmt = (
        select(Appointment)
        .join(Slot, Slot.id == Appointment.slot_id)
        .where(
            Appointment.clinic_id == clinic_id,
            Appointment.patient_id == patient_id,
            Appointment.status == "confirmed",
            Slot.end_utc < now,
        )
        .order_by(Slot.end_utc.asc())
    )
    return list(db.execute(stmt).scalars().all())
