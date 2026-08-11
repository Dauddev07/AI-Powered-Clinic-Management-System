import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.core.db import get_db
from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.slot import Slot
from app.models.user import User
from app.schemas.appointment import (
    AppointmentHistoryOut,
    AppointmentListOut,
    AppointmentOut,
    AppointmentSummaryOut,
    BookAppointmentRequest,
    RescheduleRequest,
)
from app.services import booking_engine
from app.services.appointments import auto_complete_past_appointments

router = APIRouter(prefix="/appointments", tags=["appointments"])


@router.post("", response_model=AppointmentOut, status_code=status.HTTP_201_CREATED)
def book_appointment(
    payload: BookAppointmentRequest,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> AppointmentOut:
    appointment = booking_engine.book_appointment(
        db,
        clinic_id=current_user.clinic_id,
        patient_id=current_user.id,
        slot_id=payload.slot_id,
        reason=payload.reason,
        booked_via="manual",
    )
    return booking_engine.serialize_appointment(db, appointment)


@router.get("", response_model=AppointmentListOut)
def list_my_appointments(
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> AppointmentListOut:
    clinic_id = current_user.clinic_id
    auto_complete_past_appointments(db, clinic_id)
    db.commit()

    clinic = db.get(Clinic, clinic_id)

    stmt = (
        select(Appointment)
        .join(Slot, Slot.id == Appointment.slot_id)
        .where(
            Appointment.clinic_id == clinic_id,
            Appointment.patient_id == current_user.id,
            Appointment.status == "confirmed",
        )
        .order_by(Slot.start_utc.asc())
    )
    appointments = db.execute(stmt).scalars().all()
    return AppointmentListOut(
        clinic_timezone=clinic.timezone,
        appointments=[booking_engine.serialize_appointment(db, a) for a in appointments],
    )


@router.get("/history", response_model=AppointmentHistoryOut)
def list_my_appointment_history(
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> AppointmentHistoryOut:
    """Past and cancelled appointments — everything except the still-active
    'confirmed' state — separate from list_my_appointments' upcoming list above.
    Same clinic_id + patient_id ownership scoping as every other appointment
    read here; there is no client-side filtering anywhere in this stack.

    Ordered by whichever actually happened most recently — cancelled_at for a
    cancelled appointment, the slot's own end_utc for a completed one (there's
    no cancelled_at to fall back to) — not by the appointment's original slot
    date. Reported live: sorting by slot date put a just-cancelled appointment
    for a far-future slot above one cancelled weeks ago for an earlier slot,
    which reads wrong for a screen whose whole point is "what happened most
    recently to my appointments."
    """
    clinic_id = current_user.clinic_id
    auto_complete_past_appointments(db, clinic_id)
    db.commit()

    clinic = db.get(Clinic, clinic_id)

    resolved_at = func.coalesce(Appointment.cancelled_at, Slot.end_utc)
    stmt = (
        select(Appointment)
        .join(Slot, Slot.id == Appointment.slot_id)
        .where(
            Appointment.clinic_id == clinic_id,
            Appointment.patient_id == current_user.id,
            Appointment.status != "confirmed",
        )
        .order_by(resolved_at.desc())
    )
    appointments = db.execute(stmt).scalars().all()
    return AppointmentHistoryOut(
        clinic_timezone=clinic.timezone,
        appointments=[booking_engine.serialize_appointment(db, a) for a in appointments],
    )


@router.get("/summary", response_model=AppointmentSummaryOut)
def get_my_appointment_summary(
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> AppointmentSummaryOut:
    """Powers the patient dashboard's next-appointment card and upcoming/completed
    counts. Reuses the same clinic_id + patient_id scoping and 'confirmed = upcoming'
    convention as list_my_appointments above, but counts by status with a single
    grouped aggregate query instead of pulling every completed row just to len() them
    — completed history only grows over a patient's lifetime, so that would get
    more wasteful the longer they've been a patient here.
    """
    clinic_id = current_user.clinic_id
    auto_complete_past_appointments(db, clinic_id)
    db.commit()

    clinic = db.get(Clinic, clinic_id)

    counts = dict(
        db.execute(
            select(Appointment.status, func.count())
            .where(
                Appointment.clinic_id == clinic_id,
                Appointment.patient_id == current_user.id,
                Appointment.status.in_(("confirmed", "completed")),
            )
            .group_by(Appointment.status)
        ).all()
    )

    next_appointment = db.execute(
        select(Appointment)
        .join(Slot, Slot.id == Appointment.slot_id)
        .where(
            Appointment.clinic_id == clinic_id,
            Appointment.patient_id == current_user.id,
            Appointment.status == "confirmed",
        )
        .order_by(Slot.start_utc.asc())
        .limit(1)
    ).scalar_one_or_none()

    return AppointmentSummaryOut(
        clinic_timezone=clinic.timezone,
        next_appointment=booking_engine.serialize_appointment(db, next_appointment) if next_appointment else None,
        upcoming_count=counts.get("confirmed", 0),
        completed_count=counts.get("completed", 0),
    )


@router.post("/{appointment_id}/cancel", response_model=AppointmentOut)
def cancel_appointment(
    appointment_id: uuid.UUID,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> AppointmentOut:
    appointment = booking_engine.cancel_appointment(
        db, clinic_id=current_user.clinic_id, patient_id=current_user.id, appointment_id=appointment_id
    )
    return booking_engine.serialize_appointment(db, appointment)


@router.post("/{appointment_id}/reschedule", response_model=AppointmentOut)
def reschedule_appointment(
    appointment_id: uuid.UUID,
    payload: RescheduleRequest,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> AppointmentOut:
    appointment = booking_engine.reschedule_appointment(
        db,
        clinic_id=current_user.clinic_id,
        patient_id=current_user.id,
        appointment_id=appointment_id,
        new_slot_id=payload.new_slot_id,
    )
    return booking_engine.serialize_appointment(db, appointment)
