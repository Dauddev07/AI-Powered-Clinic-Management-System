"""Phase 1 booking engine — the single place every booking rule is enforced.

Extracted from app/api/appointments.py so the REST endpoints AND the chatbot's
function-calling tools (app/services/chat_tools.py) call the exact same functions.
Neither caller re-implements a single rule: slot locking, past-slot rejection,
patient overlap, cancel/reschedule refusal within CANCEL_RESCHEDULE_CUTOFF of the
appointment's start time, ownership, the daily per-department booking AND reschedule
caps, and the transactional all-or-nothing reschedule all live here only. Callers pass
clinic_id/patient_id that they themselves sourced from a verified JWT — this module
trusts whatever is handed to it, same as any other service layer.

Uses HTTPException as the error type even outside a request context: its
status_code/detail pair is exactly the shape both a REST response and a chat tool's
error-to-natural-language translation need, so there's no reason to invent a second
exception type that just wraps the same two fields.
"""
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.appointment import Appointment
from app.models.appointment_department_day_reschedule_use import AppointmentDepartmentDayRescheduleUse
from app.models.appointment_department_day_use import AppointmentDepartmentDayUse
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User
from app.schemas.appointment import AppointmentOut
from app.services.email import (
    send_appointment_booked_email,
    send_appointment_cancelled_email,
    send_appointment_rescheduled_email,
)
from app.services.notifications import create_notification, format_appointment_datetime

DAILY_DEPARTMENT_BOOKING_CAP = 2

# Separate, independent daily allowance for RESCHEDULING an appointment that
# originally sat in a given department/day — not the same counter as the booking
# cap above (a reschedule moves an appointment that already used up one of that
# day's booking slots; capping how many TIMES it can then be moved is its own rule).
DAILY_DEPARTMENT_RESCHEDULE_CAP = 2

# How close to the appointment's own start time a patient can still cancel or
# reschedule it — a flat cutoff by actual time remaining, not by calendar day, so an
# appointment later today booked well outside this window is still changeable, and
# one starting soon (even "tomorrow" by the clock, e.g. 12:30am) is correctly blocked.
CANCEL_RESCHEDULE_CUTOFF = timedelta(hours=2)


def serialize_appointment(db: Session, appointment: Appointment) -> AppointmentOut:
    slot = db.get(Slot, appointment.slot_id)
    doctor = db.get(Doctor, appointment.doctor_id)
    department = db.get(Department, doctor.department_id) if doctor else None
    return AppointmentOut(
        id=appointment.id,
        slot_id=appointment.slot_id,
        doctor_id=appointment.doctor_id,
        doctor_name=doctor.full_name if doctor else "",
        department_name=department.name if department else "",
        start_utc=slot.start_utc,
        end_utc=slot.end_utc,
        status=appointment.status,
        reason=appointment.reason,
        booked_via=appointment.booked_via,
        created_at=appointment.created_at,
        updated_at=appointment.updated_at,
        cancelled_at=appointment.cancelled_at,
    )


def _overlap_exists(db: Session, patient_id, start_utc, end_utc, exclude_appointment_id=None) -> bool:
    stmt = (
        select(Appointment.id)
        .join(Slot, Slot.id == Appointment.slot_id)
        .where(
            Appointment.patient_id == patient_id,
            Appointment.status == "confirmed",
            Slot.start_utc < end_utc,
            Slot.end_utc > start_utc,
        )
    )
    if exclude_appointment_id is not None:
        stmt = stmt.where(Appointment.id != exclude_appointment_id)
    return db.execute(stmt).scalar_one_or_none() is not None


def _department_day_use_count(db: Session, clinic_id, patient_id, department_id, local_date) -> int:
    stmt = select(AppointmentDepartmentDayUse.id).where(
        AppointmentDepartmentDayUse.clinic_id == clinic_id,
        AppointmentDepartmentDayUse.patient_id == patient_id,
        AppointmentDepartmentDayUse.department_id == department_id,
        AppointmentDepartmentDayUse.local_date == local_date,
    )
    return len(db.execute(stmt).all())


def _record_department_day_use(db: Session, clinic_id, patient_id, department_id, appointment_id, local_date) -> None:
    db.add(
        AppointmentDepartmentDayUse(
            clinic_id=clinic_id,
            patient_id=patient_id,
            department_id=department_id,
            appointment_id=appointment_id,
            local_date=local_date,
        )
    )


def _department_day_reschedule_count(db: Session, clinic_id, patient_id, department_id, local_date) -> int:
    stmt = select(AppointmentDepartmentDayRescheduleUse.id).where(
        AppointmentDepartmentDayRescheduleUse.clinic_id == clinic_id,
        AppointmentDepartmentDayRescheduleUse.patient_id == patient_id,
        AppointmentDepartmentDayRescheduleUse.department_id == department_id,
        AppointmentDepartmentDayRescheduleUse.local_date == local_date,
    )
    return len(db.execute(stmt).all())


def _record_department_day_reschedule_use(
    db: Session, clinic_id, patient_id, department_id, appointment_id, local_date
) -> None:
    db.add(
        AppointmentDepartmentDayRescheduleUse(
            clinic_id=clinic_id,
            patient_id=patient_id,
            department_id=department_id,
            appointment_id=appointment_id,
            local_date=local_date,
        )
    )


def book_appointment(
    db: Session,
    clinic_id: uuid.UUID,
    patient_id: uuid.UUID,
    slot_id: uuid.UUID,
    reason: str | None = None,
    booked_via: str = "manual",
) -> Appointment:
    # Row-level lock: a concurrent request for the same slot blocks here until this
    # transaction commits or rolls back, then re-reads the now-committed status —
    # so the loser sees a clean 409, never a double booking.
    slot = db.execute(
        select(Slot).where(Slot.id == slot_id, Slot.clinic_id == clinic_id).with_for_update()
    ).scalar_one_or_none()

    if slot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Slot not found.")

    if slot.status != "open":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This slot is no longer available.")

    now = datetime.now(timezone.utc)
    if slot.start_utc <= now:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Cannot book a slot that has already started."
        )

    if _overlap_exists(db, patient_id, slot.start_utc, slot.end_utc):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You already have an appointment that overlaps with this time.",
        )

    doctor = db.get(Doctor, slot.doctor_id)
    department = db.get(Department, doctor.department_id)
    clinic = db.get(Clinic, clinic_id)
    tz = ZoneInfo(clinic.timezone)
    local_date = slot.start_utc.astimezone(tz).date()

    if _department_day_use_count(db, clinic_id, patient_id, department.id, local_date) >= DAILY_DEPARTMENT_BOOKING_CAP:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"You've reached the limit of {DAILY_DEPARTMENT_BOOKING_CAP} appointments in {department.name} for this day.",
        )

    appointment = Appointment(
        clinic_id=clinic_id,
        slot_id=slot.id,
        patient_id=patient_id,
        doctor_id=slot.doctor_id,
        status="confirmed",
        reason=reason,
        booked_via=booked_via,
    )
    slot.status = "booked"
    db.add(appointment)
    db.flush()
    _record_department_day_use(db, clinic_id, patient_id, department.id, appointment.id, local_date)
    create_notification(
        db,
        clinic_id=clinic_id,
        user_id=patient_id,
        notif_type="appointment_booked",
        message=(
            f"Your appointment with {doctor.full_name} on "
            f"{format_appointment_datetime(slot.start_utc, clinic.timezone)} is confirmed."
        ),
        related_appointment_id=appointment.id,
    )

    try:
        db.commit()
    except IntegrityError:
        # Backstop: the partial unique index (one active appointment per slot) catching
        # a race the row lock somehow didn't — never trust application logic alone.
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This slot is no longer available.")

    db.refresh(appointment)

    # Only sent once the commit above has actually succeeded — an email for a booking
    # that got rolled back (e.g. lost the slot race) would be actively misleading.
    patient = db.get(User, patient_id)
    if patient is not None:
        send_appointment_booked_email(
            to=patient.email,
            full_name=patient.full_name,
            doctor_name=doctor.full_name,
            department_name=department.name,
            when_text=format_appointment_datetime(slot.start_utc, clinic.timezone),
        )

    return appointment


def cancel_appointment(db: Session, clinic_id: uuid.UUID, patient_id: uuid.UUID, appointment_id: uuid.UUID) -> Appointment:
    appointment = db.execute(
        select(Appointment).where(
            Appointment.id == appointment_id,
            Appointment.clinic_id == clinic_id,
            Appointment.patient_id == patient_id,
        )
    ).scalar_one_or_none()
    if appointment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Appointment not found.")

    if appointment.status != "confirmed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This appointment is not active and cannot be cancelled."
        )

    slot = db.get(Slot, appointment.slot_id)
    clinic = db.get(Clinic, clinic_id)

    if slot.start_utc - datetime.now(timezone.utc) <= CANCEL_RESCHEDULE_CUTOFF:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Appointments cannot be cancelled within 2 hours of the appointment time. "
                "Please contact the clinic directly for last-minute changes."
            ),
        )

    appointment.status = "cancelled"
    appointment.cancelled_at = datetime.now(timezone.utc)
    slot.status = "open"
    doctor = db.get(Doctor, appointment.doctor_id)
    create_notification(
        db,
        clinic_id=clinic_id,
        user_id=patient_id,
        notif_type="appointment_cancelled",
        message=(
            f"Your appointment with {doctor.full_name} on "
            f"{format_appointment_datetime(slot.start_utc, clinic.timezone)} has been cancelled."
        ),
        related_appointment_id=appointment.id,
    )
    db.commit()
    db.refresh(appointment)

    patient = db.get(User, patient_id)
    if patient is not None:
        send_appointment_cancelled_email(
            to=patient.email,
            full_name=patient.full_name,
            doctor_name=doctor.full_name,
            when_text=format_appointment_datetime(slot.start_utc, clinic.timezone),
        )

    return appointment


def confirm_visit(
    db: Session, clinic_id: uuid.UUID, patient_id: uuid.UUID, appointment_id: uuid.UUID, completed: bool
) -> Appointment:
    """The patient's own answer to "did this visit happen?" — the only way a
    'confirmed' appointment now leaves that status once its slot has ended (see
    app/services/appointments.py's get_pending_visit_confirmations; nothing auto-flips
    it on a timer anymore). `completed=True` means the visit happened, `False` means
    the patient missed it — mapped to the pre-existing 'no_show' status rather than a
    new one, since it already carries exactly that meaning and needs no schema change.
    """
    appointment = db.execute(
        select(Appointment).where(
            Appointment.id == appointment_id,
            Appointment.clinic_id == clinic_id,
            Appointment.patient_id == patient_id,
        )
    ).scalar_one_or_none()
    if appointment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Appointment not found.")

    if appointment.status != "confirmed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This appointment isn't awaiting confirmation."
        )

    slot = db.get(Slot, appointment.slot_id)
    if slot.end_utc > datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This appointment hasn't ended yet.")

    appointment.status = "completed" if completed else "no_show"
    db.commit()
    db.refresh(appointment)
    return appointment


def reschedule_appointment(
    db: Session,
    clinic_id: uuid.UUID,
    patient_id: uuid.UUID,
    appointment_id: uuid.UUID,
    new_slot_id: uuid.UUID,
) -> Appointment:
    """Releases the old slot and takes the new one in ONE transaction — nothing is
    mutated until every check has passed, and any failure (including a DB-level
    conflict on the new slot) rolls back cleanly, leaving the patient holding their
    original appointment: never zero, never two.
    """
    appointment = db.execute(
        select(Appointment).where(
            Appointment.id == appointment_id,
            Appointment.clinic_id == clinic_id,
            Appointment.patient_id == patient_id,
        )
    ).scalar_one_or_none()
    if appointment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Appointment not found.")
    if appointment.status != "confirmed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This appointment is not active and cannot be rescheduled."
        )
    if new_slot_id == appointment.slot_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="This is already your current slot.")

    old_slot = db.get(Slot, appointment.slot_id)
    clinic = db.get(Clinic, clinic_id)
    tz = ZoneInfo(clinic.timezone)

    if old_slot.start_utc - datetime.now(timezone.utc) <= CANCEL_RESCHEDULE_CUTOFF:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Appointments cannot be rescheduled within 2 hours of the appointment time. "
                "Please contact the clinic directly for last-minute changes."
            ),
        )

    try:
        new_slot = db.execute(
            select(Slot).where(Slot.id == new_slot_id, Slot.clinic_id == clinic_id).with_for_update()
        ).scalar_one_or_none()

        if new_slot is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Slot not found.")
        if new_slot.status != "open":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This slot is no longer available.")

        now = datetime.now(timezone.utc)
        if new_slot.start_utc <= now:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Cannot book a slot that has already started.",
            )

        if _overlap_exists(db, patient_id, new_slot.start_utc, new_slot.end_utc, exclude_appointment_id=appointment.id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="You already have another appointment that overlaps with this time.",
            )

        old_doctor = db.get(Doctor, appointment.doctor_id)
        new_doctor = db.get(Doctor, new_slot.doctor_id)
        new_department = db.get(Department, new_doctor.department_id)
        new_local_date = new_slot.start_utc.astimezone(tz).date()

        if (
            _department_day_use_count(db, clinic_id, patient_id, new_department.id, new_local_date)
            >= DAILY_DEPARTMENT_BOOKING_CAP
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"You've reached the limit of {DAILY_DEPARTMENT_BOOKING_CAP} appointments "
                    f"in {new_department.name} for this day."
                ),
            )

        # Independent reschedule allowance, keyed by the OLD appointment's own
        # department/day (the one being moved AWAY from) — how many times an
        # appointment that was on THAT day/department has already been rescheduled,
        # not how many new bookings exist on the target day.
        old_department = db.get(Department, old_doctor.department_id) if old_doctor is not None else None
        old_local_date = old_slot.start_utc.astimezone(tz).date()
        if old_department is not None and (
            _department_day_reschedule_count(db, clinic_id, patient_id, old_department.id, old_local_date)
            >= DAILY_DEPARTMENT_RESCHEDULE_CAP
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"You've reached the limit of {DAILY_DEPARTMENT_RESCHEDULE_CAP} reschedules "
                    f"for appointments in {old_department.name} on that day."
                ),
            )

        old_slot.status = "open"
        new_slot.status = "booked"
        appointment.slot_id = new_slot.id
        appointment.doctor_id = new_slot.doctor_id
        db.flush()
        _record_department_day_use(db, clinic_id, patient_id, new_department.id, appointment.id, new_local_date)
        if old_department is not None:
            _record_department_day_reschedule_use(
                db, clinic_id, patient_id, old_department.id, appointment.id, old_local_date
            )
        if old_doctor is not None and old_doctor.id != new_doctor.id:
            reschedule_message = (
                f"Your appointment has been rescheduled from {old_doctor.full_name} at "
                f"{format_appointment_datetime(old_slot.start_utc, clinic.timezone)} to "
                f"{new_doctor.full_name} at {format_appointment_datetime(new_slot.start_utc, clinic.timezone)}."
            )
        else:
            reschedule_message = (
                f"Your appointment with {new_doctor.full_name} has been rescheduled to "
                f"{format_appointment_datetime(new_slot.start_utc, clinic.timezone)}."
            )
        create_notification(
            db,
            clinic_id=clinic_id,
            user_id=patient_id,
            notif_type="appointment_rescheduled",
            message=reschedule_message,
            related_appointment_id=appointment.id,
        )

        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This slot is no longer available.")

    db.refresh(appointment)

    patient = db.get(User, patient_id)
    if patient is not None:
        send_appointment_rescheduled_email(
            to=patient.email,
            full_name=patient.full_name,
            old_doctor_name=old_doctor.full_name if old_doctor is not None else "your previous doctor",
            old_when_text=format_appointment_datetime(old_slot.start_utc, clinic.timezone),
            new_doctor_name=new_doctor.full_name,
            new_when_text=format_appointment_datetime(new_slot.start_utc, clinic.timezone),
        )

    return appointment
