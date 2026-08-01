"""Slot generation engine.

Materializes bookable `Slot` rows from each active doctor's `doctor_shifts` minus
`doctor_leave_dates` (both CSV-sourced and admin-blocked), over a rolling horizon
(`settings.SLOT_GENERATION_HORIZON_DAYS`). Regeneration is idempotent — re-running it
(after a CSV re-upload or an admin block-date action) diffs against what already exists
rather than recreating everything, so `UNIQUE(clinic_id, doctor_id, start_utc)` is never
violated and nothing is duplicated.

BOOKING-PRESERVATION DIFF (the highest-consequence rule here): a slot that currently
holds a confirmed appointment is NEVER deleted, blocked, or otherwise mutated by
regeneration, even if the doctor's shift shrank or a leave/block date now covers it.
It is only flagged (`needs_review=True` + an audit_logs entry) for admin review. Free,
unbooked slots that fall outside the new desired set are marked 'blocked' (not deleted —
Postgres would reject a hard delete anyway once any appointment, even a cancelled one,
has ever referenced the row, since appointments.slot_id has no ON DELETE CASCADE).

TIMEZONE NOTE: doctor_shifts.start_time_utc/end_time_utc were produced by the CSV
ingestion pipeline (app/services/doctor_csv.py) by converting the clinic-local wall-clock
time to its UTC equivalent using a fixed reference date. That shortcut is only exact for
a fixed-offset timezone (no DST) — the same assumption this module relies on when it
combines those UTC time-of-day values with each rolling-horizon calendar date directly.
"Today" and each weekday are computed in the clinic's local calendar (via ZoneInfo), not
UTC, since a shift's weekday is a local-calendar concept.

Regeneration is event-triggered only (CSV confirm, admin block-date) — not re-run on
every appointment cancellation, so a just-cancelled slot that falls outside current
availability will self-heal (get blocked) on the *next* regeneration run rather than
immediately. This matches the two triggers named in the spec.
"""
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.appointment import Appointment
from app.models.audit_log import AuditLog
from app.models.clinic import Clinic
from app.models.doctor import Doctor
from app.models.doctor_leave_date import DoctorLeaveDate
from app.models.doctor_shift import DoctorShift
from app.models.slot import Slot


@dataclass
class DoctorRegenerationStats:
    doctor_id: uuid.UUID
    inserted: int = 0
    blocked: int = 0
    unblocked: int = 0
    flagged_for_review: int = 0


@dataclass
class ClinicRegenerationStats:
    doctors_processed: int = 0
    inserted: int = 0
    blocked: int = 0
    unblocked: int = 0
    flagged_for_review: int = 0


def _desired_slots_for_doctor(
    shifts: list[DoctorShift],
    leave_dates: set[date],
    today_local: date,
    horizon_days: int,
) -> dict[datetime, datetime]:
    shifts_by_weekday = {s.weekday: s for s in shifts}
    desired: dict[datetime, datetime] = {}

    for offset in range(horizon_days):
        d = today_local + timedelta(days=offset)
        if d in leave_dates:
            continue
        shift = shifts_by_weekday.get(d.weekday())
        if shift is None:
            continue

        start_dt = datetime.combine(d, shift.start_time_utc, tzinfo=timezone.utc)
        end_dt = datetime.combine(d, shift.end_time_utc, tzinfo=timezone.utc)
        step = timedelta(minutes=shift.slot_minutes)

        cursor = start_dt
        while cursor + step <= end_dt:
            desired[cursor] = cursor + step
            cursor += step

    return desired


def regenerate_slots_for_doctor(
    db: Session,
    clinic_id: uuid.UUID,
    doctor_id: uuid.UUID,
    clinic_timezone: str,
    horizon_days: int | None = None,
    actor_user_id: uuid.UUID | None = None,
) -> DoctorRegenerationStats:
    horizon_days = horizon_days or settings.SLOT_GENERATION_HORIZON_DAYS
    stats = DoctorRegenerationStats(doctor_id=doctor_id)

    today_local = datetime.now(ZoneInfo(clinic_timezone)).date()
    doctor = db.get(Doctor, doctor_id)

    if doctor is not None and doctor.is_active:
        shifts = db.execute(
            select(DoctorShift).where(
                DoctorShift.clinic_id == clinic_id,
                DoctorShift.doctor_id == doctor_id,
                DoctorShift.is_active.is_(True),
            )
        ).scalars().all()
        leave_dates = {
            row.leave_date_utc
            for row in db.execute(
                select(DoctorLeaveDate).where(
                    DoctorLeaveDate.clinic_id == clinic_id, DoctorLeaveDate.doctor_id == doctor_id
                )
            ).scalars().all()
        }
        desired = _desired_slots_for_doctor(shifts, leave_dates, today_local, horizon_days)
    else:
        # Inactive (or missing) doctor: no slots should be desired at all — any of
        # their existing open slots get blocked below, and any booked ones flagged.
        desired = {}

    horizon_end = today_local + timedelta(days=horizon_days)
    window_start_utc = datetime.combine(today_local, time.min, tzinfo=timezone.utc)
    window_end_utc = datetime.combine(horizon_end, time.min, tzinfo=timezone.utc)

    existing_slots = db.execute(
        select(Slot).where(
            Slot.clinic_id == clinic_id,
            Slot.doctor_id == doctor_id,
            Slot.start_utc >= window_start_utc,
            Slot.start_utc < window_end_utc,
        )
    ).scalars().all()
    existing_by_start = {s.start_utc: s for s in existing_slots}

    for start_utc, existing_slot in existing_by_start.items():
        if start_utc in desired:
            if existing_slot.status == "blocked":
                # Shift widened back to cover this slot again; it was blocked, so by
                # definition nothing was booked on it — safe to reopen.
                existing_slot.status = "open"
                stats.unblocked += 1
            if existing_slot.needs_review:
                # Whatever previously made this slot fall outside the desired set
                # (shrunk shift, leave/block date) no longer applies — the flagged
                # conflict is resolved, so clear it rather than leaving a permanent,
                # unclearable false alarm for admin review.
                existing_slot.needs_review = False
            continue

        # No longer desired: shift shrank, or a new leave/admin-block date covers it.
        if existing_slot.status == "booked":
            has_active_appointment = db.execute(
                select(Appointment.id).where(
                    Appointment.slot_id == existing_slot.id, Appointment.status == "confirmed"
                )
            ).scalar_one_or_none()
            if has_active_appointment is not None:
                # BOOKING-PRESERVATION DIFF: never touch a slot with a live appointment.
                if not existing_slot.needs_review:
                    existing_slot.needs_review = True
                    stats.flagged_for_review += 1
                    db.add(
                        AuditLog(
                            clinic_id=clinic_id,
                            actor_user_id=actor_user_id,
                            action="slot_shift_conflict",
                            entity_type="slot",
                            entity_id=existing_slot.id,
                            metadata_json={
                                "doctor_id": str(doctor_id),
                                "slot_start_utc": existing_slot.start_utc.isoformat(),
                                "slot_end_utc": existing_slot.end_utc.isoformat(),
                                "reason": (
                                    "Doctor's shift/availability no longer covers this "
                                    "slot, but it has a confirmed appointment. Left "
                                    "intact for admin review."
                                ),
                            },
                        )
                    )
                continue
            # status == 'booked' with no confirmed appointment shouldn't happen in
            # practice, but fall through to the safe 'blocked' treatment rather than
            # silently deleting.

        if existing_slot.status != "blocked":
            existing_slot.status = "blocked"
            stats.blocked += 1

    for start_utc, end_utc in desired.items():
        if start_utc in existing_by_start:
            continue
        db.add(
            Slot(
                clinic_id=clinic_id,
                doctor_id=doctor_id,
                start_utc=start_utc,
                end_utc=end_utc,
                status="open",
            )
        )
        stats.inserted += 1

    db.flush()
    return stats


def regenerate_slots_for_clinic(
    db: Session, clinic_id: uuid.UUID, actor_user_id: uuid.UUID | None = None
) -> ClinicRegenerationStats:
    clinic = db.get(Clinic, clinic_id)
    doctors = db.execute(select(Doctor).where(Doctor.clinic_id == clinic_id)).scalars().all()

    totals = ClinicRegenerationStats()
    for doctor in doctors:
        stats = regenerate_slots_for_doctor(
            db, clinic_id, doctor.id, clinic.timezone, actor_user_id=actor_user_id
        )
        totals.doctors_processed += 1
        totals.inserted += stats.inserted
        totals.blocked += stats.blocked
        totals.unblocked += stats.unblocked
        totals.flagged_for_review += stats.flagged_for_review

    return totals
