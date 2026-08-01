import uuid
from datetime import date as date_type, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.core.db import get_db
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User
from app.schemas.slot import DepartmentOptionOut, DoctorOptionOut, SlotListOut, SlotOut

router = APIRouter(prefix="/slots", tags=["slots"])

TIME_OF_DAY_BANDS = {
    # Clinic-local hour ranges, mirroring the frontend's own morning/afternoon/evening bands.
    "morning": (0, 11),
    "afternoon": (12, 16),
    "evening": (17, 23),
}


@router.get("", response_model=SlotListOut)
def list_slots(
    department_id: uuid.UUID | None = Query(default=None),
    doctor_id: uuid.UUID | None = Query(default=None),
    date_from: date_type | None = Query(default=None, description="Clinic-local calendar date, inclusive"),
    date_to: date_type | None = Query(default=None, description="Clinic-local calendar date, inclusive"),
    time_of_day: str | None = Query(default=None, pattern="^(morning|afternoon|evening)$"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> SlotListOut:
    """Patients reach doctors through slots, not a directory — so both bookable ('open')
    and already-taken ('booked') future slots are returned (never 'blocked' ones), so
    the patient can see the real shape of a doctor's day rather than a sparse list.

    `date_from`/`date_to` narrow to a clinic-local calendar range, but this is always
    ANDed on top of the `start_utc >= now` clause below, never a replacement for it —
    so "Today" only ever yields slots from the current moment forward, not the whole
    day, even for a request made at the tail end of it.
    """
    clinic_id = current_user.clinic_id
    clinic = db.get(Clinic, clinic_id)
    tz = ZoneInfo(clinic.timezone)
    now = datetime.now(timezone.utc)

    stmt = (
        select(Slot, Doctor, Department)
        .join(Doctor, (Doctor.id == Slot.doctor_id) & (Doctor.clinic_id == clinic_id))
        .join(Department, (Department.id == Doctor.department_id) & (Department.clinic_id == clinic_id))
        .where(
            Slot.clinic_id == clinic_id,
            Slot.status.in_(("open", "booked")),
            Slot.start_utc >= now,
        )
    )
    if department_id is not None:
        stmt = stmt.where(Department.id == department_id)
    if doctor_id is not None:
        stmt = stmt.where(Doctor.id == doctor_id)
    if date_from is not None:
        range_start_utc = datetime.combine(date_from, time.min, tzinfo=tz).astimezone(timezone.utc)
        stmt = stmt.where(Slot.start_utc >= range_start_utc)
    if date_to is not None:
        range_end_utc = datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=tz).astimezone(timezone.utc)
        stmt = stmt.where(Slot.start_utc < range_end_utc)
    if time_of_day is not None:
        local_start = func.timezone(clinic.timezone, Slot.start_utc)
        local_hour = func.extract("hour", local_start)
        band_start, band_end = TIME_OF_DAY_BANDS[time_of_day]
        stmt = stmt.where(local_hour >= band_start, local_hour <= band_end)

    total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()

    stmt = stmt.order_by(Slot.start_utc.asc()).offset(offset).limit(limit)

    rows = db.execute(stmt).all()
    slots = [
        SlotOut(
            id=slot.id,
            doctor_id=doctor.id,
            doctor_name=doctor.full_name,
            department_id=department.id,
            department_name=department.name,
            specialization=doctor.specialization,
            start_utc=slot.start_utc,
            end_utc=slot.end_utc,
            status=slot.status,
            is_bookable=slot.status == "open",
        )
        for slot, doctor, department in rows
    ]

    return SlotListOut(clinic_timezone=clinic.timezone, total=total, limit=limit, offset=offset, slots=slots)


@router.get("/departments", response_model=list[DepartmentOptionOut])
def list_departments_with_slots(
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> list[DepartmentOptionOut]:
    """Derived from live slot data only — every department that currently has at least
    one future open/booked slot. Never a hardcoded department list.
    """
    clinic_id = current_user.clinic_id
    now = datetime.now(timezone.utc)

    stmt = (
        select(Department)
        .join(Doctor, (Doctor.department_id == Department.id) & (Doctor.clinic_id == clinic_id))
        .join(Slot, (Slot.doctor_id == Doctor.id) & (Slot.clinic_id == clinic_id))
        .where(
            Department.clinic_id == clinic_id,
            Slot.status.in_(("open", "booked")),
            Slot.start_utc >= now,
        )
        .distinct()
        .order_by(Department.name.asc())
    )
    departments = db.execute(stmt).scalars().all()
    return [DepartmentOptionOut(id=d.id, name=d.name) for d in departments]


@router.get("/doctors", response_model=list[DoctorOptionOut])
def list_doctors_with_slots(
    department_id: uuid.UUID | None = Query(default=None),
    date_from: date_type | None = Query(default=None, description="Clinic-local calendar date, inclusive"),
    date_to: date_type | None = Query(default=None, description="Clinic-local calendar date, inclusive"),
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> list[DoctorOptionOut]:
    """Every doctor with at least one future bookable ('open') slot matching the
    department/date filters — independent of the slot list's own pagination, so the
    doctor dropdown always offers every eligible doctor, not just the ones on the
    current page of results.
    """
    clinic_id = current_user.clinic_id
    clinic = db.get(Clinic, clinic_id)
    tz = ZoneInfo(clinic.timezone)
    now = datetime.now(timezone.utc)

    stmt = (
        select(Doctor)
        .join(Department, (Department.id == Doctor.department_id) & (Department.clinic_id == clinic_id))
        .join(Slot, (Slot.doctor_id == Doctor.id) & (Slot.clinic_id == clinic_id))
        .where(
            Doctor.clinic_id == clinic_id,
            Slot.status == "open",
            Slot.start_utc >= now,
        )
    )
    if department_id is not None:
        stmt = stmt.where(Department.id == department_id)
    if date_from is not None:
        range_start_utc = datetime.combine(date_from, time.min, tzinfo=tz).astimezone(timezone.utc)
        stmt = stmt.where(Slot.start_utc >= range_start_utc)
    if date_to is not None:
        range_end_utc = datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=tz).astimezone(timezone.utc)
        stmt = stmt.where(Slot.start_utc < range_end_utc)
    stmt = stmt.distinct().order_by(Doctor.full_name.asc())

    doctors = db.execute(stmt).scalars().all()
    return [DoctorOptionOut(id=d.id, name=d.full_name) for d in doctors]
