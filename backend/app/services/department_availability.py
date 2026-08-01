"""Structured-data department/doctor/slot availability — NOT KB text search.

This is what makes "who's available in cardiology" or "list every doctor across
every department" reliably answerable: a real query against doctors/doctor_shifts/
slots rather than a hope that the KB's text search retrieves every relevant chunk.
Shared by:
- symptom triage's department-context injection and doctor-matching step (once a
  department is resolved), and
- the get_department_availability chat tool (direct availability questions).
Always scoped to the caller's clinic via the clinic_id argument (sourced by the
caller from the verified JWT, never from model/user input).
"""
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot

# "dr"/"dr." on its own is never useful as a name-matching word — every doctor's name
# contains it, so treating it as a match signal would make every query "match" every
# doctor.
_NAME_STOPWORDS = frozenset({"dr", "doctor"})

# Free slots surfaced per doctor — enough to guarantee at least 4 genuinely free
# upcoming slots whenever that many exist, while still giving the patient real
# choice without dumping a doctor's entire multi-week open calendar into one turn.
MAX_SLOTS_PER_DOCTOR = 5


@dataclass
class SlotOption:
    slot_id: uuid.UUID
    start_utc: datetime
    end_utc: datetime


@dataclass
class DoctorOption:
    doctor_id: uuid.UUID
    full_name: str
    specialization: str | None
    slots: list[SlotOption]


@dataclass
class DepartmentAvailabilityResult:
    found: bool
    department_name: str | None
    doctors: list[DoctorOption]
    available_department_names: list[str]
    # Only set when a `latest_date` window was requested and it excluded every open
    # slot (doctors ends up empty because of the window, not because there's
    # genuinely nothing open at all) — the earliest real slot that exists beyond the
    # requested window, so the caller can give a grounded "not on that day, but here's
    # when" answer instead of a generic "no slots" message that looks identical to
    # there being no availability at all.
    next_available_when: datetime | None = None


@dataclass
class DoctorMatch:
    doctor_id: uuid.UUID
    full_name: str
    department_name: str


def _name_words(name: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in _NAME_STOPWORDS}


def find_doctors_by_name(db: Session, clinic_id: uuid.UUID, name_query: str) -> list[DoctorMatch]:
    """Word-level match: a real active doctor is returned whenever at least one
    non-trivial word (first name, last name, etc.) of their full_name matches a word
    the patient typed — case-insensitive, order-independent, so "Raza Iqra" still
    matches a real "Dr. Iqra Raza" even though the words are reversed. Deliberately
    broad rather than an exact/fuzzy-distance match: any overlap is surfaced to the
    patient as a candidate to confirm or choose between, rather than the model
    guessing silently or a near-miss going unmatched entirely."""
    query_words = _name_words(name_query)
    if not query_words:
        return []

    doctors = db.execute(
        select(Doctor, Department.name)
        .join(Department, Department.id == Doctor.department_id)
        .where(Doctor.clinic_id == clinic_id, Doctor.is_active.is_(True), Department.is_active.is_(True))
    ).all()

    matches = []
    for doctor, department_name in doctors:
        if query_words & _name_words(doctor.full_name):
            matches.append(DoctorMatch(doctor_id=doctor.id, full_name=doctor.full_name, department_name=department_name))
    matches.sort(key=lambda m: m.full_name)
    return matches


def list_active_department_names(db: Session, clinic_id: uuid.UUID) -> list[str]:
    """The clinic's real, currently-active department names — used both as the
    not-found response's suggestion list below and, directly, as symptom-triage
    context in app.services.chat (replacing medical_kb retrieval, which no longer
    exists: there's no KB text to ground a department-routing decision in, only this
    real structured list)."""
    rows = db.execute(
        select(Department.name)
        .where(Department.clinic_id == clinic_id, Department.is_active.is_(True))
        .order_by(Department.name.asc())
    ).all()
    return [name for (name,) in rows]


def get_department_availability(
    db: Session,
    clinic_id: uuid.UUID,
    department_name: str,
    earliest_date: date | None = None,
    latest_date: date | None = None,
) -> DepartmentAvailabilityResult:
    """Exact (case-insensitive, trimmed) department-name match only — deliberately no
    fuzzy/closest-match guessing, so a typo'd or nonexistent department is reported as
    not found rather than silently routed to the wrong one.

    `earliest_date`, when given, restricts results to slots starting on or after
    midnight UTC of that date — lets a "do you have anything on Friday instead"
    follow-up re-query for genuinely different slots instead of repeating the same
    earliest ones already shown.

    `latest_date`, when given, additionally restricts results to slots starting
    before midnight UTC the day AFTER that date (i.e. it's inclusive of the whole
    day) — this is what makes "is anyone free today or tomorrow" / "on Wednesday"
    answerable precisely instead of always falling back to whatever the earliest
    open slot happens to be, possibly days later. When the window excludes every
    slot but the department does have doctors, a second, unbounded query finds the
    real earliest upcoming slot so the caller can report exactly when that is
    instead of a generic "nothing available" message.
    """
    department = db.execute(
        select(Department).where(
            Department.clinic_id == clinic_id,
            Department.is_active.is_(True),
            Department.name.ilike(department_name.strip()),
        )
    ).scalar_one_or_none()

    if department is None:
        return DepartmentAvailabilityResult(
            found=False,
            department_name=None,
            doctors=[],
            available_department_names=list_active_department_names(db, clinic_id),
        )

    now = datetime.now(timezone.utc)
    earliest = max(now, datetime.combine(earliest_date, time.min, tzinfo=timezone.utc)) if earliest_date else now
    latest = datetime.combine(latest_date, time.min, tzinfo=timezone.utc) + timedelta(days=1) if latest_date else None

    doctors = db.execute(
        select(Doctor)
        .where(Doctor.clinic_id == clinic_id, Doctor.department_id == department.id, Doctor.is_active.is_(True))
        .order_by(Doctor.full_name.asc())
    ).scalars().all()

    doctor_options: list[DoctorOption] = []
    for doctor in doctors:
        stmt = select(Slot).where(
            Slot.clinic_id == clinic_id,
            Slot.doctor_id == doctor.id,
            Slot.status == "open",
            Slot.start_utc > earliest,
        )
        if latest is not None:
            stmt = stmt.where(Slot.start_utc < latest)
        slots = db.execute(stmt.order_by(Slot.start_utc.asc()).limit(MAX_SLOTS_PER_DOCTOR)).scalars().all()

        if not slots:
            continue

        doctor_options.append(
            DoctorOption(
                doctor_id=doctor.id,
                full_name=doctor.full_name,
                specialization=doctor.specialization,
                slots=[SlotOption(slot_id=s.id, start_utc=s.start_utc, end_utc=s.end_utc) for s in slots],
            )
        )

    next_available_when = None
    if latest is not None and not doctor_options and doctors:
        next_slot = db.execute(
            select(Slot)
            .where(
                Slot.clinic_id == clinic_id,
                Slot.doctor_id.in_([d.id for d in doctors]),
                Slot.status == "open",
                Slot.start_utc > earliest,
            )
            .order_by(Slot.start_utc.asc())
            .limit(1)
        ).scalar_one_or_none()
        if next_slot is not None:
            next_available_when = next_slot.start_utc

    return DepartmentAvailabilityResult(
        found=True,
        department_name=department.name,
        doctors=doctor_options,
        available_department_names=[],
        next_available_when=next_available_when,
    )
