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
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot

# Reported live: "only show me available slots of dr farhan rehman after 12 pm
# on monday" and "show me his available slots after 12 pm" both returned the
# same top-5-earliest-of-the-day slots (10:00 am onward) with the time-of-day
# request silently ignored entirely — there was no time-filtering mechanism
# anywhere in this function, only whole-day earliest_date/latest_date bounds.
# When a time-of-day filter is requested, this widens the initial candidate
# pool fetched per doctor (ordered earliest-first, same as always) so there's
# a real chance of finding MAX_SLOTS_PER_DOCTOR genuine matches even after
# most of the earliest candidates get filtered out by the time-of-day check,
# then filters that pool in Python by each slot's LOCAL time-of-day (not UTC —
# a slot at 5pm clinic-local can be a different UTC hour depending on the
# clinic's timezone) and keeps only the first MAX_SLOTS_PER_DOCTOR that match.
_TIME_FILTER_CANDIDATE_POOL = 50

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
    """Exact-first, word-overlap-fallback match against real active doctors.

    A doctor is an EXACT match when EVERY non-trivial word of their full_name is
    present among the patient's typed words — case-insensitive, order-independent, a
    subset check rather than a set-equality check so it still matches when the name is
    embedded in a full sentence (e.g. "I'd like to book with Dr. Ali Raza" still
    exact-matches "Dr. Ali Raza", not just a bare "Ali Raza" query). Whenever at least
    one exact match exists, ONLY exact matches are returned: a patient who typed a
    specific, complete name must resolve directly to that one doctor, not get stuck
    disambiguating against unrelated doctors who merely share a first or last name
    (Babar Ali, Fatima Raza, ...).

    When there is no exact match, falls back to the original broad word-overlap
    behavior — any doctor sharing at least one non-trivial word is surfaced as a
    candidate — since that's still the right behavior for a genuinely partial name
    (just "Ali", or just "Raza") where the patient's input doesn't uniquely identify
    anyone on its own."""
    query_words = _name_words(name_query)
    if not query_words:
        return []

    doctors = db.execute(
        select(Doctor, Department.name)
        .join(Department, Department.id == Doctor.department_id)
        .where(Doctor.clinic_id == clinic_id, Doctor.is_active.is_(True), Department.is_active.is_(True))
    ).all()

    exact_matches = []
    partial_matches = []
    for doctor, department_name in doctors:
        doctor_words = _name_words(doctor.full_name)
        if not doctor_words:
            continue
        match = DoctorMatch(doctor_id=doctor.id, full_name=doctor.full_name, department_name=department_name)
        if doctor_words <= query_words:
            exact_matches.append(match)
        elif query_words & doctor_words:
            partial_matches.append(match)

    matches = exact_matches or partial_matches
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
    doctor_id: uuid.UUID | None = None,
    earliest_time: time | None = None,
    latest_time: time | None = None,
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

    `doctor_id`, when given, narrows the doctors query to that one real doctor
    (still required to belong to this department) — used when a patient has
    already been shown a multi-doctor card and asks to narrow it down to just one
    of them (e.g. "only show me Dr. X's slots"), rather than always returning
    every doctor in the department.

    `earliest_time`/`latest_time`, when given, restrict results to slots whose
    LOCAL (clinic-timezone) time-of-day falls within that range — e.g. "after
    12 pm" (earliest_time=12:00, latest_time=None) or "in the evening"
    (17:00-21:00). Filtered in Python (see _TIME_FILTER_CANDIDATE_POOL above)
    rather than in SQL, since a UTC column can't be range-filtered by local
    time-of-day with a simple comparison.
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

    clinic_tz = None
    if earliest_time is not None or latest_time is not None:
        clinic = db.get(Clinic, clinic_id)
        clinic_tz = ZoneInfo(clinic.timezone if clinic else "UTC")

    doctor_filters = [Doctor.clinic_id == clinic_id, Doctor.department_id == department.id, Doctor.is_active.is_(True)]
    if doctor_id is not None:
        doctor_filters.append(Doctor.id == doctor_id)
    doctors = db.execute(
        select(Doctor).where(*doctor_filters).order_by(Doctor.full_name.asc())
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
        stmt = stmt.order_by(Slot.start_utc.asc())

        if clinic_tz is not None:
            candidates = db.execute(stmt.limit(_TIME_FILTER_CANDIDATE_POOL)).scalars().all()
            slots = []
            for slot in candidates:
                local_time = slot.start_utc.astimezone(clinic_tz).time()
                if earliest_time is not None and local_time < earliest_time:
                    continue
                if latest_time is not None and local_time >= latest_time:
                    continue
                slots.append(slot)
                if len(slots) >= MAX_SLOTS_PER_DOCTOR:
                    break
        else:
            slots = db.execute(stmt.limit(MAX_SLOTS_PER_DOCTOR)).scalars().all()

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
