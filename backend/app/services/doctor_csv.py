"""Doctor CSV parsing and validation.

Shared by the preview and confirm endpoints so both run the exact same checks —
confirm never trusts that a prior preview means the file is still good; it re-parses
and re-validates the freshly uploaded file from scratch every time.

Agreed column list (D2 dependency — the client's real CSV may not have landed yet;
this is built against the agreed columns and must be re-verified the moment it does):
external_doctor_id, name, department, specialty, shift_days, shift_start, shift_end,
slot_length_minutes (optional), leave_dates (optional), active.
"""
import csv
import io
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.appointment import Appointment
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.rag.embeddings import embed_texts
from app.services.slots import _desired_slots_for_doctor

REQUIRED_COLUMNS = [
    "external_doctor_id",
    "name",
    "department",
    "specialty",
    "shift_days",
    "shift_start",
    "shift_end",
    "active",
]
OPTIONAL_COLUMNS = ["slot_length_minutes", "leave_dates"]
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


@dataclass
class HeaderSuggestion:
    original_header: str
    suggested_field: str
    confidence: float


@lru_cache(maxsize=1)
def _canonical_field_embeddings() -> dict[str, list[float]]:
    # Computed once (module-level cache) and reuses the single shared embedding
    # model already loaded at process start — never a second model instance.
    return dict(zip(ALL_COLUMNS, embed_texts(ALL_COLUMNS)))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def suggest_header_mappings(
    fieldnames: list[str],
    confirmed_mapping: dict[str, str],
) -> tuple[list[HeaderSuggestion], list[str]]:
    """Every header that isn't already an exact (case-sensitive) match to a canonical
    column name, and isn't already resolved by an admin-confirmed mapping for this
    exact upload, is compared against every canonical field name using the shared
    embedding model. A header clearing HEADER_MAPPING_SIMILARITY_THRESHOLD produces a
    suggestion the admin must explicitly confirm before it's used for anything;
    nothing below the threshold is ever guessed — it comes back as unrecognized.

    A canonical field already claimed (by an exact-match header, or by another
    header's confirmed mapping) is removed from the candidate pool so two headers
    in the same file can never both be suggested — or confirmed — onto it.
    """
    canonical_vectors = _canonical_field_embeddings()
    claimed_targets = {h for h in fieldnames if h in ALL_COLUMNS} | set(confirmed_mapping.values())
    available_targets = [c for c in ALL_COLUMNS if c not in claimed_targets]

    unresolved = [h for h in fieldnames if h not in ALL_COLUMNS and h not in confirmed_mapping]
    if not unresolved:
        return [], []

    header_vectors = embed_texts(unresolved)
    suggestions: list[HeaderSuggestion] = []
    unrecognized: list[str] = []
    for header, vec in zip(unresolved, header_vectors):
        best_field, best_score = None, -1.0
        for field_name in available_targets:
            score = _cosine_similarity(vec, canonical_vectors[field_name])
            if score > best_score:
                best_field, best_score = field_name, score
        if best_field is not None and best_score >= settings.HEADER_MAPPING_SIMILARITY_THRESHOLD:
            suggestions.append(HeaderSuggestion(header, best_field, round(best_score, 4)))
        else:
            unrecognized.append(header)
    return suggestions, unrecognized

_WEEKDAY_MAP = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}
_TRUE_VALUES = {"true", "1", "yes", "y", "active"}
_FALSE_VALUES = {"false", "0", "no", "n", "inactive"}


@dataclass
class ParsedShift:
    weekday: int
    start_time_utc: time
    end_time_utc: time
    slot_minutes: int


@dataclass
class ParsedDoctorRow:
    row_number: int
    external_doctor_id: str
    full_name: str
    department_name: str
    specialization: str
    is_active: bool
    shifts: list[ParsedShift]
    leave_dates: list[date]
    department_exists: bool


@dataclass
class RejectedRow:
    row_number: int
    external_doctor_id: str | None
    reason: str


@dataclass
class CSVValidationResult:
    accepted: list[ParsedDoctorRow] = field(default_factory=list)
    rejected: list[RejectedRow] = field(default_factory=list)
    # Advisory only — never blocks confirm. Same name under different external_doctor_id
    # values is a real, legitimate possibility (common names exist); this just prompts the
    # admin to double-check before committing rather than silently risking a mix-up.
    warnings: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    # Two headers whose confirmed mapping (or exact match) both point at the same
    # canonical column — resolved defensively rather than letting one silently
    # overwrite the other's values row by row.
    duplicate_mapped_columns: list[str] = field(default_factory=list)
    # Confident synonym suggestions awaiting explicit admin confirmation, and
    # headers with no suggestion clearing the threshold (rejected, never guessed).
    header_suggestions: list[HeaderSuggestion] = field(default_factory=list)
    unrecognized_headers: list[str] = field(default_factory=list)

    @property
    def needs_header_mapping(self) -> bool:
        return bool(self.header_suggestions or self.unrecognized_headers)

    @property
    def is_structurally_valid(self) -> bool:
        return not self.missing_columns and not self.duplicate_mapped_columns


def find_confirmed_appointment_conflicts(
    db: Session,
    clinic_id,
    doctor_id,
    shifts: list[ParsedShift],
    leave_dates: list[date],
    clinic_timezone: str,
) -> list[datetime]:
    """Returns the start_utc of every slot this doctor currently holds a confirmed
    appointment on that would fall OUTSIDE the schedule described by `shifts`/
    `leave_dates` — computed via the exact same diff (_desired_slots_for_doctor)
    the post-confirm slot-regeneration engine uses, so a row is rejected here
    precisely when regeneration would otherwise have to flag one of its slots for
    review after the fact.

    Pass empty `shifts`/`leave_dates` to check against a fully-empty schedule —
    the same "desired = {}" treatment regeneration gives an inactive doctor —
    which is what the CSV-absence auto-deactivation path needs.
    """
    horizon_days = settings.SLOT_GENERATION_HORIZON_DAYS
    today_local = datetime.now(ZoneInfo(clinic_timezone)).date()
    desired = _desired_slots_for_doctor(shifts, set(leave_dates), today_local, horizon_days)

    horizon_end = today_local + timedelta(days=horizon_days)
    window_start_utc = datetime.combine(today_local, time.min, tzinfo=timezone.utc)
    window_end_utc = datetime.combine(horizon_end, time.min, tzinfo=timezone.utc)

    rows = db.execute(
        select(Slot.start_utc)
        .join(Appointment, Appointment.slot_id == Slot.id)
        .where(
            Slot.clinic_id == clinic_id,
            Slot.doctor_id == doctor_id,
            Slot.start_utc >= window_start_utc,
            Slot.start_utc < window_end_utc,
            Appointment.status == "confirmed",
        )
    ).all()

    return sorted(start_utc for (start_utc,) in rows if start_utc not in desired)


def format_conflict_reason(conflicts: list[datetime], clinic_timezone: str) -> str:
    tz = ZoneInfo(clinic_timezone)
    formatted = ", ".join(dt.astimezone(tz).strftime("%Y-%m-%d %H:%M") for dt in conflicts)
    return (
        f"Doctor has confirmed appointment(s) on {formatted} not covered by the new "
        "schedule — resolve manually before re-uploading."
    )


def _parse_weekday_list(raw: str) -> list[int] | None:
    parts = [p.strip().lower() for p in raw.replace(";", ",").split(",") if p.strip()]
    if not parts:
        return None
    days: list[int] = []
    for p in parts:
        if p not in _WEEKDAY_MAP:
            return None
        day = _WEEKDAY_MAP[p]
        if day not in days:
            days.append(day)
    return days


def _parse_time(raw: str) -> time | None:
    raw = raw.strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).time()
        except ValueError:
            continue
    return None


def _local_time_to_utc(t: time, tz_name: str) -> time:
    local_dt = datetime.combine(date(2000, 1, 1), t, tzinfo=ZoneInfo(tz_name))
    return local_dt.astimezone(ZoneInfo("UTC")).time()


def _parse_bool(raw: str) -> bool | None:
    v = raw.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    return None


def _parse_slot_length(raw: str, default_slot_minutes: int) -> tuple[int | None, str | None]:
    raw = raw.strip()
    if not raw:
        return default_slot_minutes, None
    try:
        value = int(raw)
    except ValueError:
        return None, "slot_length_minutes must be a positive integer or blank"
    if value <= 0:
        return None, "slot_length_minutes must be a positive integer or blank"
    return value, None


def _parse_leave_dates(raw: str) -> tuple[list[date], str | None]:
    raw = raw.strip()
    if not raw:
        return [], None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    dates: list[date] = []
    for p in parts:
        try:
            dates.append(datetime.strptime(p, "%Y-%m-%d").date())
        except ValueError:
            return [], f"leave_dates contains an unparseable date: '{p}' (expected YYYY-MM-DD, comma-separated)"
    return dates, None


def validate_csv(
    file_bytes: bytes,
    db: Session,
    clinic_id,
    clinic_timezone: str,
    clinic_default_slot_minutes: int,
    confirmed_header_mapping: dict[str, str] | None = None,
) -> CSVValidationResult:
    result = CSVValidationResult()

    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        result.missing_columns = ALL_COLUMNS
        result.rejected.append(RejectedRow(row_number=0, external_doctor_id=None, reason="File is not valid UTF-8 text"))
        return result

    reader = csv.DictReader(io.StringIO(text))
    fieldnames = [f.strip() for f in (reader.fieldnames or [])]

    # Only trust confirmed-mapping entries that point at a real canonical column —
    # anything else is treated as unconfirmed and goes back through suggestion.
    confirmed_header_mapping = {
        h: v for h, v in (confirmed_header_mapping or {}).items() if v in ALL_COLUMNS
    }
    suggestions, unrecognized = suggest_header_mappings(fieldnames, confirmed_header_mapping)
    if suggestions or unrecognized:
        result.header_suggestions = suggestions
        result.unrecognized_headers = unrecognized
        return result

    effective_mapping = {h: confirmed_header_mapping[h] for h in fieldnames if h in confirmed_header_mapping}
    mapped_fieldnames = [effective_mapping.get(h, h) for h in fieldnames]

    mapped_canonical = [c for c in mapped_fieldnames if c in ALL_COLUMNS]
    duplicates = sorted({c for c in mapped_canonical if mapped_canonical.count(c) > 1})
    if duplicates:
        result.duplicate_mapped_columns = duplicates
        return result

    missing = [c for c in REQUIRED_COLUMNS if c not in mapped_fieldnames]
    if missing:
        result.missing_columns = missing
        return result

    existing_departments = {
        d.name.strip().lower(): d
        for d in db.query(Department).filter(Department.clinic_id == clinic_id).all()
    }
    existing_doctors = {
        d.external_doctor_id: d
        for d in db.query(Doctor).filter(Doctor.clinic_id == clinic_id).all()
    }

    seen_ids: dict[str, list[int]] = {}
    rows = list(reader)

    for idx, row in enumerate(rows, start=2):  # row 1 is the header
        row = {
            effective_mapping.get(k.strip(), k.strip()): (v.strip() if isinstance(v, str) else v)
            for k, v in row.items()
        }
        ext_id = (row.get("external_doctor_id") or "").strip()
        if ext_id:
            seen_ids.setdefault(ext_id, []).append(idx)

    duplicate_ids = {ext_id for ext_id, rows_ in seen_ids.items() if len(rows_) > 1}

    for idx, row in enumerate(rows, start=2):
        row = {
            effective_mapping.get(k.strip(), k.strip()): (v.strip() if isinstance(v, str) else v)
            for k, v in row.items()
        }
        ext_id = (row.get("external_doctor_id") or "").strip()

        if not ext_id:
            result.rejected.append(RejectedRow(idx, None, "external_doctor_id is required and cannot be empty"))
            continue
        if ext_id in duplicate_ids:
            other_rows = [r for r in seen_ids[ext_id] if r != idx]
            result.rejected.append(
                RejectedRow(idx, ext_id, f"duplicate external_doctor_id within file (also on row(s) {other_rows})")
            )
            continue

        name = (row.get("name") or "").strip()
        if not name:
            result.rejected.append(RejectedRow(idx, ext_id, "name is required and cannot be empty"))
            continue

        department_name = (row.get("department") or "").strip()
        if not department_name:
            result.rejected.append(RejectedRow(idx, ext_id, "department is required and cannot be empty"))
            continue

        specialty = (row.get("specialty") or "").strip()

        active_raw = row.get("active") or ""
        is_active = _parse_bool(active_raw)
        if is_active is None:
            result.rejected.append(
                RejectedRow(idx, ext_id, f"active could not be parsed as a boolean: '{active_raw}'")
            )
            continue

        shift_days_raw = row.get("shift_days") or ""
        weekdays = _parse_weekday_list(shift_days_raw)
        if weekdays is None:
            result.rejected.append(
                RejectedRow(idx, ext_id, f"shift_days could not be parsed: '{shift_days_raw}' (expected e.g. 'Mon,Tue,Wed,Thu,Fri')")
            )
            continue

        start_raw = row.get("shift_start") or ""
        end_raw = row.get("shift_end") or ""
        start_local = _parse_time(start_raw)
        end_local = _parse_time(end_raw)
        if start_local is None or end_local is None:
            result.rejected.append(
                RejectedRow(idx, ext_id, f"shift_start/shift_end must be HH:MM (got '{start_raw}' / '{end_raw}')")
            )
            continue
        if start_local >= end_local:
            result.rejected.append(RejectedRow(idx, ext_id, "shift_start must be before shift_end"))
            continue

        slot_minutes, slot_error = _parse_slot_length(row.get("slot_length_minutes") or "", clinic_default_slot_minutes)
        if slot_error:
            result.rejected.append(RejectedRow(idx, ext_id, slot_error))
            continue

        leave_dates, leave_error = _parse_leave_dates(row.get("leave_dates") or "")
        if leave_error:
            result.rejected.append(RejectedRow(idx, ext_id, leave_error))
            continue

        start_utc = _local_time_to_utc(start_local, clinic_timezone)
        end_utc = _local_time_to_utc(end_local, clinic_timezone)
        if start_utc >= end_utc:
            result.rejected.append(
                RejectedRow(idx, ext_id, "shift crosses a UTC day boundary after timezone conversion; adjust the shift times")
            )
            continue

        shifts = [
            ParsedShift(weekday=w, start_time_utc=start_utc, end_time_utc=end_utc, slot_minutes=slot_minutes)
            for w in weekdays
        ]

        existing_doctor = existing_doctors.get(ext_id)
        if existing_doctor is not None:
            # An inactive doctor has no desired slots at all (regeneration's own
            # treatment), regardless of what shifts the row still lists — so a row
            # that flips an existing doctor to inactive is checked the same way.
            effective_shifts = shifts if is_active else []
            conflicts = find_confirmed_appointment_conflicts(
                db, clinic_id, existing_doctor.id, effective_shifts, leave_dates, clinic_timezone
            )
            if conflicts:
                result.rejected.append(
                    RejectedRow(idx, ext_id, format_conflict_reason(conflicts, clinic_timezone))
                )
                continue

        result.accepted.append(
            ParsedDoctorRow(
                row_number=idx,
                external_doctor_id=ext_id,
                full_name=name,
                department_name=department_name,
                specialization=specialty or None,
                is_active=is_active,
                shifts=shifts,
                leave_dates=leave_dates,
                department_exists=department_name.strip().lower() in existing_departments,
            )
        )

    result.warnings = _duplicate_name_warnings(result.accepted)
    return result


def _duplicate_name_warnings(accepted: list[ParsedDoctorRow]) -> list[str]:
    rows_by_name: dict[str, list[ParsedDoctorRow]] = {}
    for row in accepted:
        rows_by_name.setdefault(row.full_name.strip().lower(), []).append(row)

    warnings: list[str] = []
    for rows in rows_by_name.values():
        distinct_ids = sorted({r.external_doctor_id for r in rows})
        if len(distinct_ids) <= 1:
            continue
        display_name = rows[0].full_name
        id_list = ", ".join(distinct_ids)
        warnings.append(
            f"Note: '{display_name}' appears under {len(distinct_ids)} different IDs "
            f"({id_list}) — confirm this is intentional (e.g. genuinely two different "
            f"doctors with the same name) before proceeding."
        )
    return warnings
