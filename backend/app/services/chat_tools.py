"""The five LLM function-calling tools for booking, reschedule, cancel,
appointment-lookup, and availability (SOW task 6.2.4).

Design principles enforced throughout this module:

- THE LLM NEVER DECIDES; IT ONLY CALLS. book_appointment/reschedule_appointment/
  cancel_appointment each call straight into app.services.booking_engine — the exact
  same functions the REST endpoints use — so every booking rule (slot lock, overlap,
  no past slots, no cancel/reschedule within the 2-hour cutoff, ownership,
  transactional reschedule, daily department cap) re-runs server-side exactly once,
  never re-implemented here.
- patient_id/clinic_id are captured from the ClinicContext (itself built only from
  the verified JWT — see app.core.tenancy) at tool-build time via closures. The
  model-facing tool schemas below deliberately have NO patient_id/clinic_id
  parameter, so there is nothing for a model to spoof even if it tried.
- book_appointment/reschedule_appointment/get_department_availability compose their
  OWN final reply text deterministically from real DB rows — the LLM is not asked to
  freehand doctor names, times, or confirmation details into prose, which is exactly
  the kind of hallucination-risk surface this project avoids everywhere else.
- get_my_appointments is the one read-only exception: it returns structured data for
  the calling agent loop to phrase conversationally, per the SOW's explicit "let the
  LLM summarize, never a raw dump" requirement for that one tool only.
"""
import json
import re
import uuid
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from langchain_core.tools import StructuredTool
from langsmith import traceable
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.tenancy import ClinicContext
from app.models.appointment_feedback import AppointmentFeedback
from app.models.clinic import Clinic
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.services import booking_engine
from app.services.chat_markers import BOOKING_MARKER, DEPARTMENT_LIST_MARKER, DOCTOR_OPTIONS_MARKER, NO_SLOTS_MARKER
from app.services.department_availability import find_doctors_by_name, format_department_list, get_department_availability


def _format_when(start_utc: datetime, clinic_tz: str) -> str:
    local = start_utc.astimezone(ZoneInfo(clinic_tz))
    day = f"{local.strftime('%a, %b')} {local.day}"
    hour_12 = local.strftime("%I:%M %p").lstrip("0")
    return f"{day} at {hour_12}"


def _clinic_timezone(db: Session, clinic_id: uuid.UUID) -> str:
    clinic = db.get(Clinic, clinic_id)
    return clinic.timezone if clinic else "UTC"


def _department_name_for_slot(db: Session, clinic_id: uuid.UUID, slot: Slot) -> str | None:
    doctor = db.get(Doctor, slot.doctor_id)
    if doctor is None:
        return None
    from app.models.department import Department

    department = db.get(Department, doctor.department_id)
    return department.name if department else None


def _doctor_ratings(db: Session, doctor_ids: list[uuid.UUID]) -> dict[uuid.UUID, tuple[float, int]]:
    """Same avg/count-over-AppointmentFeedback shape used by the public
    top-rated-doctors endpoint (see api/clinics.py) and the patient slot list
    (see api/slots.py), scoped here to just the doctors already in this
    payload rather than a join, since `availability.doctors` is already a
    small, resolved list by the time this runs. A doctor absent from the
    returned dict simply has no ratings yet.
    """
    if not doctor_ids:
        return {}
    rows = db.execute(
        select(
            AppointmentFeedback.doctor_id,
            func.avg(AppointmentFeedback.rating),
            func.count(AppointmentFeedback.id),
        )
        .where(AppointmentFeedback.doctor_id.in_(doctor_ids))
        .group_by(AppointmentFeedback.doctor_id)
    ).all()
    return {doctor_id: (round(float(avg_rating), 1), rating_count) for doctor_id, avg_rating, rating_count in rows}


def _doctor_options_payload(db: Session, clinic_id: uuid.UUID, availability, note: str | None = None) -> str:
    tz = _clinic_timezone(db, clinic_id)
    ratings = _doctor_ratings(db, [d.doctor_id for d in availability.doctors])
    payload = {
        "note": note,
        "department_name": availability.department_name,
        "doctors": [
            {
                "doctor_id": str(d.doctor_id),
                "doctor_name": d.full_name,
                "specialization": d.specialization,
                "average_rating": ratings.get(d.doctor_id, (None, 0))[0],
                "rating_count": ratings.get(d.doctor_id, (None, 0))[1],
                "slots": [
                    {"slot_id": str(s.slot_id), "when": _format_when(s.start_utc, tz)} for s in d.slots
                ],
            }
            for d in availability.doctors
        ],
    }
    return DOCTOR_OPTIONS_MARKER + json.dumps(payload, default=str)


def _booking_card_payload(db: Session, appointment, note: str | None = None) -> str:
    out = booking_engine.serialize_appointment(db, appointment)
    tz = _clinic_timezone(db, appointment.clinic_id)
    payload = {
        "note": note,
        "doctor_name": out.doctor_name,
        "department_name": out.department_name,
        "when": _format_when(out.start_utc, tz),
    }
    return BOOKING_MARKER + json.dumps(payload, default=str)


def _no_slots_message(department_name: str) -> str:
    return (
        f"I couldn't find any doctors with free upcoming slots in {department_name} right now. "
        "Please check back later or contact the clinic directly."
    )


def _no_slots_in_window_message(db: Session, clinic_id: uuid.UUID, department_name: str, next_available_when) -> str:
    if next_available_when is None:
        return _no_slots_message(department_name)
    tz = _clinic_timezone(db, clinic_id)
    return (
        f"No one in {department_name} has a free slot in that window. "
        f"The earliest open slot I can find is {_format_when(next_available_when, tz)}. "
        "Would you like me to book that instead, or check a different day?"
    )


def _no_slots_payload(department_name: str, message: str) -> str:
    """Tags a "department is real but nothing's free" message with NO_SLOTS_MARKER so
    combine_department_availability_results can tell it apart from a plain "not
    found" string instead of silently dropping it — see NO_SLOTS_MARKER's docstring.
    _finalize_reply strips this marker back off for the single-call case, so a lone
    get_department_availability call still reads exactly as before."""
    return NO_SLOTS_MARKER + json.dumps({"department_name": department_name, "message": message})


def _department_not_found_message(availability) -> str:
    if availability.available_department_names:
        names = format_department_list(availability.available_department_names)
        return f"I couldn't find a department called that. The departments we have are:\n\n{names}"
    return "I couldn't find a matching department, and no departments are currently configured."


_WEEKDAY_INDEX = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

# Full weekday names are unambiguous anywhere in the message. Abbreviations
# ("sun", "sat", "mon"...) collide with ordinary English words ("the sun",
# "I sat down"), so those are only matched right after a preposition that's
# actually date-shaped ("on sun", "for sat", "by mon") — narrow enough to
# avoid a false positive while still covering how patients actually type a
# bare day-of-week reference.
_FULL_WEEKDAY_RE = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)
_ABBR_WEEKDAY_TOKEN_RE = re.compile(
    r"\b(?:on|for|this|next|by|until|til)\s+([a-z]{3,9})\b",
    re.IGNORECASE,
)
# Reported live: "sat 1045", "sat 10.45", "sat 10:45" (a fragmented reply, no
# "on"/"for"/etc. word at all before the abbreviation) matched neither the
# full-weekday regex above nor the trigger-word-gated abbreviation regex, so
# the weekday was silently dropped entirely. An abbreviation immediately
# followed by a clock-shaped token is unambiguous enough to trust on its own
# even with no preceding trigger word — a bare "sat" with nothing after it
# stays gated behind the trigger-word requirement above, since standalone it's
# also a real English word ("I sat down").
_ABBR_WEEKDAY_BEFORE_TIME_RE = re.compile(
    r"\b([a-z]{3,9})\b(?=\s*\d{1,2}(?:[:.]?\d{2})?\s*(?:am|pm)?\b)",
    re.IGNORECASE,
)
_WEEKDAY_ABBR_CANDIDATES = ("mon", "tue", "tues", "wed", "thu", "thurs", "fri", "sat", "sun")


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def _closest_weekday_abbr(token: str) -> str | None:
    """Reported live: "show me her available slots on thus" (a typo for
    "thurs") silently matched no weekday at all and got treated as an
    unrelated turn — the exact-string abbreviation match had zero typo
    tolerance. Accepts a single-character-edit typo of a real weekday
    abbreviation (e.g. "thus" -> "thu"/"thurs", "wedd" -> "wed") so a patient's
    near-miss spelling still resolves instead of being silently dropped;
    anything further off (genuinely unrelated words like "cardiology") stays
    unmatched."""
    token = token.lower()
    if token in _WEEKDAY_INDEX:
        return token
    if not (3 <= len(token) <= 7):
        return None
    best_abbr, best_dist = None, 2
    for abbr in _WEEKDAY_ABBR_CANDIDATES:
        dist = _levenshtein(token, abbr)
        if dist < best_dist:
            best_abbr, best_dist = abbr, dist
    return best_abbr


def resolve_bare_weekday_window(message: str) -> tuple[str, str] | None:
    """Reported live: "book a slot on sun" got silently answered with Friday's
    slots instead of "nothing open on Sunday, the earliest is Friday" — the system
    prompt already instructs the model to resolve a bare weekday to a real date and
    set earliest_date/latest_date, but it simply didn't call the tool with those
    arguments set, the same "prompt instruction alone isn't a reliable guarantee"
    gap already fixed elsewhere in this codebase for other tool-calling steps. This
    is the deterministic backstop: resolve the weekday actually named in the
    patient's own message, in code, to the next real occurrence of that weekday
    on/after today (UTC, matching app.services.llm._current_date_str) — the caller
    forces this onto the get_department_availability call regardless of what the
    model passed. Returns None when the message names no single weekday (including
    when it names more than one, e.g. a real date range, which this deliberately
    doesn't try to resolve) so an unrelated turn is left completely untouched.
    Also tolerates a single-character typo in an abbreviation (see
    _closest_weekday_abbr) so "on thus" still resolves to Thursday.
    """
    matches = _FULL_WEEKDAY_RE.findall(message)
    if not matches:
        abbr_tokens = _ABBR_WEEKDAY_TOKEN_RE.findall(message)
        matches = [m for m in (_closest_weekday_abbr(t) for t in abbr_tokens) if m]
    if not matches:
        abbr_tokens = _ABBR_WEEKDAY_BEFORE_TIME_RE.findall(message)
        matches = [m for m in (_closest_weekday_abbr(t) for t in abbr_tokens) if m]
    if len(matches) != 1:
        return None
    return _weekday_name_to_next_occurrence_window(matches[0])


def _weekday_name_to_next_occurrence_window(weekday_name: str) -> tuple[str, str]:
    target_weekday = _WEEKDAY_INDEX[weekday_name.lower()]
    today = datetime.now(timezone.utc).date()
    delta_days = (target_weekday - today.weekday()) % 7
    resolved = today + timedelta(days=delta_days)
    iso = resolved.isoformat()
    return iso, iso


def resolve_bare_weekday_reply(message: str) -> tuple[str, str] | None:
    """Like resolve_bare_weekday_window, but ALSO accepts a bare weekday name/
    abbreviation with no surrounding trigger word or adjacent time token at
    all ("sat" on its own) — only safe to use where the caller already knows
    a day-only answer is expected (e.g. answering this module's own "which
    day would you like?" clarifying question), since that context removes the
    "sat" vs "I sat down" ambiguity resolve_bare_weekday_window's stricter
    gating exists for everywhere else."""
    window = resolve_bare_weekday_window(message)
    if window is not None:
        return window
    token = message.strip().lower()
    if not token.isalpha():
        return None
    resolved_name = _closest_weekday_abbr(token)
    if resolved_name is None:
        return None
    return _weekday_name_to_next_occurrence_window(resolved_name)


_MONTH_NAME_TO_NUMBER = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}
# Longest names first so e.g. "september" doesn't get cut short by "sep" matching
# a prefix of it first in the alternation.
_MONTH_NAME_ALTERNATION = "|".join(sorted(_MONTH_NAME_TO_NUMBER, key=len, reverse=True))
_DAY_THEN_MONTH_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAME_ALTERNATION})\b", re.IGNORECASE)
_MONTH_THEN_DAY_RE = re.compile(rf"\b({_MONTH_NAME_ALTERNATION})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.IGNORECASE)


def resolve_explicit_calendar_date(message: str) -> str | None:
    """Resolves an explicit "26 aug"/"aug 26"-style calendar date the patient
    named directly in their own message to a real ISO date, or None if the
    message names no such date (including an ambiguous case naming more than
    one, which this deliberately doesn't try to resolve — same "leave an
    unrelated/ambiguous turn completely untouched" principle as
    resolve_bare_weekday_window above, which this is the day+month counterpart
    to: a weekday name attached to an explicit date, e.g. "wed 26 aug", falls
    entirely outside that function's own bare-weekday-only scope).

    Assumes the current year, rolling forward to next year only when the
    resulting date has already passed — mirrors resolve_bare_weekday_window's
    own forward-roll for a bare weekday name.
    """
    # Both regexes' captures are normalized to (day_str, month_name) order here,
    # regardless of which order the patient actually typed them in.
    day_month_matches = _DAY_THEN_MONTH_RE.findall(message)
    month_day_matches = [(day, month) for month, day in _MONTH_THEN_DAY_RE.findall(message)]
    all_matches = day_month_matches + month_day_matches
    if len(all_matches) != 1:
        return None
    day_str, month_name = all_matches[0]
    month = _MONTH_NAME_TO_NUMBER[month_name.lower()]
    day = int(day_str)
    today = datetime.now(timezone.utc).date()
    try:
        candidate = date(today.year, month, day)
    except ValueError:
        return None
    if candidate < today:
        try:
            candidate = date(today.year + 1, month, day)
        except ValueError:
            return None
    return candidate.isoformat()


# Reported live: "mon aug 31st" (a weekday name attached to an explicit calendar
# date) got resolved to the nearest upcoming Monday from TODAY, silently ignoring
# "aug 31" entirely — several call sites called resolve_bare_weekday_window
# directly on its own, on the assumption (stated in that function's own docstring
# and _resolve_date_and_time_window's below) that a bare weekday name and an
# explicit date are "mutually exclusive phrasings" for the same message. That
# assumption is false: resolve_bare_weekday_window has no awareness of a
# following month/day at all, so it happily matches "mon" even when "aug 31"
# is sitting right next to it. This is the single shared, correctly-ordered
# resolver every date-resolving call site should use instead of calling
# resolve_bare_weekday_window on its own — an explicit calendar date always
# wins over a bare weekday name when a message somehow contains both.
def resolve_date_window(message: str) -> tuple[str, str] | None:
    explicit_date = resolve_explicit_calendar_date(message)
    if explicit_date is not None:
        return explicit_date, explicit_date
    return resolve_bare_weekday_window(message)


# Reported live: "only show me available slots of dr farhan rehman after 12 pm
# on monday" and its follow-up "show me his available slots after 12 pm" both
# silently ignored the time-of-day request and returned the same top-5-
# earliest-of-the-day slots — there was no time-of-day filtering mechanism
# anywhere at all, only whole-day earliest_date/latest_date bounds. This is the
# deterministic counterpart to resolve_bare_weekday_window above, for the
# time-of-day half of the same class of request: an explicit clock time
# ("after 12 pm", "before 5:30pm") or a named part of the day ("in the
# evening", "in the morning"). Named phrases are checked first since they're
# unambiguous; a bare clock time is only trusted as a filter when paired with
# "after"/"before" (a clock time appearing for some other reason in the
# message, with neither word, is left alone rather than guessed at).
_TIME_OF_DAY_PHRASES: tuple[tuple[str, time, time], ...] = (
    ("morning", time(0, 0), time(12, 0)),
    ("afternoon", time(12, 0), time(17, 0)),
    ("evening", time(17, 0), time(21, 0)),
    ("night", time(21, 0), time(23, 59, 59)),
)

_AFTER_CLOCK_RE = re.compile(r"\bafter\s+(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_BEFORE_CLOCK_RE = re.compile(r"\bbefore\s+(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)\b", re.IGNORECASE)

# Reported live: "book me with dr waqas on mon aug 24 at 3.30 instead" and its
# follow-up "at 3.30" were both silently ignored by resolve_time_of_day_window —
# only "after"/"before" were recognized, so a bare "at X" request fell through
# with no time bound at all and the caller got the day's earliest slots instead.
# "at X" is treated the same as "after X" (an inclusive lower bound rather than
# an exact-only match) since the caller always surfaces a short list of
# candidate slots rather than a single exact one, so "at 3:30" naturally reads
# as "3:30 or the nearest slots after it."
_AT_CLOCK_RE = re.compile(r"\bat\s+(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)

# Reported live: "sat 10.45", "sat 1045", "sat 10:45" (a fragmented reply to a
# "which doctor" clarifying question, no "at"/"after"/"before" word at all)
# were all silently ignored — every clock-time pattern above requires one of
# those trigger words first. A bare clock time with a colon/dot separator is
# unambiguous enough to trust on its own (same "at X" semantics: an inclusive
# lower bound, not an exact-only match).
_BARE_SEPARATOR_CLOCK_RE = re.compile(r"\b(\d{1,2})[:.](\d{2})\s*(am|pm)?\b", re.IGNORECASE)

# Companion to the separator case above: "1045"/"945" with NO separator at
# all. Ambiguous on its own (could be a 3-4 digit ID/quantity/year), so this is
# only trusted once _parse_bare_digit_clock validates it as a plausible time
# (hour 1-12, minute 00-59) — e.g. "2024" (a year) yields hour=20, rejected;
# "1045" yields hour=10/minute=45, accepted.
_BARE_DIGIT_CLOCK_RE = re.compile(r"\b(\d{3,4})\s*(am|pm)?\b", re.IGNORECASE)

# Reported live: "mon 12 pm" (an answer to a "what day would you like?"
# question — the day resolved fine, but the time silently didn't) and "book at
# mon 12 pm" (the "at" here modifies "mon", not "12 pm", so _AT_CLOCK_RE never
# matches at all — it requires "at" immediately before the number) both went
# completely unrecognized as a time. A bare hour with an explicit am/pm
# marker directly after it, no colon/dot/minutes needed, is unambiguous
# enough to trust on its own — an accidental "12 pm" appearing for some other
# reason is vanishingly rare, unlike a bare digit run (see
# _BARE_DIGIT_CLOCK_RE's own comment on why THAT needs validation first).
_BARE_HOUR_WITH_MERIDIEM_RE = re.compile(r"\b(\d{1,2})\s*(am|pm)\b", re.IGNORECASE)


def _parse_bare_digit_clock(digits: str) -> tuple[str, str] | None:
    """'1045' -> ('10', '45'); '945' -> ('9', '45'). None if the split isn't a
    plausible 12-hour clock time at all (hour 1-12, minute 00-59) — guards
    against misreading an unrelated 3-4 digit number as a time."""
    hour_str, minute_str = (digits[:2], digits[2:]) if len(digits) == 4 else (digits[:1], digits[1:])
    hour, minute = int(hour_str), int(minute_str)
    if not (1 <= hour <= 12) or not (0 <= minute <= 59):
        return None
    return hour_str, minute_str


def _infer_meridiem(hour: int) -> str:
    """Best-effort default for "at X" when no am/pm was said at all — a bare
    hour in a same-day scheduling chat is read the way a patient means it: 1-7
    as afternoon/evening, 12 as noon, 8-11 as morning. "after"/"before" never
    hit this path since they always require an explicit am/pm."""
    return "pm" if hour == 12 or 1 <= hour <= 7 else "am"


def _parse_12_hour_clock(hour_str: str, minute_str: str | None, meridiem: str | None) -> time:
    hour_value = int(hour_str)
    resolved_meridiem = (meridiem or _infer_meridiem(hour_value)).lower()
    hour = hour_value % 12
    if resolved_meridiem == "pm":
        hour += 12
    return time(hour, int(minute_str or 0))


def resolve_time_of_day_window(message: str) -> tuple[time | None, time | None] | None:
    """Returns (earliest_time, latest_time) — either may be None — in the
    patient's own wording, or None when the message names no time-of-day
    constraint at all. The caller resolves these against the CLINIC's local
    timezone (see department_availability.get_department_availability's
    earliest_time/latest_time), never UTC directly, since "12 pm" means
    whatever local noon is for that clinic."""
    after_match = _AFTER_CLOCK_RE.search(message)
    before_match = _BEFORE_CLOCK_RE.search(message)
    if after_match or before_match:
        earliest = _parse_12_hour_clock(*after_match.groups()) if after_match else None
        latest = _parse_12_hour_clock(*before_match.groups()) if before_match else None
        return (earliest, latest)

    at_match = _AT_CLOCK_RE.search(message)
    if at_match:
        return (_parse_12_hour_clock(*at_match.groups()), None)

    # Bare clock time, no "at"/"after"/"before" word at all — see
    # _BARE_SEPARATOR_CLOCK_RE/_BARE_DIGIT_CLOCK_RE's own comments. Separator
    # form checked first since it's unambiguous; the no-separator form only
    # engages once digit-split validation confirms it's plausibly a time.
    bare_sep_match = _BARE_SEPARATOR_CLOCK_RE.search(message)
    if bare_sep_match:
        return (_parse_12_hour_clock(*bare_sep_match.groups()), None)
    bare_digit_match = _BARE_DIGIT_CLOCK_RE.search(message)
    if bare_digit_match:
        split = _parse_bare_digit_clock(bare_digit_match.group(1))
        if split is not None:
            return (_parse_12_hour_clock(split[0], split[1], bare_digit_match.group(2)), None)
    # Checked after the digit-clock cases above so a longer run like "1045 pm"
    # is still matched as 10:45 by _BARE_DIGIT_CLOCK_RE first, rather than
    # this pattern grabbing just the trailing "45 pm" on its own.
    bare_hour_meridiem_match = _BARE_HOUR_WITH_MERIDIEM_RE.search(message)
    if bare_hour_meridiem_match:
        return (_parse_12_hour_clock(bare_hour_meridiem_match.group(1), None, bare_hour_meridiem_match.group(2)), None)

    lowered = message.lower()
    for phrase, start, end in _TIME_OF_DAY_PHRASES:
        if re.search(rf"\b{phrase}\b", lowered):
            return (start, end)

    return None


def _parse_date_arg(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        # An unparseable date is treated as "no filter" rather than an error — the
        # tool still returns real, current availability instead of failing the turn
        # over a malformed argument the model produced.
        return None


def combine_department_availability_results(raw_results: list[str]) -> str:
    """Assembles a DEPARTMENT_LIST:: payload from multiple get_department_availability
    calls made within one agent turn — built entirely in code from each call's own
    real result, never handed to the LLM to summarize itself (that's how a raw
    slot_id previously leaked into freehand prose).

    A "department not found" result (bad/invented name) is skipped rather than
    fabricated into a fake department entry — but a NO_SLOTS_MARKER result (a REAL
    department with nobody free right now) is kept and surfaced under
    "unavailable", not dropped. Silently skipping both alike used to mean a
    multi-department turn (e.g. neck pain + itchy skin) would show a card for
    Orthopedics but say nothing at all about Dermatology, even though the model's
    own note named both — reported live. The patient now sees why the department
    they were told about isn't in the card, instead of it just vanishing.

    Deduplicated by department_name, first-call-wins: the model can call the tool
    more than once for the SAME department in one turn (e.g. confused by a typo'd day
    name and retrying), and without this the same department/doctor/slot list would
    render twice in one card — reported live as "Cardiology" appearing back-to-back
    with identical doctors and slots.

    Each department entry also carries its own `note` (omitted/None when that call
    didn't have one) — reported live: a symptom-based multi-department reply (nose
    pain + skin issues -> ENT + Dermatology) showed both departments' doctors but
    explained the reasoning for NEITHER, even though the single-department card
    (DOCTOR_OPTIONS_MARKER) has always shown its note. This was silently dropped
    here, not a deliberate omission — the patient could see the doctors but never
    why they were routed there. A department reached via a direct request naming
    it (e.g. "book me with a cardiologist") never gets a note in the first place —
    see the TOOL USE RULES' "Omit `note` entirely" rule — so this doesn't
    resurface reasoning for a case that never had any."""
    departments = []
    unavailable = []
    seen_department_names = set()
    for raw in raw_results:
        if raw.startswith(DOCTOR_OPTIONS_MARKER):
            payload = json.loads(raw[len(DOCTOR_OPTIONS_MARKER):])
            department_name = payload["department_name"]
            if department_name in seen_department_names:
                continue
            seen_department_names.add(department_name)
            departments.append(
                {"department_name": department_name, "note": payload.get("note"), "doctors": payload["doctors"]}
            )
        elif raw.startswith(NO_SLOTS_MARKER):
            payload = json.loads(raw[len(NO_SLOTS_MARKER):])
            department_name = payload["department_name"]
            if department_name in seen_department_names:
                continue
            seen_department_names.add(department_name)
            unavailable.append({"department_name": department_name, "message": payload["message"]})
        # else: a "department not found" string — skipped, not fabricated into an entry.

    if not departments and not unavailable:
        return (
            "I couldn't find any departments with doctors who have free upcoming slots right now. "
            "Please check back later or contact the clinic directly."
        )
    if not departments:
        return " ".join(entry["message"] for entry in unavailable)

    return DEPARTMENT_LIST_MARKER + json.dumps({"departments": departments, "unavailable": unavailable}, default=str)


# ---------------------------------------------------------------------------
# Tool implementations (each @traceable so it appears as its own step in the
# LangSmith trace tree, alongside retrieval and LLM call steps).
# ---------------------------------------------------------------------------


@traceable(name="book_appointment")
def _book_appointment_impl(db: Session, ctx: ClinicContext, slot_id: str, reason: str | None) -> str:
    try:
        parsed_slot_id = uuid.UUID(slot_id)
    except (ValueError, TypeError, AttributeError):
        return "I couldn't recognize that slot. Could you pick one of the listed options again?"

    slot_before = db.get(Slot, parsed_slot_id)

    try:
        appointment = booking_engine.book_appointment(
            db,
            clinic_id=ctx.clinic_id,
            patient_id=ctx.user_id,
            slot_id=parsed_slot_id,
            reason=reason,
            booked_via="chatbot",
        )
    except HTTPException as exc:
        # Only "This slot is no longer available." is a genuine lost-race case (the
        # slot was taken between offer and confirmation) — matched on the exact
        # detail text, not just "any 409", because book_appointment also raises a
        # 409 when the slot the patient picked overlaps one of THEIR OWN existing
        # appointments (self-overlap, nothing to do with another patient). Treating
        # every 409 as a lost race used to swallow that overlap detail entirely and
        # show "that slot was just taken by another patient" instead — wrong and
        # misleading when the real reason was the patient's own double-booking.
        if exc.status_code == 409 and slot_before is not None and exc.detail == "This slot is no longer available.":
            # Lost race: the slot was taken between offer and confirmation. Never
            # report a booking that didn't happen — apologise and hand back fresh,
            # genuinely free alternatives in the same department, in the same turn.
            department_name = _department_name_for_slot(db, ctx.clinic_id, slot_before)
            if department_name:
                availability = get_department_availability(db, ctx.clinic_id, department_name)
                if availability.found and availability.doctors:
                    return _doctor_options_payload(
                        db,
                        ctx.clinic_id,
                        availability,
                        note="Sorry — that slot was just taken by another patient. Here are fresh options:",
                    )
                return "Sorry — that slot was just taken by another patient, and " + _no_slots_message(department_name)
        return exc.detail

    return _booking_card_payload(db, appointment)


@traceable(name="reschedule_appointment")
def _reschedule_appointment_impl(db: Session, ctx: ClinicContext, appointment_id: str, new_slot_id: str) -> str:
    try:
        parsed_appointment_id = uuid.UUID(appointment_id)
        parsed_new_slot_id = uuid.UUID(new_slot_id)
    except (ValueError, TypeError, AttributeError):
        return "I couldn't recognize that appointment or slot. Could you try again?"

    new_slot_before = db.get(Slot, parsed_new_slot_id)

    try:
        appointment = booking_engine.reschedule_appointment(
            db,
            clinic_id=ctx.clinic_id,
            patient_id=ctx.user_id,
            appointment_id=parsed_appointment_id,
            new_slot_id=parsed_new_slot_id,
        )
    except HTTPException as exc:
        if exc.status_code == 409 and new_slot_before is not None and exc.detail == "This slot is no longer available.":
            department_name = _department_name_for_slot(db, ctx.clinic_id, new_slot_before)
            if department_name:
                availability = get_department_availability(db, ctx.clinic_id, department_name)
                if availability.found and availability.doctors:
                    return _doctor_options_payload(
                        db,
                        ctx.clinic_id,
                        availability,
                        note="Sorry — that slot was just taken by another patient. Here are fresh options:",
                    )
                return "Sorry — that slot was just taken by another patient, and " + _no_slots_message(department_name)
        return exc.detail

    return _booking_card_payload(db, appointment, note="Your appointment has been rescheduled.")


@traceable(name="cancel_appointment")
def _cancel_appointment_impl(db: Session, ctx: ClinicContext, appointment_id: str) -> str:
    try:
        parsed_appointment_id = uuid.UUID(appointment_id)
    except (ValueError, TypeError, AttributeError):
        return "I couldn't recognize that appointment. Could you try again?"

    try:
        appointment = booking_engine.cancel_appointment(
            db, clinic_id=ctx.clinic_id, patient_id=ctx.user_id, appointment_id=parsed_appointment_id
        )
    except HTTPException as exc:
        # Verbatim pass-through, including the 2-hour-cutoff refusal — must read
        # identically to what the REST /appointments/{id}/cancel endpoint returns.
        return exc.detail

    out = booking_engine.serialize_appointment(db, appointment)
    tz = _clinic_timezone(db, ctx.clinic_id)
    return f"Your appointment with {out.doctor_name} in {out.department_name} on {_format_when(out.start_utc, tz)} has been cancelled."


def cancel_appointment_now(db: Session, ctx: ClinicContext, appointment_id: str) -> str:
    """Public entry point for appointment_agent's deterministic cancel-confirmation
    handoff (see app.services.orchestrator.agents.appointment_agent) — cancels
    immediately once the patient has explicitly confirmed, without going through the
    LLM tool-calling loop at all. Reported live: "can I cancel that?" (a question,
    not a command) got treated identically to an imperative "cancel it" and the
    appointment was cancelled with zero confirmation step. Mirrors the
    never-let-the-model-freehand-a-mutating-action principle used elsewhere in this
    module (e.g. reschedule/cancel redirect ids in build_tools) — a destructive
    action gets a real, code-enforced confirm step, not just a prompt instruction
    the model might skip.
    """
    return _cancel_appointment_impl(db, ctx, appointment_id)


def book_appointment_now(db: Session, ctx: ClinicContext, slot_id: str, reason: str | None = None) -> str:
    """Public entry point for appointment_agent's deterministic book-confirmation
    handoff — books immediately once the patient has explicitly confirmed a specific
    slot, without going through the LLM tool-calling loop at all. Same
    never-let-the-model-freehand-a-mutating-action reasoning as cancel_appointment_now
    above, applied to booking instead of cancelling.
    """
    return _book_appointment_impl(db, ctx, slot_id, reason)


def reschedule_appointment_now(db: Session, ctx: ClinicContext, appointment_id: str, new_slot_id: str) -> str:
    """Public entry point for appointment_agent's deterministic reschedule-
    confirmation handoff — reschedules immediately once the patient has explicitly
    confirmed the new slot, without going through the LLM tool-calling loop at all.
    Same reasoning as cancel_appointment_now/book_appointment_now above.
    """
    return _reschedule_appointment_impl(db, ctx, appointment_id, new_slot_id)


def get_slot_summary(db: Session, clinic_id: uuid.UUID, slot_id: str) -> dict | None:
    """Real doctor/department/time data for a given slot_id, composed entirely from
    the DB — used by appointment_agent to build its deterministic book/reschedule
    confirmation questions, so the confirming question is never freehanded by the
    model (same principle as _booking_card_payload etc.). Returns None for a
    malformed id, an unknown slot, or a slot belonging to a different clinic — the
    caller falls back to the normal LLM tool-calling path in that case.
    """
    try:
        parsed_slot_id = uuid.UUID(slot_id)
    except (ValueError, TypeError, AttributeError):
        return None
    slot = db.get(Slot, parsed_slot_id)
    if slot is None or slot.clinic_id != clinic_id:
        return None
    doctor = db.get(Doctor, slot.doctor_id)
    if doctor is None:
        return None
    department_name = _department_name_for_slot(db, clinic_id, slot)
    tz = _clinic_timezone(db, clinic_id)
    return {
        "slot_id": slot_id,
        "doctor_name": doctor.full_name,
        "department_name": department_name,
        "when": _format_when(slot.start_utc, tz),
    }


def list_upcoming_appointments(db: Session, ctx: ClinicContext) -> list[dict]:
    """Real, current confirmed-and-future appointments for this patient — used by
    appointment_agent's deterministic pre-check (see
    app.services.orchestrator.agents.appointment_agent) to decide whether a cancel/
    reschedule request is already unambiguous or genuinely needs to ask which
    appointment, rather than trusting the model's own get_my_appointments call plus
    freehand reasoning for that decision. Reported live: with 2 real upcoming
    appointments, "cancel my upcoming appointments" silently cancelled the most
    recently booked one instead of asking which — the model never actually asked as
    instructed. Mirrors _get_my_appointments_impl's own "upcoming" query, kept as its
    own function since that one also serializes a `status` field and serves several
    other filters this doesn't need."""
    from sqlalchemy import select

    from app.models.appointment import Appointment

    now = datetime.now(timezone.utc)
    stmt = (
        select(Appointment)
        .join(Slot, Slot.id == Appointment.slot_id)
        .where(
            Appointment.clinic_id == ctx.clinic_id,
            Appointment.patient_id == ctx.user_id,
            Appointment.status == "confirmed",
            Slot.start_utc >= now,
        )
        .order_by(Slot.start_utc.asc())
    )
    appointments = db.execute(stmt).scalars().all()
    tz = _clinic_timezone(db, ctx.clinic_id)

    results = []
    for appointment in appointments:
        out = booking_engine.serialize_appointment(db, appointment)
        results.append(
            {
                "appointment_id": str(out.id),
                "doctor_id": str(out.doctor_id),
                "doctor_name": out.doctor_name,
                "department_name": out.department_name,
                "when": _format_when(out.start_utc, tz),
                # Raw ISO timestamp alongside the human-formatted "when" above —
                # not shown to the patient, but lets appointment_agent's
                # deterministic cancel/reschedule-cutoff pre-check (see
                # _is_within_cancel_reschedule_cutoff) compare against "now"
                # without re-parsing the formatted string.
                "start_utc": out.start_utc.isoformat(),
            }
        )
    return results


@traceable(name="get_my_appointments")
def _get_my_appointments_impl(
    db: Session, ctx: ClinicContext, status_filter: str, limit: int, oldest_first: bool = False
) -> str:
    from sqlalchemy import select

    from app.models.appointment import Appointment

    now = datetime.now(timezone.utc)
    stmt = (
        select(Appointment)
        .join(Slot, Slot.id == Appointment.slot_id)
        .where(Appointment.clinic_id == ctx.clinic_id, Appointment.patient_id == ctx.user_id)
    )

    # Reported live: "what's my EARLIEST cancelled appointment" needs the exact
    # opposite ordering of the default "most recent" behavior every branch below
    # is built for — same status filter, reversed direction. Rather than
    # duplicating each branch's order_by, every `.desc()`/`.asc()` call below is
    # written in matched pairs and this flag just picks which one applies,
    # keeping the "sort by when the status change actually happened" reasoning
    # (see each branch's own comment) identical either way.
    def _dir(desc_col, asc_col):
        return asc_col if oldest_first else desc_col

    normalized = (status_filter or "all").strip().lower()
    if normalized == "upcoming":
        stmt = stmt.where(Appointment.status == "confirmed", Slot.start_utc >= now).order_by(Slot.start_utc.asc())
    elif normalized == "past":
        # Reported live (same class of bug as "cancelled" below): "most recent
        # completed appointment" must reflect when the visit was actually
        # marked completed (booking_engine.confirm_visit), not the slot's
        # original scheduled time — a patient confirming an older visit AFTER
        # already confirming a more-recently-scheduled one would otherwise get
        # the wrong one shown first. There's no dedicated `completed_at`
        # column (unlike `cancelled_at` for cancellation), but `updated_at`
        # (onupdate=func.now()) is bumped by that exact status change, so it's
        # the real "most recent" signal here; falling back to slot time only
        # to break a tie between two rows updated in the same instant.
        stmt = stmt.where(Appointment.status == "completed").order_by(
            _dir(Appointment.updated_at.desc(), Appointment.updated_at.asc()),
            _dir(Slot.start_utc.desc(), Slot.start_utc.asc()),
        )
    elif normalized == "cancelled":
        # Reported live: "what's my most recent cancelled appointment" was
        # ordered by the appointment's original SLOT time, not by when it was
        # actually cancelled — a patient who cancelled an appointment scheduled
        # further out AFTER already cancelling one scheduled sooner would get
        # the wrong one shown first. `cancelled_at` (set in
        # booking_engine.cancel_appointment) is the real "most recent" signal
        # for this filter; falling back to slot time only for the vanishingly
        # rare legacy row with no cancelled_at recorded.
        stmt = stmt.where(Appointment.status == "cancelled").order_by(
            _dir(Appointment.cancelled_at.desc().nullslast(), Appointment.cancelled_at.asc().nullslast()),
            _dir(Slot.start_utc.desc(), Slot.start_utc.asc()),
        )
    elif normalized == "missed":
        # Instructed live: "missed" appointments (booking_engine.confirm_visit's
        # completed=False path, mapped to the pre-existing 'no_show' status —
        # see that function's own docstring) had no filter value at all, so
        # "what appointments have I missed" fell through to the `else` branch
        # below and returned EVERY status mixed together instead of answering
        # the actual question. Same "sort by when the action happened, not the
        # original slot time" fix as "past"/"cancelled" above — no_show is set
        # via that same status assignment, so `updated_at` is the real
        # "most recent" signal here too.
        stmt = stmt.where(Appointment.status == "no_show").order_by(
            _dir(Appointment.updated_at.desc(), Appointment.updated_at.asc()),
            _dir(Slot.start_utc.desc(), Slot.start_utc.asc()),
        )
    else:
        stmt = stmt.order_by(_dir(Slot.start_utc.desc(), Slot.start_utc.asc()))

    stmt = stmt.limit(max(1, min(limit, 50)))
    appointments = db.execute(stmt).scalars().all()
    tz = _clinic_timezone(db, ctx.clinic_id)

    results = []
    for appointment in appointments:
        out = booking_engine.serialize_appointment(db, appointment)
        entry = {
            "appointment_id": str(out.id),
            "doctor_name": out.doctor_name,
            "department_name": out.department_name,
            "when": _format_when(out.start_utc, tz),
            "status": out.status,
        }
        # Reported live: a "most recent cancelled/completed appointment" question
        # was answered with fabricated doctor/department/date details instead of
        # the real data — nothing in this tool's own JSON distinguished "when
        # this was scheduled for" from "when this status change actually
        # happened," so there was no grounded signal for the model to anchor a
        # recency answer on beyond the (already-correctly-ordered, see the sort
        # comments above) row order. Surfacing the real action timestamp
        # explicitly gives both the model and any deterministic caller
        # (app.services.orchestrator.agents.appointment_agent's own "most
        # recent by status" short-circuit) something concrete to cite.
        if out.status == "cancelled" and out.cancelled_at is not None:
            entry["cancelled_when"] = _format_when(out.cancelled_at, tz)
        elif out.status in ("completed", "no_show"):
            entry["status_changed_when"] = _format_when(out.updated_at, tz)
        results.append(entry)

    return json.dumps({"appointments": results}, default=str)


@traceable(name="get_department_availability")
def _get_department_availability_impl(
    db: Session,
    ctx: ClinicContext,
    department_name: str,
    note: str | None = None,
    earliest_date: str | None = None,
    latest_date: str | None = None,
    earliest_time: time | None = None,
    latest_time: time | None = None,
) -> str:
    availability = get_department_availability(
        db,
        ctx.clinic_id,
        department_name,
        earliest_date=_parse_date_arg(earliest_date),
        latest_date=_parse_date_arg(latest_date),
        earliest_time=earliest_time,
        latest_time=latest_time,
    )
    if not availability.found:
        return _department_not_found_message(availability)
    if not availability.doctors:
        if availability.next_available_when is not None:
            message = _no_slots_in_window_message(
                db, ctx.clinic_id, availability.department_name, availability.next_available_when
            )
        else:
            message = _no_slots_message(availability.department_name)
        return _no_slots_payload(availability.department_name, message)
    # `note` carries the model's own one-sentence triage reasoning (e.g. "Based on
    # what you've described, this sounds like something Cardiology should look at")
    # so it renders in the DOCTOR_OPTIONS:: card BEFORE the doctor/slot list, never
    # after — the note field is placed first in the payload dict for exactly that
    # reason. A direct, non-triage availability question ("who's available in
    # Cardiology") simply omits it.
    return _doctor_options_payload(db, ctx.clinic_id, availability, note=note)


@traceable(name="find_doctors_by_name")
def _find_doctors_by_name_impl(db: Session, ctx: ClinicContext, name_query: str) -> str:
    matches = find_doctors_by_name(db, ctx.clinic_id, name_query)
    return json.dumps(
        {
            "matches": [
                {"doctor_name": m.full_name, "department_name": m.department_name} for m in matches
            ]
        }
    )


# ---------------------------------------------------------------------------
# LangChain tool schemas — arguments the model is allowed to supply. No
# patient_id/clinic_id field exists on any of these, by design.
# ---------------------------------------------------------------------------


class _BookArgs(BaseModel):
    slot_id: str = Field(description="The exact slot_id (UUID) of the slot the patient chose, from a previously shown list of options.")
    reason: str | None = Field(default=None, description="Short reason for the visit, if the patient mentioned one.")


class _RescheduleArgs(BaseModel):
    appointment_id: str = Field(description="The exact appointment_id (UUID) of the patient's existing appointment to move.")
    new_slot_id: str = Field(description="The exact slot_id (UUID) of the new slot the patient chose.")


class _CancelArgs(BaseModel):
    appointment_id: str = Field(description="The exact appointment_id (UUID) of the appointment to cancel.")


class _GetMyAppointmentsArgs(BaseModel):
    status: str = Field(
        default="all",
        description=(
            "One of: upcoming, past, cancelled, missed, all. Infer from the patient's phrasing "
            "(e.g. 'next appointment' -> upcoming, 'last visit' -> past, 'appointments I missed'/"
            "'didn't show up to' -> missed, 'my history' -> all). 'missed' is a DIFFERENT status "
            "from 'past' — 'past' only means a completed visit, never a missed one."
        ),
    )
    limit: int = Field(default=10, description="Max number of appointments to return.")


class _GetDepartmentAvailabilityArgs(BaseModel):
    department_name: str = Field(description="The real department name (e.g. 'Cardiology') — see system prompt TOOL USE RULES for how to resolve this.")
    note: str | None = Field(
        default=None,
        description=(
            "Optional. Either your one-sentence triage reasoning (omit if the patient named the "
            "department themselves), or a short answer to another question in the same message — "
            "full rules in the system prompt's `note` paragraphs. Never restate doctor names/slots here."
        ),
    )
    earliest_date: str | None = Field(
        default=None,
        description="Optional ISO date (YYYY-MM-DD) — only slots on/after this date. Use for 'a different day' follow-ups.",
    )
    latest_date: str | None = Field(
        default=None,
        description="Optional ISO date (YYYY-MM-DD), inclusive. Set with earliest_date for a bounded window (e.g. 'today or tomorrow').",
    )


class _FindDoctorsByNameArgs(BaseModel):
    name_query: str = Field(
        description="The doctor name as the patient typed it — don't correct spelling/word order first, let the search match on individual words."
    )


def build_tools(
    db: Session,
    ctx: ClinicContext,
    include_booking_action_tools: bool = True,
    reschedule_redirect_appointment_id: str | None = None,
    cancel_redirect_appointment_id: str | None = None,
    forced_date_window: tuple[str, str] | None = None,
    forced_time_window: tuple[time | None, time | None] | None = None,
    suppress_bare_confirmation_booking: bool = False,
) -> list[StructuredTool]:
    """Builds the tools bound to this request's db session and verified
    ClinicContext via closures — clinic_id/patient_id are never parameters the model
    can set, they are captured here from the JWT-derived ctx.

    get_my_appointments/get_department_availability/find_doctors_by_name (the three
    information-gathering tools) are always bound. book_appointment/
    reschedule_appointment/cancel_appointment (the three that actually mutate an
    appointment) are only bound when include_booking_action_tools is True — see
    app.services.message_classifier.needs_booking_action_tools for the caller-side
    decision, which is deliberately biased toward True whenever unsure. Omitting
    them when genuinely not needed saves a few hundred prompt tokens on turns with
    no booking-action signal at all (e.g. a fresh symptom description, a plain
    clinic-info question); the default of True means nothing changes for any turn
    where booking/reschedule/cancel could plausibly be relevant.

    reschedule_redirect_appointment_id / cancel_redirect_appointment_id: set by
    app.services.orchestrator.agents.appointment_agent once it has already
    deterministically resolved, in code, exactly which of the patient's real
    appointments a cancel/reschedule request refers to. When set, book_appointment
    is transparently redirected to reschedule_appointment for that appointment_id,
    and cancel_appointment always uses that id regardless of what the model passes.
    This is a deliberate belt-and-suspenders enforcement, not just a prompt
    instruction: live testing showed the model still calling book_appointment for a
    reschedule's slot-pick despite an explicit prompt rule against it, creating a
    stray duplicate appointment instead of moving the existing one — a mutating
    action like this needs a real guarantee, not just a hope the model follows the
    rule correctly every time.

    forced_date_window: set by the caller (via
    app.services.chat_tools.resolve_bare_weekday_window on the patient's raw
    message) to an (earliest_date, latest_date) ISO-date pair when the message
    names exactly one bare weekday (e.g. "book a slot on sun"). When set, every
    get_department_availability call this turn uses this window regardless of
    what earliest_date/latest_date the model itself passes (or omits) — reported
    live: the model silently called the tool with no date bounds at all for "sun"
    and relayed whatever the tool's own "earliest available" fallback happened to
    return (Friday), instead of a real "nothing open Sunday, earliest is Friday"
    answer. Same belt-and-suspenders reasoning as reschedule_redirect_appointment_id
    above — a correctness-critical argument gets a real guarantee, not just a
    prompt instruction to compute it correctly.

    forced_time_window: set by the caller (via
    app.services.chat_tools.resolve_time_of_day_window on the patient's raw
    message) to an (earliest_time, latest_time) pair when the message names a
    time-of-day constraint ("after 12 pm", "in the evening"). Reported live:
    this had no equivalent mechanism at all — a time-of-day request was simply
    never applied, silently returning whatever the top 5 earliest-of-the-day
    slots happened to be regardless of what the patient actually asked for.
    Same belt-and-suspenders reasoning as forced_date_window immediately above.

    suppress_bare_confirmation_booking: set by appointment_agent when THIS message
    resolved NOTHING deterministically this turn at all (no action word, no
    doctor name matched, no pending confirmation that's still the assistant's
    literal last turn, no explicit slot_id — see run_appointment_agent's own
    computation). Reported live TWICE: "book with Dr. X at 3pm" -> "Just to
    confirm — book...?" -> an unrelated question (answered correctly) -> a
    generic reply (correctly not booked) -> "Yes"/"do it" — booked for real, from
    a confirmation question that was already stale. The first fix only covered
    literal "yes"/"no"-shaped wording; the second live report showed "do it"
    reaching this exact path too, since it matches neither of those functions —
    there is no fixed list of ways a patient might phrase "go ahead", so this is
    now keyed on the ABSENCE of anything real resolved this turn, not on
    matching specific confirming words. The model, with book_appointment always
    available regardless of the current turn's own context, read its own
    earlier "would you like me to book...?" question still sitting in
    conversation history and decided this later, unrelated message answered it —
    extracting the slot_id straight out of that stale marker's own JSON text.
    Unlike cancel/reschedule (which always have a real appointment_id to
    redirect to), a genuinely fresh booking has no equivalent "correct" id to
    fall back on — refusing outright when this flag is set is the only safe
    response, same principle as cancel_appointment/reschedule_appointment's own
    redirect-required gates just below, adapted for booking's different shape. A
    genuine natural-language slot description ("book with Dr. X at 3pm") always
    sets resolved_match via find_doctors_by_name, so this flag is never set for a
    message with real booking content of its own — only for one that has none.
    """

    def _book(slot_id: str, reason: str | None = None) -> str:
        if reschedule_redirect_appointment_id is not None:
            return _reschedule_appointment_impl(db, ctx, reschedule_redirect_appointment_id, slot_id)
        if suppress_bare_confirmation_booking:
            return (
                "I want to make sure before booking anything — could you tell me again which "
                "appointment you'd like to book?"
            )
        return _book_appointment_impl(db, ctx, slot_id, reason)

    def _reschedule(appointment_id: str, new_slot_id: str) -> str:
        # Same vulnerability class as _cancel just below, closed the same way: the
        # `or appointment_id` fallback used to let the model reschedule using
        # whatever appointment_id IT supplied whenever reschedule_redirect_appointment_id
        # wasn't set — but reschedule_redirect_appointment_id IS set, unconditionally,
        # every time appointment_agent's own deterministic resolution has genuinely
        # determined resolved_appointment_action == "reschedule" this turn (see
        # run_appointment_agent's reschedule_redirect_id) — there is no legitimate
        # scenario where the model is meant to supply its own appointment_id here.
        # The fallback existed purely as an unused escape hatch that also happened to
        # be a live vulnerability: a stale "reschedule_confirm" marker sitting
        # earlier in history (the confirmation no longer the assistant's literal
        # last turn — see _pending_reschedule_confirmation's own history[-1]-only
        # check) combined with appointment_agent always having reschedule_appointment
        # available regardless of the current turn's own context meant the model
        # could independently decide a later, unrelated "yes" answered that stale
        # question and reschedule using an appointment_id it recalled from the old
        # marker's own JSON — with nothing stopping it. Refusing without a verified
        # redirect closes this exactly like _cancel's own fix.
        if reschedule_redirect_appointment_id is None:
            return (
                "I want to make sure before rescheduling anything — could you tell me again which "
                "appointment you'd like to reschedule, and to when?"
            )
        return _reschedule_appointment_impl(db, ctx, reschedule_redirect_appointment_id, new_slot_id)

    def _cancel(appointment_id: str) -> str:
        # Reported live: a cancel-confirmation question was asked ("...let me know
        # if you'd like me to go ahead and cancel it"), the patient then asked an
        # unrelated question, got answered, and only THEN said "yes" — three turns
        # later. The code-level pending-confirmation gate (_pending_cancel_confirmation
        # in appointment_agent.py) correctly requires the confirmation question to be
        # the assistant's LITERAL last turn, so it correctly saw this "yes" as not
        # answering anything live. But needs_booking_action_tools (message_classifier.py)
        # scans the WHOLE history for any marker card ever shown, with no such
        # recency check — so cancel_appointment was still bound as a callable tool,
        # and the LLM, seeing its own earlier "would you like me to cancel it?"
        # question sitting in the full conversation history, independently decided
        # "yes" was answering IT and called this tool directly — with nothing
        # stopping it, since this closure used to trust any appointment_id the model
        # supplied unconditionally.
        #
        # cancel_redirect_appointment_id is ONLY ever set by appointment_agent's own
        # deterministic resolution (see run_appointment_agent's reschedule_redirect_id/
        # cancel_redirect_id, and _pending_cancel_confirmation's own history[-1]-only
        # check) — a genuinely live, code-verified confirmation, never the model's own
        # guess from raw history. Refusing to cancel anything without it closes this at
        # the one place a real mutation can actually happen, regardless of what
        # (possibly stale) context led the model to attempt the call — same
        # never-let-the-model-freehand-a-mutating-action principle already used for
        # reschedule_redirect_appointment_id/forced_date_window above, just enforced as
        # an outright refusal here instead of a silent redirect, since there is no safe
        # appointment_id to redirect a cancel to without a verified, live confirmation.
        if cancel_redirect_appointment_id is None:
            return (
                "I want to make sure before cancelling anything — could you tell me again which "
                "appointment you'd like to cancel?"
            )
        return _cancel_appointment_impl(db, ctx, cancel_redirect_appointment_id)

    def _get_my_appointments(status: str = "all", limit: int = 10) -> str:
        return _get_my_appointments_impl(db, ctx, status, limit)

    # oldest_first is deliberately NOT exposed on the LLM-facing tool above —
    # only appointment_agent's own deterministic "earliest status" short-circuit
    # calls _get_my_appointments_impl directly with it (see that module's
    # _message_asks_for_most_recent_appointment_by_status/_RECENCY_DIRECTION
    # handling), same reasoning as every other deterministic reply in this file:
    # never let the model freehand which direction "earliest" vs "most recent"
    # sorts real DB rows.

    def _get_department_availability(
        department_name: str,
        note: str | None = None,
        earliest_date: str | None = None,
        latest_date: str | None = None,
    ) -> str:
        if forced_date_window is not None:
            earliest_date, latest_date = forced_date_window
        earliest_time, latest_time = forced_time_window if forced_time_window is not None else (None, None)
        return _get_department_availability_impl(
            db, ctx, department_name, note, earliest_date, latest_date, earliest_time, latest_time
        )

    def _find_doctors_by_name(name_query: str) -> str:
        return _find_doctors_by_name_impl(db, ctx, name_query)

    booking_action_tools = [
        StructuredTool.from_function(
            func=_book,
            name="book_appointment",
            description=(
                "Books an appointment for the patient on a specific slot_id that was already "
                "shown to them as an option. Only call this once the patient has clearly picked "
                "a specific slot."
            ),
            args_schema=_BookArgs,
        ),
        StructuredTool.from_function(
            func=_reschedule,
            name="reschedule_appointment",
            description="Reschedules one of the patient's existing confirmed appointments to a new slot_id.",
            args_schema=_RescheduleArgs,
        ),
        StructuredTool.from_function(
            func=_cancel,
            name="cancel_appointment",
            description="Cancels one of the patient's existing confirmed appointments.",
            args_schema=_CancelArgs,
        ),
    ]

    information_tools = [
        StructuredTool.from_function(
            func=_get_my_appointments,
            name="get_my_appointments",
            description=(
                "Looks up the patient's own appointment history/upcoming visits. Use for questions "
                "like 'when is my next appointment', 'when was my last visit', or 'show my history'. "
                "Returns structured data — summarize it conversationally; if it's empty, say so plainly "
                "and never invent an appointment."
            ),
            args_schema=_GetMyAppointmentsArgs,
        ),
        StructuredTool.from_function(
            func=_get_department_availability,
            name="get_department_availability",
            description=(
                "Looks up every ACTIVE doctor in a given real department with their genuinely free "
                "upcoming slots (real structured data, never a guess). For a cross-department question, "
                "call once per real department name and the results are combined into one reply "
                "automatically — never write that summary yourself. Full usage rules (note/earliest_date/"
                "latest_date, doctor-name resolution) are in the system prompt's TOOL USE RULES."
            ),
            args_schema=_GetDepartmentAvailabilityArgs,
        ),
        StructuredTool.from_function(
            func=_find_doctors_by_name,
            name="find_doctors_by_name",
            description=(
                "Searches real, active doctors by name (word-level match, order-independent — e.g. "
                "'raza iqra' matches 'Dr. Iqra Raza'). Call this BEFORE asking anything whenever a "
                "patient names a doctor who isn't already an exact match in context. Returns 0/1/many "
                "matches — see system prompt TOOL USE RULES for how to respond to each case."
            ),
            args_schema=_FindDoctorsByNameArgs,
        ),
    ]

    return information_tools + (booking_action_tools if include_booking_action_tools else [])
