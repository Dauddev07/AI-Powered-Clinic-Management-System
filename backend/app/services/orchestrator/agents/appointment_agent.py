"""Appointment specialist agent (app.services.orchestrator architecture).

Tools bound: book_appointment, reschedule_appointment, cancel_appointment,
get_my_appointments, get_department_availability. find_doctors_by_name is
deliberately NOT bound here — doctor-name resolution happens deterministically in
code below, reusing app.services.department_availability.find_doctors_by_name's own
real-DB word-level matcher directly (the exact function the find_doctors_by_name
TOOL wraps for symptom_agent) rather than re-implementing name matching.

HANDOFF MECHANISM — the reason this module is more than a thin tool-binding wrapper:
when a patient replies to a previously-shown DOCTOR_OPTIONS/DEPARTMENT_LIST card with
a natural-language reference ("book me with Dr. Ahmed at 4pm"), this agent has no
memory of that card unless it's explicitly recovered. Before invoking the LLM agent
loop at all:

  1. Run the patient's raw message through find_doctors_by_name() against the REAL,
     current doctor table (not the card's possibly-stale text). If that turns up
     MORE than one real doctor, short-circuit immediately with a deterministic
     DOCTOR_DISAMBIGUATION_MARKER reply composed in code from real DB rows — never
     asking the model to freehand a marker-prefixed JSON payload itself, same
     "the LLM never composes data-bearing replies" principle every other marker in
     this system already follows (book_appointment/get_department_availability
     etc.). Zero matches falls through normally (no doctor-name reference in this
     message at all). Exactly one match is unambiguous — falls through too, but its
     real department_name is rendered into the prompt as "RESOLVED DOCTOR" so the
     model doesn't have to ask the patient a question this code already answered.
     If that same doctor+department was ALREADY shown to the patient earlier this
     conversation in a real DOCTOR_OPTIONS/DEPARTMENT_LIST card, the confirming
     question is skipped entirely — reported live as the assistant re-asking "did
     you mean Dr. X in Y?" for a doctor it had just listed in a card of its own two
     turns earlier, which reads as not having listened.
  2. Locate the most recent DOCTOR_OPTIONS_MARKER/DEPARTMENT_LIST_MARKER payload in
     history (the exact same scan needs_booking_action_tools already does) and
     render it back into this agent's own system prompt as "PREVIOUSLY SHOWN
     OPTIONS" — the model matches the patient's wording against it exactly the way
     it already matches a FRESH tool result today, then calls book_appointment
     with the real slot_id it finds there. If nothing in it matches (a mismatched
     time, or no card was ever shown this session), the model still has
     get_department_availability bound and can look it up fresh instead of being
     stuck with no way forward.

APPOINTMENT-AMBIGUITY HANDOFF (cancel/reschedule): a cancel or reschedule request
("cancel my upcoming appointments", "reschedule my appointment") is resolved against
the patient's REAL, current appointment list in code, before the LLM ever runs —
mirroring step 1 above, not the plain-prose disambiguation this module used to rely
on. Reported live: with 2 real upcoming appointments, "cancel my upcoming
appointments" was silently applied to the most recently booked one instead of asking
which — the model never actually asked as the prompt instructed. Zero or exactly one
active appointment resolves outright (no ambiguity to ask about). More than one
either narrows to a single match via a doctor name mentioned in the same message, or
short-circuits with a deterministic DOCTOR_DISAMBIGUATION_MARKER (kind="appointment")
built from real appointment rows — same never-let-the-model-freehand-a-marker
principle as step 1. Once resolved (this turn or as a reply to that question), the
appointment_id is both rendered into the prompt AND enforced at the tools-function
level (see app.services.chat_tools.build_tools' reschedule_redirect_appointment_id/
cancel_redirect_appointment_id) — reported live: even with an explicit prompt rule
against it, the model still called book_appointment for a reschedule's slot-pick and
created a stray duplicate appointment instead of moving the existing one. A mutating
action gets a real guarantee, not just a prompt instruction it might not follow.
"""
import json
import re
import uuid
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.core.tenancy import ClinicContext
from app.models.conversation_memory import ConversationMemory
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.services import booking_engine, llm
from app.services.chat_markers import (
    BOOKING_MARKER,
    DEPARTMENT_LIST_MARKER,
    DOCTOR_DISAMBIGUATION_MARKER,
    DOCTOR_OPTIONS_MARKER,
    NO_SLOTS_MARKER,
)
from app.services.chat_tools import (
    _clinic_timezone,
    _doctor_options_payload,
    _format_when,
    _get_my_appointments_impl,
    _no_slots_message,
    book_appointment_now,
    build_tools,
    cancel_appointment_now,
    get_slot_summary,
    list_upcoming_appointments,
    reschedule_appointment_now,
    resolve_bare_weekday_window,
    resolve_time_of_day_window,
)
from app.services.department_availability import (
    find_doctors_by_name,
    get_department_availability,
    list_active_department_names,
)
from app.services.message_classifier import (
    _CANCEL_ACTION_WORDS,
    _RESCHEDULE_ACTION_WORDS,
    DEPARTMENT_TITLE_HINTS,
    _fuzzy_word_in,
    _preceding_assistant_turn_looks_like_a_question,
)
from app.services.orchestrator.symptom_hints import departments_hinted_by_patient_symptom_words

_APPOINTMENT_AGENT_TOOL_NAMES = {
    "book_appointment",
    "reschedule_appointment",
    "cancel_appointment",
    "get_my_appointments",
    "get_department_availability",
}

_INTRO = (
    "You are Cura, a clinic assistant chatbot for a hospital management system, helping "
    "patients book, reschedule, cancel, and check their own appointments."
)

_HANDOFF_INSTRUCTIONS = """\
PREVIOUSLY SHOWN OPTIONS above (if present) is a doctor/availability card this same \
conversation already showed the patient earlier this session — match their wording \
against it exactly the way you would match it against a tool result you just \
received this turn, and use the real slot_id/doctor/department it contains. If \
nothing in it matches what the patient is asking for (a different time, a different \
doctor, or no such list is present at all), call get_department_availability \
yourself to check fresh rather than guessing or claiming nothing is available.
"""

# Reuses message_classifier's own _CANCEL_ACTION_WORDS/_RESCHEDULE_ACTION_WORDS —
# see that module for why these are shared rather than defined twice.
_CANCEL_KEYWORDS = _CANCEL_ACTION_WORDS
_RESCHEDULE_KEYWORDS = _RESCHEDULE_ACTION_WORDS

# Reported live: "book with dr raza ali" -> "Did you mean Dr. Ali Raza in General
# Medicine?" -> patient replies "yes" -> the very next reply asks "which department
# does Dr. Raza Ali work in?", as if the confirmation never happened. Root cause:
# handoff step 1 re-runs find_doctors_by_name on the CURRENT message every turn —
# "yes" contains no doctor name, so resolved_match comes back None and the
# RESOLVED DOCTOR block is dropped from the prompt entirely for this turn. The
# model still technically has the prior question in its raw chat history, but
# live testing shows it isn't reliable about using it (same "prompt alone isn't a
# guarantee" gap fixed elsewhere in this codebase) — it even got the doctor's name
# backwards ("Dr. Raza Ali") in the process. This regex + _most_recent_user_message
# below recover the doctor name deterministically from the patient's OWN prior
# message instead of trusting the model to remember its own question correctly.
_AFFIRMATIVE_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|ya|sure|correct|right|that'?s (?:him|her|it|right|correct))\b",
    re.IGNORECASE,
)


def _is_short_affirmative_reply(message: str) -> bool:
    return bool(_AFFIRMATIVE_RE.match(message.strip()))


# Reported live: tapping a slot button on an already-shown DOCTOR_OPTIONS card sends
# a message that both names the doctor directly AND carries the exact slot_id chosen
# ("I'd like to book the appointment with Dr. Farhan Malik at ... (slot_id: <uuid>)."
# — see ChatPage.jsx's selectSlot). Naming the doctor made direct_name_match_this_turn
# True below, which routed straight into the availability-narrowing short-circuit and
# re-showed the same card instead of ever reaching book_appointment — repeatable on
# every tap, since the short-circuit runs before the LLM/tool-calling loop even starts
# and has no path to booking at all. A message carrying an explicit slot_id is always
# a booking action, never a bare availability check, regardless of what else it names.
_EXPLICIT_SLOT_ID_RE = re.compile(r"slot_id:\s*[0-9a-f-]{8,}", re.IGNORECASE)

# Same "re-showed the same card instead of ever reaching book_appointment" failure
# shape as _EXPLICIT_SLOT_ID_RE above, one level earlier: a message with an explicit
# BOOKING VERB plus a concrete TIME ("book with dr X on sat aug 22 at 8 am") never
# carries a raw slot_id at all — that only exists after the patient clicks a button
# — so it needs its own signal to keep the doctor-narrowing short-circuit below from
# intercepting it. Deliberately requires BOTH the verb AND a real clock time (not
# "book" alone): "book with Dr. Ahmed Khan" names no specific slot yet — that's
# still a genuine narrowing request ("show me his slots so I can pick one"), and
# must keep hitting the short-circuit below exactly as before, or a doctor already
# narrowed to would incorrectly fall through to the LLM with nothing to book.
_EXPLICIT_BOOK_INTENT_RE = re.compile(r"\bbook(?:ing)?\b", re.IGNORECASE)


def _message_expresses_a_specific_slot_pick_intent(message: str) -> bool:
    return bool(_EXPLICIT_BOOK_INTENT_RE.search(message)) and resolve_time_of_day_window(message) is not None

# Reported live: "dr ahmed rana in pulmonology dept" (no such doctor at this clinic)
# silently showed the WHOLE Pulmonology department's real doctors (Dr. Babar Ali —
# a completely different person) instead of telling the patient plainly that
# "Dr. Ahmed Rana" isn't here. Root cause: handoff step 1's zero-matches case is
# documented (see this module's own docstring) as "no doctor-name reference in this
# message at all" and falls through silently — but zero matches from
# find_doctors_by_name means EITHER that, OR the patient plainly attempted a name
# that just doesn't exist, and those two cases need very different replies. This
# regex distinguishes them: "dr"/"doctor" (case-insensitive, "dr" or "dr.") followed
# by real name-like words is a genuine attempt; nothing captured, or only filler
# words like "available"/"free"/"in" right after the title (as in "is there a doctor
# available in Pulmonology" — a real, generic availability question, not a specific
# name), is not.
_DOCTOR_TITLE_RE = re.compile(r"\b(?:dr\.?|doctor)\s+([a-z][a-z'\-]*(?:\s+[a-z][a-z'\-]*){0,4})", re.IGNORECASE)
_NAME_ATTEMPT_STOPWORDS = frozenset({
    "available", "free", "here", "there", "today", "tomorrow", "please", "who",
    "is", "are", "am", "can", "could", "would", "will", "in", "at", "for", "near",
    "working", "open", "busy", "appointment", "appointments", "slot", "slots",
    "schedule", "scheduled", "book", "booking", "the", "a", "an", "any", "some",
    "this", "that", "department", "dept", "clinic",
})


def _extract_attempted_doctor_name(message: str) -> str | None:
    """The leading run of non-filler words right after "dr"/"doctor" in the
    message, or None if nothing there looks like an attempted name — see the
    comment on _DOCTOR_TITLE_RE above for why this distinction matters."""
    match = _DOCTOR_TITLE_RE.search(message)
    if not match:
        return None
    name_words: list[str] = []
    for word in re.findall(r"[a-z']+", match.group(1).lower()):
        if word in _NAME_ATTEMPT_STOPWORDS:
            break
        name_words.append(word)
        if len(name_words) >= 4:
            break
    return " ".join(name_words) if name_words else None


def _message_has_explicit_slot_id(message: str) -> bool:
    return bool(_EXPLICIT_SLOT_ID_RE.search(message))


# Capturing variant of _EXPLICIT_SLOT_ID_RE above — used only by the book/reschedule
# confirmation gate below, which needs the actual id value (to look up the real slot
# via get_slot_summary), not just whether one is present.
_EXPLICIT_SLOT_ID_CAPTURE_RE = re.compile(r"slot_id:\s*([0-9a-f-]{8,})", re.IGNORECASE)


def _extract_slot_id(message: str) -> str | None:
    match = _EXPLICIT_SLOT_ID_CAPTURE_RE.search(message)
    return match.group(1) if match else None


# Reported live: patient asked "show me my upcoming appointment" (one appointment
# shown), then "can i cancel that?" — a QUESTION, not a command — and the
# appointment was cancelled outright with no confirmation. Root cause:
# _detect_action_intent below is a pure keyword check with no concept of sentence
# mood, so a question and an imperative both resolve identically, and the resolved
# single appointment used to go straight into the prompt as an unconditional "Call
# cancel_appointment" instruction. Cancel is the one action here with no natural
# confirming step of its own (reschedule's confirmation IS the patient picking a
# slot from a shown list) — so it gets an explicit, code-enforced confirm-then-act
# gate instead, same "the LLM never freehands a mutating action" principle as
# reschedule/cancel redirect ids in chat_tools.build_tools.
_NEGATIVE_RE = re.compile(
    r"^\s*(?:no|nope|nah|don'?t|do ?not|never ?mind|forget it|cancel that)\b",
    re.IGNORECASE,
)

# Reported live: "no,book me with dr waqas on mon aug 24 at 3.30 instead" — a
# decline PLUS a real new request packed into one message — matched _NEGATIVE_RE's
# prefix check and was treated as a flat decline, silently discarding the new
# request entirely. A pure decline only has filler (punctuation, "thanks",
# "please") after the negative word, e.g. "no thanks" or "don't book that" —
# anything naming a doctor, a date/weekday, a clock time, or "instead" after the
# negative word means the patient moved straight on to a new instruction in the
# same breath, which must fall through to normal handling below instead of
# being swallowed here.
_NEGATIVE_TRAILING_FILLER_RE = re.compile(r"^[\s,.!]*(?:thanks?|thank you|please)?[\s,.!]*$", re.IGNORECASE)
_NEGATIVE_NEW_INSTRUCTION_SIGNAL_RE = re.compile(
    r"\binstead\b|\bdr\.?\b|\bdoctor\b|\bslot_id:|\d"
    r"|\b(?:mon|tue|tues|wed|thu|thurs|fri|sat|sun"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


def _is_short_negative_reply(message: str) -> bool:
    stripped = message.strip()
    match = _NEGATIVE_RE.match(stripped)
    if not match:
        return False
    remainder = stripped[match.end():]
    if _NEGATIVE_TRAILING_FILLER_RE.match(remainder):
        return True
    return not _NEGATIVE_NEW_INSTRUCTION_SIGNAL_RE.search(remainder)


# Reported live: "can i also cancel my appointment?" got the same "Just to confirm
# — you'd like to cancel..." wording as a direct "cancel my appointment" command.
# The confirm-before-act GATE was correct and stayed unchanged (still never cancels
# without an explicit yes below) — but a "can I...?" is a genuine question about
# capability, and answering it with a command-toned confirmation reads as if the
# question was ignored. This only changes the wording for that one phrasing; the
# underlying cancel_confirm marker/pending-confirmation mechanism is identical
# either way, so "yes" after either wording still resolves the same way.
_CAPABILITY_QUESTION_RE = re.compile(
    r"^\s*(?:can|could)\s+i\b|^\s*am\s+i\s+able\s+to\b|^\s*is\s+it\s+possible\s+(?:for\s+me\s+)?to\b",
    re.IGNORECASE,
)


def _message_is_a_capability_question(message: str) -> bool:
    return bool(_CAPABILITY_QUESTION_RE.match(message.strip()))


# Every "I need one more thing from you before I can act" question this module
# asks (cancel confirmation, appointment disambiguation, doctor-name
# disambiguation) is rendered the same way: a DOCTOR_DISAMBIGUATION_MARKER reply
# tagged with a `kind`. Originally each had its own copy of "check the assistant's
# last turn, parse the marker, check the kind" — which is exactly how the
# doctor-name case ended up with no shared recovery path when a later fix (the
# narrowing filter) needed to look at it too. One shared lookup means a new kind
# added later automatically gets the same "is this actually a live pending
# question, not a stale one" guarantee as the others, instead of needing its own
# copy remembered.
def _last_assistant_disambiguation_payload(history: list[ConversationMemory]) -> dict | None:
    """Returns the parsed JSON payload of the assistant's own last turn if it was
    a DOCTOR_DISAMBIGUATION_MARKER-prefixed reply — None for any other last turn,
    including an earlier marker reply the conversation has already moved past, so
    a stale question is never mistaken for a live one."""
    if not history:
        return None
    last = history[-1]
    if getattr(last, "role", None) != "assistant":
        return None
    content = getattr(last, "content", "") or ""
    if not content.startswith(DOCTOR_DISAMBIGUATION_MARKER):
        return None
    try:
        return json.loads(content[len(DOCTOR_DISAMBIGUATION_MARKER):])
    except (ValueError, TypeError):
        return None


def _is_within_cancel_reschedule_cutoff(appointment: dict) -> bool:
    """True once the appointment is too close to its own start time to cancel or
    reschedule — checked deterministically here, BEFORE asking the patient to
    confirm, using the same real CANCEL_RESCHEDULE_CUTOFF booking_engine itself
    enforces server-side. Without this, the confirm question gets asked first and
    only THEN fails once the patient says yes — asking a question whose answer is
    already known to be a dead end. booking_engine's own check remains the real,
    final guard either way (e.g. if enough time passes between this check and the
    patient's confirmation reply)."""
    raw_start_utc = appointment.get("start_utc")
    if not raw_start_utc:
        # No raw timestamp on hand (e.g. an older cached candidate) — fail open and
        # let booking_engine's own server-side check be the real guard, same as
        # before this pre-check existed, rather than blocking on a guess.
        return False
    start_utc = datetime.fromisoformat(raw_start_utc)
    return start_utc - datetime.now(timezone.utc) <= booking_engine.CANCEL_RESCHEDULE_CUTOFF


def _cancel_reschedule_cutoff_reply(appointment: dict, action: str) -> str:
    verb = "cancelled" if action == "cancel" else "rescheduled"
    return (
        f"Your appointment with {appointment['doctor_name']} in {appointment['department_name']} "
        f"on {appointment['when']} is coming up in less than 2 hours, so it can no longer be "
        f"{verb} online. Please contact the clinic directly for last-minute changes."
    )


def _reschedule_cap_reached(db: Session, ctx: ClinicContext, appointment: dict) -> bool:
    """Same "check before asking" principle as the 2-hour cutoff above, for the
    daily per-department reschedule cap (booking_engine.DAILY_DEPARTMENT_RESCHEDULE_
    CAP) — keyed by the appointment's OWN department/day being rescheduled away
    from, same as booking_engine.reschedule_appointment's own real check. Without
    this, the patient gets shown a full list of new times to pick from, only for
    the actual reschedule to fail once they've already chosen one."""
    doctor_id = appointment.get("doctor_id")
    raw_start_utc = appointment.get("start_utc")
    if not doctor_id or not raw_start_utc:
        return False
    doctor = db.get(Doctor, uuid.UUID(doctor_id))
    if doctor is None:
        return False
    tz = ZoneInfo(_clinic_timezone(db, ctx.clinic_id))
    local_date = datetime.fromisoformat(raw_start_utc).astimezone(tz).date()
    return (
        booking_engine._department_day_reschedule_count(db, ctx.clinic_id, ctx.user_id, doctor.department_id, local_date)
        >= booking_engine.DAILY_DEPARTMENT_RESCHEDULE_CAP
    )


def _reschedule_cap_reply(appointment: dict) -> str:
    return (
        f"Your appointment with {appointment['doctor_name']} in {appointment['department_name']} has "
        f"already been rescheduled {booking_engine.DAILY_DEPARTMENT_RESCHEDULE_CAP} times today for that "
        f"department — please contact the clinic directly for any further changes to it today."
    )


def _booking_cap_reached(db: Session, ctx: ClinicContext, picked_slot: dict) -> bool:
    """Same "check before asking" principle as _reschedule_cap_reached above, for
    a FRESH booking — checked against the picked slot's own department/day,
    before ever asking the patient to confirm a booking that
    booking_engine.book_appointment would just reject anyway (see its own
    DAILY_DEPARTMENT_BOOKING_CAP check). Reported live: the patient was asked
    "Just to confirm — book...?", said "yes", and only THEN got told the limit
    was already reached — a whole confirmation round-trip wasted on a booking
    that could never have succeeded."""
    slot_id = picked_slot.get("slot_id")
    if not slot_id:
        return False
    try:
        parsed_slot_id = uuid.UUID(slot_id)
    except (ValueError, TypeError, AttributeError):
        return False
    slot = db.get(Slot, parsed_slot_id)
    if slot is None:
        return False
    doctor = db.get(Doctor, slot.doctor_id)
    if doctor is None:
        return False
    tz = ZoneInfo(_clinic_timezone(db, ctx.clinic_id))
    local_date = slot.start_utc.astimezone(tz).date()
    return (
        booking_engine._department_day_use_count(db, ctx.clinic_id, ctx.user_id, doctor.department_id, local_date)
        >= booking_engine.DAILY_DEPARTMENT_BOOKING_CAP
    )


def _booking_cap_reply(department_name: str) -> str:
    return (
        f"You've reached the limit of {booking_engine.DAILY_DEPARTMENT_BOOKING_CAP} appointments "
        f"in {department_name} for this day."
    )


def _appointment_local_date(db: Session, ctx: ClinicContext, appointment: dict) -> date | None:
    raw_start_utc = appointment.get("start_utc")
    if not raw_start_utc:
        return None
    tz = ZoneInfo(_clinic_timezone(db, ctx.clinic_id))
    return datetime.fromisoformat(raw_start_utc).astimezone(tz).date()


def _reschedule_different_day_reply(appointment: dict) -> str:
    """Matches booking_engine.reschedule_appointment's own same-day-only rule
    (see its docstring) — checked here too, deterministically, before ever
    showing the patient a different day's slots, rather than letting them pick
    one only for the real reschedule call to fail afterward."""
    return (
        f"Your appointment with {appointment['doctor_name']} in {appointment['department_name']} on "
        f"{appointment['when']} can only be rescheduled to a different time on the SAME day. To move it "
        f"to a different day, please cancel this appointment and book a new one for that day instead."
    )


def _cancel_confirmation_reply(appointment: dict, phrased_as_capability_question: bool = False) -> str:
    """Composed entirely from a real, current appointment row — never model-
    generated, same principle as _disambiguation_marker_reply/
    _appointment_disambiguation_reply below."""
    if phrased_as_capability_question:
        question = (
            f"Yes, you can — you have an appointment with {appointment['doctor_name']} in "
            f"{appointment['department_name']} on {appointment['when']}. Just let me know if "
            f"you'd like me to go ahead and cancel it."
        )
    else:
        question = (
            f"Just to confirm — you'd like to cancel your appointment with "
            f"{appointment['doctor_name']} in {appointment['department_name']} on "
            f"{appointment['when']}?"
        )
    payload = {"kind": "cancel_confirm", "question": question, "candidate": appointment}
    return DOCTOR_DISAMBIGUATION_MARKER + json.dumps(payload)


def _pending_cancel_confirmation(history: list[ConversationMemory]) -> dict | None:
    """True only when the assistant's OWN last turn was
    _cancel_confirmation_reply's own question — a genuine reply to it, not just any
    later message, mirroring _pending_appointment_disambiguation below."""
    payload = _last_assistant_disambiguation_payload(history)
    if payload is None or payload.get("kind") != "cancel_confirm":
        return None
    return payload.get("candidate")


# Instructed live: cancel already gets a code-enforced confirm-then-act gate (see
# _cancel_confirmation_reply above) — book and reschedule get the exact same
# treatment here, even though picking a slot from a shown list already signals
# clear intent. Same "the LLM never freehands a mutating action" principle: a
# patient's explicit slot pick is composed into a real, DB-backed confirming
# question (via get_slot_summary — never model-generated) and only actually
# booked/rescheduled once _is_short_affirmative_reply sees an explicit "yes" to
# THAT exact question, mirroring cancel's own kind/payload/pending-check shape
# exactly so this reuses the very same _last_assistant_disambiguation_payload
# lookup instead of a parallel one.
def _book_confirmation_reply(slot: dict) -> str:
    question = (
        f"Just to confirm — book your appointment with {slot['doctor_name']} in "
        f"{slot['department_name']} on {slot['when']}?"
    )
    payload = {"kind": "book_confirm", "question": question, "candidate": slot}
    return DOCTOR_DISAMBIGUATION_MARKER + json.dumps(payload)


def _pending_book_confirmation(history: list[ConversationMemory]) -> dict | None:
    payload = _last_assistant_disambiguation_payload(history)
    if payload is None or payload.get("kind") != "book_confirm":
        return None
    return payload.get("candidate")


def _reschedule_confirmation_reply(appointment: dict, new_slot: dict) -> str:
    question = (
        f"Just to confirm — reschedule your appointment with {appointment['doctor_name']} in "
        f"{appointment['department_name']} from {appointment['when']} to {new_slot['when']}?"
    )
    candidate = {
        "appointment_id": appointment["appointment_id"],
        "new_slot_id": new_slot["slot_id"],
        "doctor_name": appointment["doctor_name"],
        "department_name": appointment["department_name"],
        "when": new_slot["when"],
    }
    payload = {"kind": "reschedule_confirm", "question": question, "candidate": candidate}
    return DOCTOR_DISAMBIGUATION_MARKER + json.dumps(payload)


def _pending_reschedule_confirmation(history: list[ConversationMemory]) -> dict | None:
    payload = _last_assistant_disambiguation_payload(history)
    if payload is None or payload.get("kind") != "reschedule_confirm":
        return None
    return payload.get("candidate")


def _most_recent_user_message(history: list[ConversationMemory]) -> str | None:
    for row in reversed(history):
        if getattr(row, "role", None) == "user":
            return getattr(row, "content", "") or None
    return None


# Reported live: "hook me up with dr waqas" got a plain KB-prose reply describing
# his typical hours (routed to general_info_agent, no marker card — nothing to
# recover a name from later). The very next message, "show me his available slots
# on fri", names no doctor at all (just the pronoun "his") and isn't a bare "yes"
# either, so neither the RESOLVED DOCTOR name-in-message check nor the
# _is_short_affirmative_reply recovery path below fired — the LLM was left to guess
# a department_name from raw history alone, and apparently echoed the KB prose's
# "otolaryngology" (a specialization, not the real department name "ENT"), which
# get_department_availability's exact-match lookup naturally rejected. This pronoun
# check plus _most_recently_named_doctor below recovers the doctor deterministically
# from the patient's own earlier message instead, same principle as the affirmative-
# reply recovery path just below.
_DOCTOR_PRONOUN_WORDS = frozenset({"his", "her", "him", "he", "she", "their", "them"})


def _message_references_a_doctor_by_pronoun(message: str) -> bool:
    words = set(re.findall(r"[a-z0-9']+", message.lower()))
    return bool(words & _DOCTOR_PRONOUN_WORDS)


# Both "recover a doctor named a turn or two ago" and "recover a cancel/reschedule
# verb stated a turn or two ago" (below) need the identical shape of scan: walk
# recent USER turns only, most-recent-first, stop at the first hit, never reach
# past a shared bound. Two independent copies of that loop is how the two
# lookback depths could quietly drift apart if only one were ever tuned — one
# shared scanner removes that risk.
def _scan_recent_user_messages(history: list[ConversationMemory], matcher, limit: int):
    """Returns the first non-None result of `matcher(content)` when scanning the
    most recent `limit` user turns, most recent first. `matcher` takes a single
    message string and returns whatever "found" value matters to the caller (a
    DoctorMatch, an action string, ...) or None to keep scanning."""
    user_messages = [
        getattr(row, "content", "") or "" for row in history if getattr(row, "role", None) == "user"
    ][-limit:]
    for content in reversed(user_messages):
        result = matcher(content)
        if result is not None:
            return result
    return None


_DOCTOR_REFERENCE_LOOKBACK_TURNS = 6


_STRUCTURED_ASSISTANT_MARKERS = (
    BOOKING_MARKER, DOCTOR_OPTIONS_MARKER, DEPARTMENT_LIST_MARKER, NO_SLOTS_MARKER, DOCTOR_DISAMBIGUATION_MARKER,
)


# Reported live: "i am having blury vision" was answered by general_info_agent
# (plain KB prose naming Dr. Farhan Mirza and his schedule — symptom_agent never
# saw this turn at all) rather than a real DOCTOR_OPTIONS card, so the doctor's
# name exists only inside that free-text assistant reply, not in any structured
# marker payload and not in anything the PATIENT typed. "show me available slots
# for him" right after then failed to resolve at all — _most_recently_named_doctor
# only ever scanned patient turns, so a doctor the ASSISTANT named in plain prose
# was invisible to it, and the patient got asked to repeat a department name the
# assistant itself had just given them. Scans assistant turns too, but only as a
# fallback after the patient-turn scan finds nothing (a patient's own naming
# should always win when both exist) and only over plain prose — a
# marker-prefixed reply (a real DOCTOR_OPTIONS/BOOKING/etc. card) is deliberately
# skipped here since that's already handled by its own dedicated mechanism (see
# "PREVIOUSLY SHOWN OPTIONS"/doctor_already_shown above), and scanning it again
# here with a plain word-overlap matcher would just duplicate/race that path.
def _most_recently_named_doctor_in_assistant_prose(db: Session, clinic_id, history: list[ConversationMemory]):
    assistant_messages = [
        getattr(row, "content", "") or ""
        for row in history
        if getattr(row, "role", None) == "assistant"
    ][-_DOCTOR_REFERENCE_LOOKBACK_TURNS:]
    for content in reversed(assistant_messages):
        if content.startswith(_STRUCTURED_ASSISTANT_MARKERS):
            continue
        matches = find_doctors_by_name(db, clinic_id, content)
        if len(matches) == 1:
            return matches[0]
    return None


def _most_recently_named_doctor(db: Session, clinic_id, history: list[ConversationMemory]):
    """Recovers a doctor named a turn or two ago when the patient's current message
    only references them by pronoun ("his"/"her"/"their") instead of repeating the
    name — bounded lookback over user turns only, mirroring
    _most_recent_action_intent's own bounded scan, so this can't reach back into an
    unrelated, much older part of the conversation. Falls back to the assistant's
    own recent plain-prose replies (see
    _most_recently_named_doctor_in_assistant_prose) only when no patient turn names
    one at all."""
    def matcher(content: str):
        matches = find_doctors_by_name(db, clinic_id, content)
        return matches[0] if len(matches) == 1 else None

    from_patient = _scan_recent_user_messages(history, matcher, limit=_DOCTOR_REFERENCE_LOOKBACK_TURNS)
    if from_patient is not None:
        return from_patient
    return _most_recently_named_doctor_in_assistant_prose(db, clinic_id, history)


# Reported live: after a two-doctor DOCTOR_OPTIONS card was shown (General
# Medicine: Dr. Ali Raza, Dr. Farhan Rehman), "only show me available slots for
# dr farhan rehman" re-showed the exact same unfiltered two-doctor card again.
# Root cause: get_department_availability has no doctor-filtering capability at
# all, and the system prompt (see llm._TOOL_RULES_SHARED) explicitly tells the
# model "there is no way to query one doctor's slots directly... relay its card
# as usual" — so even once resolved_match correctly identifies exactly one
# doctor, nothing downstream could act on it. Handled deterministically instead,
# same "never let the model freehand a card, build it from real DB rows" principle
# as every other marker-prefixed reply in this module.
_NARROW_TO_ONE_DOCTOR_WORDS = frozenset({"only", "just", "specifically", "solely"})


def _message_asks_to_narrow_to_one_doctor(message: str) -> bool:
    words = set(re.findall(r"[a-z0-9']+", message.lower()))
    return bool(words & _NARROW_TO_ONE_DOCTOR_WORDS)


# Reported live: "only show available slots for dr iqra" matched TWO doctors
# (Dr. Iqra Qureshi, Dr. Iqra Raza) and correctly asked "did you mean X or Y?" —
# but once the patient answered "I mean Dr. Iqra Raza (ENT)." (a plain name, no
# narrowing word of its own), the ORIGINAL narrowing request was lost: this reply
# message alone doesn't say "only"/"just", so the narrowing check above never
# fired, and the full unfiltered two-doctor card was shown again. The patient's
# intent to narrow was stated once, in the message that triggered the name
# disambiguation — it shouldn't have to be repeated in the very next reply just
# because that reply happened to be answering a different (name) question.
def _pending_doctor_name_disambiguation(history: list[ConversationMemory]) -> dict | None:
    """True only when the assistant's OWN last turn was
    _disambiguation_marker_reply's own question — the doctor-name counterpart to
    _pending_cancel_confirmation/_pending_appointment_disambiguation above, added
    once the narrowing check below needed it too."""
    payload = _last_assistant_disambiguation_payload(history)
    if payload is None or payload.get("kind") != "doctor_name":
        return None
    return payload


def _message_or_pending_name_disambiguation_asks_to_narrow(
    message: str, history: list[ConversationMemory]
) -> bool:
    if _message_asks_to_narrow_to_one_doctor(message):
        return True
    if _pending_doctor_name_disambiguation(history) is None:
        return False
    prior_message = _most_recent_user_message(history)
    return bool(prior_message and _message_asks_to_narrow_to_one_doctor(prior_message))


# Reported live: "no no not reschedule but book" still resolved to
# _detect_action_intent returning "reschedule" — the old check was a plain SET
# intersection against the message's words, which only tells you a keyword is
# PRESENT somewhere, never whether it's negated. A few tokens of local lookback
# is enough to catch the common negation phrasings ("not reschedule", "don't
# reschedule", "never mind rescheduling") without needing real NLP — deliberately
# a short, fixed window rather than scanning the whole message, so an unrelated
# negation earlier in a longer message ("no, my name's Ali, I want to reschedule")
# can't wrongly suppress a genuine request stated later in the same sentence.
_NEGATION_TOKENS = frozenset({
    "not", "no", "never", "without",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
    "won't", "wont", "wouldn't", "wouldnt", "can't", "cant", "cannot",
})
# A negation needs more than one token of lookback to catch realistic phrasing
# ("not going to reschedule", "don't want to reschedule") — but a plain fixed
# window is too blunt on its own: "don't reschedule, just cancel it" put
# "cancel" within a 3-token window of "don't", wrongly negating an unrelated,
# later clause. These reset words mark a clause boundary — walking backward
# from the action word, hitting one of these BEFORE hitting an actual negation
# token means "stop, that negation belongs to a different clause."
_NEGATION_RESET_TOKENS = frozenset({"but", "instead", "actually", "rather", "just"})
_NEGATION_LOOKBACK_TOKENS = 4


def _action_word_is_negated(tokens: list[str], index: int) -> bool:
    window_start = max(0, index - _NEGATION_LOOKBACK_TOKENS)
    for token in reversed(tokens[window_start:index]):
        if token in _NEGATION_TOKENS:
            return True
        if token in _NEGATION_RESET_TOKENS:
            return False
    return False


def _detect_action_intent(message: str) -> str | None:
    """Cheap keyword check for a fresh cancel/reschedule request — deliberately
    narrow (unlike the broad, over-inclusive keyword lists elsewhere in this
    system): a false positive here would trigger the appointment-ambiguity handoff
    below for an unrelated message, and a false negative just falls through to the
    LLM exactly as before this handoff existed, so there's no safety asymmetry
    pushing this one broader.

    Reported live: "reshedule my upcmoing appointment" (a typo'd "reschedule")
    didn't get this same deterministic handoff a correctly-spelled message does —
    exact word-set matching has no tolerance for a dropped/transposed letter.
    Fuzzy (edit-distance) matching via message_classifier._fuzzy_word_in absorbs
    that without loosening this into a broad keyword list — see its own comment
    for why the distance threshold stays safe against false positives.

    Tokens are walked in order (not just checked as an unordered set, see
    _action_word_is_negated above) so an explicitly negated mention — the patient
    retracting their own earlier wording — is skipped rather than accepted."""
    tokens = re.findall(r"[a-z0-9']+", message.lower())
    for index, token in enumerate(tokens):
        if _fuzzy_word_in(token, _CANCEL_KEYWORDS) and not _action_word_is_negated(tokens, index):
            return "cancel"
    for index, token in enumerate(tokens):
        if _fuzzy_word_in(token, _RESCHEDULE_KEYWORDS) and not _action_word_is_negated(tokens, index):
            return "reschedule"
    return None


_ACTION_INTENT_LOOKBACK_TURNS = 6

# Reported live: "reschedule it" -> slots shown -> a slot pick started a reschedule
# confirmation -> "i want to book a new slot" (an explicit retraction — the patient
# now wants to KEEP the existing appointment and book a separate, new one instead)
# -> a later slot pick still got silently treated as completing the ORIGINAL
# reschedule from several turns earlier, producing a reschedule-confirmation prompt
# for what the patient clearly asked to be a fresh booking. Root cause:
# _most_recent_action_intent's plain backward scan only skips a turn that has NO
# reschedule/cancel keyword at all — which is exactly what "book a new slot" looks
# like to it, so it silently scanned straight past the retraction and resurrected
# the stale "reschedule" from further back. Requires a booking word AND one of the
# words below in the SAME message — deliberately narrow, same "no safety asymmetry
# pushing this broader" reasoning as _detect_action_intent itself: a plain "book
# <time>" (the ordinary way a patient continues an ALREADY-in-progress reschedule,
# e.g. step 3 of the reported scenario) must NOT count, or every live reschedule
# would be wrongly dropped the moment the patient names a new time for it.
_NEW_BOOKING_SUPERSEDE_WORDS = frozenset({"new", "another", "separate", "different", "additional", "second"})
_BOOKING_VERB_WORDS = frozenset({"book", "booking"})


def _detect_new_booking_supersede_intent(message: str) -> bool:
    """True when the message explicitly asks to book a NEW/separate appointment —
    e.g. "book a new slot", "I want to book another appointment" — which supersedes
    (rather than merely fails to restate) any earlier reschedule/cancel intent found
    further back in _most_recent_action_intent's lookback scan."""
    words = set(re.findall(r"[a-z0-9']+", message.lower()))
    return bool(words & _BOOKING_VERB_WORDS) and bool(words & _NEW_BOOKING_SUPERSEDE_WORDS)


def _most_recent_action_intent(history: list[ConversationMemory]) -> str | None:
    """Recovers a cancel/reschedule action stated a turn or two ago, for when the
    patient's very next message just repeats a doctor name without restating the
    action verb (e.g. "I mean Dr. X" sent again after the assistant already acted
    on their first request, expecting that same context to still apply). Bounded
    lookback over user turns only, so this can't reach back into an unrelated,
    much older part of the conversation.

    A later turn that explicitly asks to book a new/separate appointment (see
    _detect_new_booking_supersede_intent above) stops the scan outright and reports
    no active action at all, rather than being skipped over like an ordinary
    non-matching turn — see this function's own module comment above for the live
    report this closes. Deliberately NOT built on the shared
    _scan_recent_user_messages helper (unlike _most_recently_named_doctor above):
    that helper only supports "skip and keep scanning" or "stop and return a
    result," with no way to express "stop and return NOTHING," which is exactly
    what a superseding turn needs to do here.
    """
    user_messages = [
        getattr(row, "content", "") or "" for row in history if getattr(row, "role", None) == "user"
    ][-_ACTION_INTENT_LOOKBACK_TURNS:]
    for content in reversed(user_messages):
        if _detect_new_booking_supersede_intent(content):
            return None
        action = _detect_action_intent(content)
        if action is not None:
            return action
    return None


# Reported live: patient described fever + body aches (correctly routed to General
# Medicine), then said "book appointment in Neurology" style requests naming a
# DIFFERENT department their own symptoms don't support — appointment_agent has no
# symptom awareness at all and just showed that department's availability directly,
# no reasoning attached. This is the deterministic mismatch check: a directly-named
# department gets cross-checked against the real symptom-to-department mapping over
# what the patient has actually described THIS session (see
# app.services.orchestrator.symptom_hints), and if they clearly point elsewhere, the
# patient is told so BEFORE any slots are shown — never silently complying with a
# request that contradicts their own stated symptoms.
_MISMATCH_WARNING_MARKER = "might be a better fit than"


# Common professional-title synonyms patients use INSTEAD of a department's own
# name — reported live: "i think cardiologist can be a best fit for it?" (after
# describing leg pain + ear pain, nothing cardiac at all) skipped the mismatch
# check entirely and went straight to a Cardiology availability card, because
# "cardiologist" was never recognized as referring to the "Cardiology" department
# at all — the literal string "cardiology" isn't even a substring of
# "cardiologist" (they diverge before the end: "cardiolog-y" vs "cardiolog-ist"),
# so the old exact-department-name check found nothing. Each entry is (title
# phrase, department-name hint prefix) — the hint is checked as a word-START match
# against the real department name (same technique as
# app.services.orchestrator.symptom_hints), so it tolerates whatever a specific
# clinic actually named the department ("Cardiology", "Cardiology Department",
# etc.) rather than requiring one exact canonical string.
# Reuses message_classifier's own DEPARTMENT_TITLE_HINTS — see that module for
# why this is shared rather than defined twice.
_DEPARTMENT_TITLE_HINTS = DEPARTMENT_TITLE_HINTS


def _departments_named_directly_in_message(message: str, department_names: list[str]) -> list[str]:
    lowered = message.lower()
    named = {name for name in department_names if re.search(rf"\b{re.escape(name.lower())}\b", lowered)}
    for title, hint in _DEPARTMENT_TITLE_HINTS:
        if not re.search(rf"\b{re.escape(title)}\b", lowered):
            continue
        for name in department_names:
            if re.search(rf"\b{re.escape(hint)}", name.lower()):
                named.add(name)
    return list(named)


# Reported live: "i want to see a dentist" got a plain KB-prose reply naming the
# department as "General Dentistry" (prose text, not this clinic's real
# department name, which is just "Dentistry") — then "is there any doctor
# available there?" referred back to it with "there" instead of a name, and
# named no doctor either. The LLM was left to guess a department_name from raw
# history and apparently echoed the KB prose's wrong name, which
# get_department_availability's exact-match lookup naturally rejected. This
# recovers the REAL department name deterministically — via the same
# title-hint matching used for a direct mention — from the patient's own
# earlier message, same principle as _most_recently_named_doctor for a doctor
# pronoun.
_DEPARTMENT_REFERENCE_WORDS = frozenset({"there", "their", "them"})

# Reported live (a second case of the same gap): "show me available slots for
# cardiology" showed a real two-doctor card, then "are there only 2 doctors in
# this dept?" — "this dept" is the same kind of back-reference as "there", just
# phrased differently, and named no specific doctor either. This wasn't
# recognized as a reference at all, so it fell through to the LLM with only the
# (correct, this time) previously-shown card in context — the system prompt
# already tells the model to answer a genuinely separate question like this one
# via the card's own `note` field, but the model didn't reliably do so, just
# re-showing the same card with no explicit answer. Recognizing the phrase lets
# this resolve the same deterministic way "there" does, AND (see
# _doctor_count_question_note below) attach a real, code-computed answer to the
# specific "how many doctors" question instead of hoping the model's own note
# covers it.
_DEPARTMENT_REFERENCE_PHRASES = (
    "this dept", "this department", "that dept", "that department",
    "same dept", "same department",
)


def _message_references_a_department_by_pronoun(message: str) -> bool:
    words = set(re.findall(r"[a-z0-9']+", message.lower()))
    if words & _DEPARTMENT_REFERENCE_WORDS:
        return True
    lowered = message.lower()
    return any(phrase in lowered for phrase in _DEPARTMENT_REFERENCE_PHRASES)


# "doctor"/"doctors" plus every profession-title word ("cardiologist",
# "dermatologist", ...) — a message naming the specialty directly ("how many
# cardiologist are there") is asking about doctor count exactly as much as one
# saying "doctor" literally, so both count as the same trigger word here.
_DOCTOR_COUNT_TRIGGER_WORDS = frozenset({"doctor", "doctors", "specialist", "specialists"}) | frozenset(
    title.split()[0] for title, _ in _DEPARTMENT_TITLE_HINTS
)


def _message_asks_about_doctor_count(message: str) -> bool:
    """True for a question about HOW MANY doctors a department has ("are there
    only 2 doctors", "how many cardiologist are there", "just 1 doctor?") —
    deliberately narrow (requires a doctor/specialty word plus a counting word)
    so this doesn't misfire on an unrelated message that merely contains
    "only"."""
    words = set(re.findall(r"[a-z0-9']+", message.lower()))
    if not words & _DOCTOR_COUNT_TRIGGER_WORDS:
        return False
    if {"how", "many"} <= words:
        return True
    return bool(words & {"only", "just"})


# Reported live: "just give me general info about them not show slots" — asked
# right after a Cardiology DOCTOR_OPTIONS card was shown — got the exact same
# slots card shown back again instead of an answer, because appointment_agent's
# only tool (get_department_availability) always returns a full card with
# slots; there was no way for it to relay "just the doctors, no slots" at all.
# Deliberately broad phrase set (this module's usual fail-safe bias) so it
# catches the common ways a patient asks for this without the word "slots"
# necessarily appearing at all (e.g. "just tell me about them").
_DOCTOR_INFO_NO_SLOTS_RE = re.compile(
    r"\b(general\s+info|just\s+(?:the\s+)?info|just\s+tell\s+me\s+about|"
    r"no\s+slots?\b|not\s+(?:the\s+)?slots?\b|without\s+(?:the\s+)?slots?\b|"
    r"don'?t\s+show\s+(?:the\s+)?slots?\b|not\s+show\s+(?:the\s+)?slots?\b)",
    re.IGNORECASE,
)


def _message_asks_for_doctor_info_no_slots(message: str) -> bool:
    return bool(_DOCTOR_INFO_NO_SLOTS_RE.search(message))


def _doctor_info_no_slots_reply(department_name: str, doctors) -> str:
    """A real, code-computed doctor listing with NO slots — never left to the
    model, which has no tool that can omit slots from get_department_availability's
    card (see _DOCTOR_INFO_NO_SLOTS_RE's own comment)."""
    lines = [
        f"- {d.full_name} — {d.specialization}" if d.specialization else f"- {d.full_name}"
        for d in doctors
    ]
    return f"Here are the doctors in {department_name}:\n" + "\n".join(lines)


# Requested directly: once a slot list has been shown, "how do I book an
# appointment"/"how to book"/"how do I choose a slot" should get an exact,
# deterministic two-step answer — never left to the LLM to freehand (same
# "never let the model freehand it" principle as every other reply in this
# module). Deliberately narrow — requires "how" plus a booking/choosing verb —
# so it can't misfire on an unrelated message that merely contains "book".
_HOW_TO_BOOK_RE = re.compile(
    r"\bhow\s+(?:do|can|could|would|to|does)\b.{0,20}\b"
    r"(?:book|booking|choose|choosing|pick|picking|select|selecting)\b",
    re.IGNORECASE,
)


def _message_asks_how_to_book(message: str) -> bool:
    return bool(_HOW_TO_BOOK_RE.search(message))


_HOW_TO_BOOK_REPLY = (
    "Here's how to book:\n\n"
    "1) Choose any one of the available slots above — they're clickable, you can pick any of them.\n"
    "2) Or just tell me the exact date and time you'd like, and I'll book that slot for you."
)


# Reported live: "what's my most recent cancelled appointment" got a reply naming
# a doctor, department, AND date that matched NOTHING in the patient's real
# appointment history — get_my_appointments is deliberately not a "terminal tool"
# (its result is meant to be phrased conversationally by the model, see
# app.services.llm's own comment on why), which left nothing enforcing that the
# model's final sentence actually reflected the tool's real JSON rather than
# free-texting a plausible-sounding but fabricated answer. A "most recent
# cancelled/completed/missed" question is answered ENTIRELY from the real DB
# instead — same never-let-the-model-freehand-grounded-data principle as every
# other deterministic reply in this module, and reuses _get_my_appointments_impl's
# own already-correct "sort by when the status change actually happened, not the
# slot's original time" ordering (see that function's own comments) rather than
# re-querying separately.
_RECENCY_WORD_RE = re.compile(r"\b(most recent(?:ly)?|latest|last)\b", re.IGNORECASE)
_CANCELLED_STATUS_WORD_RE = re.compile(r"\bcancel+(?:l?ed|lation)?\b", re.IGNORECASE)
_COMPLETED_STATUS_WORD_RE = re.compile(r"\bcomplet+ed\b|\b(?:past|previous) (?:visit|appointment)\b", re.IGNORECASE)
_MISSED_STATUS_WORD_RE = re.compile(
    r"\bmiss(?:ed|ing)?\b|\bno[\s-]?show\b|\bdid ?n[o']?t (?:show|attend|go)\b", re.IGNORECASE
)
_STATUS_FILTER_TO_LABEL = {"cancelled": "cancelled", "past": "completed", "missed": "missed"}


def _message_asks_for_most_recent_appointment_by_status(message: str) -> str | None:
    """Returns the get_my_appointments status_filter value ("cancelled", "past",
    or "missed") if the message is asking for the single most recent appointment
    of that status, else None. Requires an explicit recency word ("most recent",
    "latest", "last") — a plain "show my cancelled appointments" (a real LIST
    request, not a single most-recent one) is deliberately left to the normal
    LLM/tool-calling flow below, which already handles that fine."""
    if not _RECENCY_WORD_RE.search(message):
        return None
    if _CANCELLED_STATUS_WORD_RE.search(message):
        return "cancelled"
    if _COMPLETED_STATUS_WORD_RE.search(message):
        return "past"
    if _MISSED_STATUS_WORD_RE.search(message):
        return "missed"
    return None


def _most_recent_appointment_by_status_reply(db: Session, ctx: ClinicContext, status_filter: str) -> str:
    label = _STATUS_FILTER_TO_LABEL[status_filter]
    parsed = json.loads(_get_my_appointments_impl(db, ctx, status_filter, 1))
    appointments = parsed.get("appointments") or []
    if not appointments:
        return f"You don't have any {label} appointments."
    appt = appointments[0]
    sentence = (
        f"Your most recent {label} appointment was with {appt['doctor_name']} in "
        f"{appt['department_name']}, scheduled for {appt['when']}."
    )
    if appt.get("cancelled_when"):
        sentence += f" It was cancelled on {appt['cancelled_when']}."
    elif appt.get("status_changed_when"):
        marked_word = "confirmed completed" if status_filter == "past" else "marked as missed"
        sentence += f" It was {marked_word} on {appt['status_changed_when']}."
    return sentence


def _doctor_count_note(department_name: str, count: int) -> str:
    """A real, code-computed answer to a "how many doctors" question — never
    left to the model's own note-writing, which live testing showed doesn't
    reliably answer this kind of follow-up even though the system prompt
    already instructs it to."""
    doctor_word = "doctor" if count == 1 else "doctors"
    verb = "is" if count == 1 else "are"
    return f"There {verb} currently {count} {doctor_word} in {department_name}:"


_DEPARTMENT_REFERENCE_LOOKBACK_TURNS = 6


def _most_recently_referenced_department(
    history: list[ConversationMemory], department_names: list[str]
) -> str | None:
    """Recovers a department named a turn or two ago (directly, or via a
    professional-title synonym like "dentist") when the patient's current
    message only refers back to it with "there" — bounded lookback over user
    turns only, mirroring _most_recently_named_doctor's own bounded scan."""
    def matcher(content: str):
        named = _departments_named_directly_in_message(content, department_names)
        return named[0] if len(named) == 1 else None

    return _scan_recent_user_messages(history, matcher, limit=_DEPARTMENT_REFERENCE_LOOKBACK_TURNS)


def _already_warned_about_department_mismatch(history: list[ConversationMemory], named_department: str) -> bool:
    """True if this exact mismatch warning was already given for this department
    earlier in the session — a patient who repeats the same request after seeing it
    once is making an informed choice to proceed anyway, not missing the warning."""
    marker_dept = named_department.lower()
    for row in history:
        if getattr(row, "role", None) != "assistant":
            continue
        content = (getattr(row, "content", "") or "").lower()
        if _MISMATCH_WARNING_MARKER in content and marker_dept in content:
            return True
    return False


def _department_mismatch_reply(named_department: str, hinted: dict[str, str]) -> str:
    labels = ", ".join(sorted(set(hinted.values())))
    hinted_names = ", ".join(hinted.keys())
    return (
        f"Based on the {labels} you described, {hinted_names} {_MISMATCH_WARNING_MARKER} {named_department}. "
        f"Would you like me to show {hinted_names} availability instead, or would you still like to see "
        f"{named_department}'s availability?"
    )


def _most_recent_availability_marker(history: list[ConversationMemory]) -> dict | None:
    """Recovers the structured payload of the most recent DOCTOR_OPTIONS_MARKER or
    DEPARTMENT_LIST_MARKER assistant turn in `history` — the same scan
    app.services.message_classifier.needs_booking_action_tools already does to
    decide these tools are even needed, reused here to actually resolve one."""
    for row in reversed(history):
        if getattr(row, "role", None) != "assistant":
            continue
        content = getattr(row, "content", "") or ""
        for marker in (DOCTOR_OPTIONS_MARKER, DEPARTMENT_LIST_MARKER):
            if content.startswith(marker):
                try:
                    return json.loads(content[len(marker):])
                except (ValueError, TypeError):
                    return None
    return None


def _doctor_already_shown(history: list[ConversationMemory], doctor_name: str, department_name: str) -> bool:
    """True when this doctor was already surfaced to the patient earlier in
    `history` — either in a real DOCTOR_OPTIONS_MARKER/DEPARTMENT_LIST_MARKER
    card (scans every such card, not just the most recent one, since it may be
    several turns back), OR in a plain free-form assistant reply that names
    them (see the fallback below). Used to skip the RESOLVED DOCTOR confirming
    question — and, downstream, to let the narrowing short-circuit build a
    real single-doctor card — when the patient is plainly just referencing
    someone the assistant itself already told them about, not naming a doctor
    cold.

    Reported live: "dr farhan" got a plain prose confirmation ("Dr Farhan
    Malik – General Cardiology – available on Saturday and Sunday...") with
    no DOCTOR_OPTIONS/DEPARTMENT_LIST card at all — RESOLVED DOCTOR's
    "ask ONE confirming question... unless the patient already confirmed"
    instruction left the LLM to informally acknowledge it in prose instead.
    The very next turn, "show me his available slots", found no card to
    match here, so doctor_already_shown stayed False, the narrowing
    short-circuit below never fired (it requires doctor_already_shown), and
    the fallback LLM tool call — which cannot filter get_department_availability
    by doctor at all — showed the whole department instead of just this one
    doctor. A free-form assistant reply that names this exact doctor now also
    counts as "already shown": the department isn't cross-checked in that
    branch (plain prose has no structured department field to check against),
    but a real doctor's full name is specific enough on its own. Worst case
    is skipping one redundant confirming question when the name only
    appeared in passing — a smaller cost than losing doctor-scoping
    outright."""
    target_name = doctor_name.strip().lower()
    target_department = department_name.strip().lower()
    for row in history:
        if getattr(row, "role", None) != "assistant":
            continue
        content = getattr(row, "content", "") or ""
        if content.startswith(DOCTOR_OPTIONS_MARKER):
            try:
                department_groups = [json.loads(content[len(DOCTOR_OPTIONS_MARKER):])]
            except (ValueError, TypeError):
                continue
        elif content.startswith(DEPARTMENT_LIST_MARKER):
            try:
                department_groups = json.loads(content[len(DEPARTMENT_LIST_MARKER):]).get("departments", [])
            except (ValueError, TypeError):
                continue
        elif content.startswith(DOCTOR_DISAMBIGUATION_MARKER):
            # A pending "did you mean X or Y?" question — the patient hasn't
            # confirmed anything yet, so naming this doctor here doesn't count.
            continue
        else:
            # Periods after "Dr" are inconsistent in free-form prose (the LLM
            # doesn't always write "Dr." the same way it's stored in the DB) —
            # stripped from both sides before comparing so "Dr Farhan Malik"
            # in prose still matches a stored "Dr. Farhan Malik".
            if target_name.replace(".", "") in content.strip().lower().replace(".", ""):
                return True
            continue
        for group in department_groups:
            if (group.get("department_name") or "").strip().lower() != target_department:
                continue
            for doctor in group.get("doctors", []):
                if (doctor.get("doctor_name") or "").strip().lower() == target_name:
                    return True
    return False


def _disambiguation_marker_reply(matches: list) -> str:
    """Composed entirely from real DoctorMatch rows returned by
    find_doctors_by_name() — never model-generated, same principle as every other
    marker-prefixed reply in this system."""
    candidates = [{"doctor_name": m.full_name, "department_name": m.department_name} for m in matches]
    names = ", ".join(f"{m.full_name} ({m.department_name})" for m in matches)
    question = f"I found more than one doctor matching that name — did you mean {names}? Could you tell me which one?"
    payload = {"kind": "doctor_name", "question": question, "candidates": candidates}
    return DOCTOR_DISAMBIGUATION_MARKER + json.dumps(payload)


def _appointment_disambiguation_reply(action: str, candidates: list[dict]) -> str:
    """Composed entirely from real, current appointment rows (see
    list_upcoming_appointments) — never model-generated, same principle as
    _disambiguation_marker_reply above."""
    verb = "cancel" if action == "cancel" else "reschedule"
    # Reported live: when both pending appointments are with the SAME doctor,
    # asking "which one" by doctor name is meaningless — the answer is always
    # that same name, so the patient's reply matched every candidate at once
    # and the question just repeated itself forever (see _match_candidate).
    # Ask by date/time instead in that case, since that's the only thing that
    # actually distinguishes them.
    doctor_names = {c["doctor_name"] for c in candidates}
    if len(doctor_names) == 1:
        whens = ", ".join(c["when"] for c in candidates)
        question = (
            f"You have more than one upcoming appointment with {candidates[0]['doctor_name']} — "
            f"which one would you like to {verb}: {whens}?"
        )
    else:
        names = ", ".join(f"{c['doctor_name']} ({c['department_name']}, {c['when']})" for c in candidates)
        question = f"You have more than one upcoming appointment — which one would you like to {verb}: {names}?"
    payload = {"kind": "appointment", "action": action, "question": question, "candidates": candidates}
    return DOCTOR_DISAMBIGUATION_MARKER + json.dumps(payload)


def _pending_appointment_disambiguation(history: list[ConversationMemory]) -> dict | None:
    """True only when the assistant's OWN last turn was
    _appointment_disambiguation_reply's own question — a genuine reply to it, not
    just any later message, so an unrelated message sent after being asked doesn't
    get misread as answering a stale question."""
    payload = _last_assistant_disambiguation_payload(history)
    if payload is None or payload.get("kind") != "appointment":
        return None
    return payload


def _match_candidate(message: str, candidates: list[dict]) -> dict | None:
    """Matches the patient's reply against exactly one candidate by doctor name
    (full name, or its last word as a plain surname reference like "the Sheikh
    one") — deliberately simple and exact rather than fuzzy, since an ambiguous or
    failed match should re-ask (see run_appointment_agent), never guess.

    When two candidates share the same doctor (see _appointment_disambiguation_reply),
    the name tier alone matches both at once and stays ambiguous on purpose — falls
    through to matching the appointment's own "when" (date/time) text instead, which
    is what actually distinguishes them."""
    lowered = message.lower()
    name_matches = [
        c for c in candidates
        if c["doctor_name"].strip().lower() in lowered
        or c["doctor_name"].strip().split()[-1].lower() in lowered
    ]
    if len(name_matches) == 1:
        return name_matches[0]

    pool = name_matches if len(name_matches) > 1 else candidates
    when_matches = [c for c in pool if c.get("when") and c["when"].strip().lower() in lowered]
    return when_matches[0] if len(when_matches) == 1 else None


def _build_system_prompt(
    language_name: str,
    shown_options_payload: dict | None,
    resolved_match=None,
    doctor_already_shown: bool = False,
    resolved_appointment: dict | None = None,
    resolved_appointment_action: str | None = None,
) -> str:
    # Handoff step 1 resolved the patient's message to exactly one real doctor —
    # surface that directly so the model uses the real department_name it already
    # has (e.g. to call get_department_availability) instead of asking the patient
    # which department this doctor is in. It still confirms the doctor's IDENTITY
    # first though (word-overlap matching, even exact-subset matching, can still land
    # on the wrong doctor for genuinely unusual name overlaps) — same one-line
    # confirm-before-proceeding pattern find_doctors_by_name's TOOL USE RULES already
    # require of symptom_agent for its own single-match case. Skipped entirely when
    # this doctor+department was already shown in a real card earlier this
    # conversation (doctor_already_shown) — asking again in that case reads as not
    # having listened, not as being careful.
    if resolved_match is None:
        resolved_match_block = ""
    elif doctor_already_shown:
        resolved_match_block = (
            f"RESOLVED DOCTOR: the patient's message names {resolved_match.full_name} in "
            f"{resolved_match.department_name} — this exact doctor and department were "
            f"already shown to the patient earlier this conversation in a real availability "
            f"card, so this is not an unconfirmed guess. Do NOT ask a confirming question "
            f"about who this doctor is or which department they're in — proceed directly "
            f"(e.g. call get_department_availability for {resolved_match.department_name} if "
            f"you still need current slots).\n\n"
        )
    else:
        resolved_match_block = (
            f"RESOLVED DOCTOR: the patient's message names a doctor that matches exactly "
            f"one real doctor at this clinic — {resolved_match.full_name}, department "
            f"{resolved_match.department_name}. Before calling get_department_availability "
            f"for them, ask ONE direct confirming question naming the doctor and real "
            f'department (e.g. "Did you mean {resolved_match.full_name} in '
            f'{resolved_match.department_name}?") and wait for the patient to confirm — do '
            f"not check availability in the same turn you resolve the name. Skip asking "
            f"again ONLY if the patient already confirmed this exact doctor earlier in "
            f"this same conversation (check the conversation history above) — never ask "
            f"the same confirming question twice for the same doctor.\n\n"
        )
    if resolved_appointment is None:
        resolved_appointment_block = ""
    else:
        # Only "reschedule" ever reaches here — a resolved "cancel" always
        # short-circuits earlier in run_appointment_agent with a deterministic
        # confirm-first question, never reaching the LLM/tool layer at all.
        action_sentence = (
            "This is a RESCHEDULE: once the patient picks a new time from the slots you "
            "show them, call reschedule_appointment with this exact appointment_id and the "
            "new slot_id — never book_appointment, even if their wording sounds like "
            'booking a fresh appointment (e.g. "I\'d like to book the appointment with Dr. '
            'X").\n\n'
        )
        resolved_appointment_block = (
            f"RESOLVED APPOINTMENT: the patient's {resolved_appointment_action} request refers "
            f"to exactly one real, current appointment — {resolved_appointment['doctor_name']} "
            f"in {resolved_appointment['department_name']} on {resolved_appointment['when']}, "
            f"appointment_id {resolved_appointment['appointment_id']}. This has already been "
            f"identified deterministically — do NOT call get_my_appointments again this turn, "
            f"and do NOT ask the patient which appointment they mean, that's already settled.\n\n"
            f"{action_sentence}"
        )
    shown_options_block = (
        f"PREVIOUSLY SHOWN OPTIONS:\n{json.dumps(shown_options_payload)}\n\n{_HANDOFF_INSTRUCTIONS}\n"
        if shown_options_payload
        else ""
    )
    tail = llm._TAIL_STYLE_RULES.format(language_name=language_name)
    return (
        f"{_INTRO}\n\n"
        f"{llm._OFF_TOPIC_AND_INTEGRITY_RULES}"
        f"TOOL USE RULES:\n{llm._TOOL_RULES_SHARED}{llm._TOOL_RULES_APPOINTMENT_ACTIONS}\n"
        f"{resolved_match_block}"
        f"{resolved_appointment_block}"
        f"{shown_options_block}"
        f"{tail}"
        f"Today's date is {llm._current_date_str()}.\n"
    )


def run_appointment_agent(
    db: Session,
    ctx: ClinicContext,
    message: str,
    language: str,
    history: list[ConversationMemory],
) -> str:
    # Appointment-ambiguity handoff: resolved BEFORE the doctor-name handoff below,
    # so a reply naming a doctor to answer "which appointment?" (e.g. "the one with
    # Dr. Sheikh") resolves the APPOINTMENT, not a fresh RESOLVED DOCTOR confirmation
    # for that same doctor.
    resolved_appointment: dict | None = None
    resolved_appointment_action: str | None = None

    # Cancel-confirmation handoff: resolved BEFORE anything else, so a reply to the
    # assistant's own "just to confirm — cancel X?" question is never re-interpreted
    # as a fresh, unrelated message.
    cancel_pending = _pending_cancel_confirmation(history)
    if cancel_pending is not None:
        if _is_short_affirmative_reply(message):
            return cancel_appointment_now(db, ctx, cancel_pending["appointment_id"])
        if _is_short_negative_reply(message):
            return (
                f"No problem — your appointment with {cancel_pending['doctor_name']} in "
                f"{cancel_pending['department_name']} on {cancel_pending['when']} was not cancelled."
            )
        # Anything else (a new question, a change of mind stated differently, etc.)
        # falls through to normal handling below rather than being forced into a
        # yes/no box — worst case it re-asks, which is always safe for a mutating
        # action like this.

    # Book/reschedule-confirmation handoffs — same reasoning and same "resolved
    # before anything else" placement as cancel's above, so a reply to either
    # confirming question is never re-interpreted as an unrelated fresh message.
    book_pending = _pending_book_confirmation(history)
    if book_pending is not None:
        if _is_short_affirmative_reply(message):
            return book_appointment_now(db, ctx, book_pending["slot_id"])
        if _is_short_negative_reply(message):
            return (
                f"No problem — the appointment with {book_pending['doctor_name']} in "
                f"{book_pending['department_name']} on {book_pending['when']} was not booked."
            )
        # Falls through otherwise, same as cancel's own pending check above.

    reschedule_pending = _pending_reschedule_confirmation(history)
    if reschedule_pending is not None:
        if _is_short_affirmative_reply(message):
            return reschedule_appointment_now(
                db, ctx, reschedule_pending["appointment_id"], reschedule_pending["new_slot_id"]
            )
        if _is_short_negative_reply(message):
            return (
                f"No problem — your appointment with {reschedule_pending['doctor_name']} in "
                f"{reschedule_pending['department_name']} was not rescheduled."
            )
        # Falls through otherwise, same as cancel's own pending check above.

    # "How do I book an appointment" once slots have already been shown gets an
    # exact, deterministic two-step answer instead of the LLM freehanding one —
    # see _HOW_TO_BOOK_REPLY's own comment. Only fires when a real slot list is
    # actually on screen (a DOCTOR_OPTIONS_MARKER/DEPARTMENT_LIST_MARKER card
    # somewhere in history) — otherwise there's nothing to "choose from above"
    # yet, and the message falls through to the normal LLM/tool-calling flow.
    if _message_asks_how_to_book(message) and _most_recent_availability_marker(history) is not None:
        return _HOW_TO_BOOK_REPLY

    # "What's my most recent cancelled/completed/missed appointment" is answered
    # entirely from the real DB — see _most_recent_appointment_by_status_reply's
    # own comment for the reported hallucination this replaces.
    most_recent_status_filter = _message_asks_for_most_recent_appointment_by_status(message)
    if most_recent_status_filter is not None:
        return _most_recent_appointment_by_status_reply(db, ctx, most_recent_status_filter)

    pending = _pending_appointment_disambiguation(history)
    if pending is not None:
        candidate = _match_candidate(message, pending["candidates"])
        if candidate is None:
            return _appointment_disambiguation_reply(pending["action"], pending["candidates"])
        resolved_appointment = candidate
        resolved_appointment_action = pending["action"]
    else:
        action = _detect_action_intent(message)
        name_matches = find_doctors_by_name(db, ctx.clinic_id, message)
        if action is None and len(name_matches) == 1:
            # No cancel/reschedule keyword in THIS message, but it names exactly
            # one real doctor — check whether that action was requested recently
            # rather than assuming this is unrelated to appointments at all.
            action = _most_recent_action_intent(history)
        if action is not None:
            active = list_upcoming_appointments(db, ctx)
            # Reported live: patient cancelled their appointment with Dr. Hashmi,
            # then repeated "I mean Dr. Mahnoor Hashmi (Orthopedics)." — with only
            # one appointment left (Dr. Sheikh's), the old `elif len(active) == 1`
            # branch below picked it BLINDLY, ignoring that the named doctor was
            # Hashmi, not Sheikh, and silently cancelled the wrong one. Checking
            # a named doctor against the real active list FIRST — regardless of
            # how many appointments remain — means a doctor with no current
            # appointment always gets a clear "you don't have one" reply instead
            # of ever falling back to any other appointment on the list.
            if len(name_matches) == 1:
                narrowed = [a for a in active if a["doctor_name"] == name_matches[0].full_name]
                if len(narrowed) == 1:
                    resolved_appointment = narrowed[0]
                    resolved_appointment_action = action
                else:
                    verb = "cancel" if action == "cancel" else "reschedule"
                    return f"You don't have an upcoming appointment with {name_matches[0].full_name} to {verb}."
            elif len(active) > 1:
                return _appointment_disambiguation_reply(action, active)
            elif len(active) == 1:
                resolved_appointment = active[0]
                resolved_appointment_action = action

    # Cancel is a one-way, immediately-effective mutation with no natural
    # confirming step of its own (unlike reschedule, where picking a new slot IS
    # the confirmation) — so a freshly resolved cancel never reaches the LLM/tool
    # layer at all; it always asks first, deterministically, and only actually
    # cancels once _pending_cancel_confirmation above sees an explicit "yes".
    if resolved_appointment is not None and resolved_appointment_action == "cancel":
        if _is_within_cancel_reschedule_cutoff(resolved_appointment):
            return _cancel_reschedule_cutoff_reply(resolved_appointment, "cancel")
        return _cancel_confirmation_reply(resolved_appointment, _message_is_a_capability_question(message))

    # Reported live: "reschedule my appointment with Dr. X" correctly resolved
    # WHICH appointment (resolved_appointment above), but get_department_availability
    # has no doctor_id parameter exposed to the LLM at all — its only tool-callable
    # shape is department-wide — so the model's only option was to show the ENTIRE
    # department's doctors, not just the one being rescheduled. Skipped once the
    # message already carries the slot the patient picked (explicit slot_id) — this
    # is a "show me options" step, and re-firing it on that later turn would re-show
    # the same card forever instead of ever reaching reschedule_appointment, the
    # same loop already fixed for a fresh booking's slot-pick step.
    if (
        resolved_appointment is not None
        and resolved_appointment_action == "reschedule"
        and not _message_has_explicit_slot_id(message)
    ):
        if _is_within_cancel_reschedule_cutoff(resolved_appointment):
            return _cancel_reschedule_cutoff_reply(resolved_appointment, "reschedule")
        if _reschedule_cap_reached(db, ctx, resolved_appointment):
            return _reschedule_cap_reply(resolved_appointment)
        forced_window = resolve_bare_weekday_window(message)
        appointment_date = _appointment_local_date(db, ctx, resolved_appointment)
        if forced_window is not None and appointment_date is not None:
            requested_start = date.fromisoformat(forced_window[0])
            requested_end = date.fromisoformat(forced_window[1])
            if not (requested_start <= appointment_date <= requested_end):
                return _reschedule_different_day_reply(resolved_appointment)
        # Reschedule only ever offers same-day times (see booking_engine.
        # reschedule_appointment's own hard rule) — whether or not the patient
        # named a day at all, the window is always pinned to the appointment's
        # own day, never left open to "earliest available across any day".
        earliest_date = appointment_date
        latest_date = appointment_date
        time_window = resolve_time_of_day_window(message)
        earliest_time, latest_time = time_window if time_window else (None, None)
        availability = get_department_availability(
            db,
            ctx.clinic_id,
            resolved_appointment["department_name"],
            doctor_id=uuid.UUID(resolved_appointment["doctor_id"]),
            earliest_date=earliest_date,
            latest_date=latest_date,
            earliest_time=earliest_time,
            latest_time=latest_time,
        )
        if not availability.doctors:
            if availability.next_available_when is not None:
                tz = _clinic_timezone(db, ctx.clinic_id)
                return (
                    f"{resolved_appointment['doctor_name']} doesn't have any open slots in that "
                    f"window. Their earliest available slot is "
                    f"{_format_when(availability.next_available_when, tz)}. Would you like me to "
                    "reschedule to that instead, or check a different day?"
                )
            return f"{resolved_appointment['doctor_name']} doesn't have any open slots right now. Please check back later."
        return _doctor_options_payload(db, ctx.clinic_id, availability)

    # Instructed live: reschedule now gets the same code-enforced confirm-then-act
    # gate as cancel — the patient has already picked a specific new slot at this
    # point (that's exactly what _message_has_explicit_slot_id is true for, the
    # complementary case to the block just above), so this composes a real,
    # DB-backed confirming question from it via get_slot_summary rather than
    # letting the LLM call reschedule_appointment immediately. Falls through to
    # the normal LLM/tool-calling path only if the slot lookup itself fails (e.g. a
    # stale/tampered slot_id) — never silently reschedules without confirmation.
    if (
        resolved_appointment is not None
        and resolved_appointment_action == "reschedule"
        and _message_has_explicit_slot_id(message)
    ):
        picked_slot_id = _extract_slot_id(message)
        new_slot = get_slot_summary(db, ctx.clinic_id, picked_slot_id) if picked_slot_id else None
        if new_slot is not None:
            return _reschedule_confirmation_reply(resolved_appointment, new_slot)

    # Symptom-vs-department mismatch check: only for a fresh booking-department
    # reference, never for a cancel/reschedule action already resolved above (that's
    # about an EXISTING appointment, not choosing a new department). If the patient
    # directly names a real department, and their own earlier symptoms in THIS
    # session point somewhere else entirely, they're told before any slots are shown
    # — but only once per department (see _already_warned_about_department_mismatch)
    # so repeating the same request after seeing the warning proceeds normally,
    # respecting their informed choice rather than refusing forever.
    if resolved_appointment is None:
        department_names = list_active_department_names(db, ctx.clinic_id)
        named_departments = _departments_named_directly_in_message(message, department_names)

        if _message_asks_for_doctor_info_no_slots(message):
            # Resolves the department the same three ways every other "about
            # what was just shown" follow-up in this module does: named
            # directly in THIS message, a "there" pronoun back-reference, or —
            # since "them" here refers to doctors, not a department, neither of
            # those necessarily fires — falling back to whatever card was most
            # recently actually shown.
            referenced_department = (
                named_departments[0]
                if len(named_departments) == 1
                else _most_recently_referenced_department(history, department_names)
            )
            if referenced_department is None:
                last_shown = _most_recent_availability_marker(history)
                if last_shown and last_shown.get("department_name") in department_names:
                    referenced_department = last_shown["department_name"]
            if referenced_department is not None:
                availability = get_department_availability(db, ctx.clinic_id, referenced_department)
                if not availability.doctors:
                    return _no_slots_message(referenced_department)
                return _doctor_info_no_slots_reply(referenced_department, availability.doctors)

        if named_departments:
            hinted = departments_hinted_by_patient_symptom_words("", history, department_names, set())
            for named in named_departments:
                if (
                    hinted
                    and named not in hinted
                    and not _already_warned_about_department_mismatch(history, named)
                ):
                    return _department_mismatch_reply(named, hinted)
            # Fallback beyond the keyword table above: reported live, "severe pain
            # in my leg" led an EARLIER turn to recommend and show Orthopedics (a
            # real DOCTOR_OPTIONS card, via the symptom-triage agent's own
            # reasoning) — but symptom_hints' keyword table judges bare "leg" +
            # "pain" with no explicit injury verb as General Medicine territory,
            # not Orthopedics, so `hinted` above agreed with a same-turn "general
            # doc" request and the mismatch above never fired, even though it
            # contradicted what was actually shown. Comparing against the real
            # most-recently-shown card's department (ground truth, not a
            # re-guessed keyword hint) catches this gap the keyword table alone
            # cannot.
            last_shown = _most_recent_availability_marker(history)
            # Only when that card carried a real `note` — i.e. was actually
            # symptom-inferred (see `note`'s own omission rule in llm.py: it's
            # null whenever the patient named the department themselves). Two
            # independent direct specialty requests in a row ("cardiologist",
            # then "dermatologist") are the patient's own free choice each
            # time, not a contradiction — nothing to warn about.
            last_department = (
                last_shown.get("department_name") if last_shown and last_shown.get("note") else None
            )
            if last_department in department_names:
                for named in named_departments:
                    if named != last_department and not _already_warned_about_department_mismatch(history, named):
                        return _department_mismatch_reply(named, {last_department: "symptoms"})
            # Reported live: "how many cardiologist are there in this clinic??"
            # names a real department directly (via the title-hint "cardiologist"
            # -> "Cardiology") and has no symptom-mismatch issue, but nothing
            # answered the actual "how many" question — it fell through to the
            # LLM with no explicit RESOLVED DEPARTMENT guidance, leaving it to
            # independently notice the real department name. Answered directly
            # and deterministically instead, same principle as the pronoun path
            # just below.
            if len(named_departments) == 1 and _message_asks_about_doctor_count(message):
                referenced_department = named_departments[0]
                availability = get_department_availability(db, ctx.clinic_id, referenced_department)
                if not availability.doctors:
                    return _no_slots_message(referenced_department)
                note = _doctor_count_note(referenced_department, len(availability.doctors))
                return _doctor_options_payload(db, ctx.clinic_id, availability, note=note)
        elif _message_references_a_department_by_pronoun(message):
            # No department named directly in THIS message, just "there" — recover
            # the real department from the patient's own earlier message and answer
            # directly from the real DB, never leaving the LLM to guess a name from
            # possibly-wrong prose in history.
            referenced_department = _most_recently_referenced_department(history, department_names)
            if referenced_department is not None:
                availability = get_department_availability(db, ctx.clinic_id, referenced_department)
                if not availability.doctors:
                    return _no_slots_message(referenced_department)
                note = (
                    _doctor_count_note(referenced_department, len(availability.doctors))
                    if _message_asks_about_doctor_count(message)
                    else None
                )
                return _doctor_options_payload(db, ctx.clinic_id, availability, note=note)

    # Handoff step 1: deterministic ambiguity check against the real doctor table.
    # A message with no doctor-name reference at all (e.g. "cancel my appointment")
    # simply returns zero matches here and falls through normally below. Skipped
    # when the appointment-ambiguity handoff above already resolved this turn, so a
    # doctor name used only to answer "which appointment?" doesn't also trigger its
    # own separate RESOLVED DOCTOR confirmation question.
    resolved_match = None
    doctor_already_shown = False
    # True only when THIS message itself names exactly one real doctor (not a
    # pronoun/"yes" recovered from an earlier turn) — see the narrowing
    # short-circuit below for why this alone is enough to narrow even without an
    # explicit "only"/"just" word.
    direct_name_match_this_turn = False
    # True only when a bare "yes" this turn is the confirmation of a doctor
    # resolved from an earlier message — see the narrowing short-circuit below.
    confirmed_via_affirmative = False
    if resolved_appointment is None:
        matches = find_doctors_by_name(db, ctx.clinic_id, message)
        if len(matches) > 1:
            return _disambiguation_marker_reply(matches)
        resolved_match = matches[0] if matches else None
        direct_name_match_this_turn = resolved_match is not None

        # Reported live: "dr ahmed rana in pulmonology dept" (no such doctor at this
        # clinic) silently showed the whole Pulmonology department instead — see
        # _extract_attempted_doctor_name's own comment. Only fires on a genuine
        # zero-match attempted name; never overrides an actual match above, and
        # never fires for a message with no "dr"/"doctor" name attempt at all (e.g.
        # "book me an appointment in Pulmonology" correctly falls through as before).
        if resolved_match is None:
            attempted_name = _extract_attempted_doctor_name(message)
            if attempted_name is not None:
                return (
                    f"I couldn't find a doctor named \"{attempted_name.title()}\" at this clinic. "
                    "Could you double-check the spelling, or let me know which department you'd "
                    "like to see instead?"
                )

        if (
            resolved_match is None
            and _is_short_affirmative_reply(message)
            and _preceding_assistant_turn_looks_like_a_question(history)
        ):
            # A bare "yes" to the assistant's own confirming question names no
            # doctor itself — re-resolve from the patient's own prior message(s)
            # instead of leaving the model with nothing and letting it re-ask
            # from scratch.
            #
            # Reported live: "hook me up with dr ali raza" (KB prose, no card) ->
            # "can u book an appointment for me with him?" (named no doctor
            # itself, just the pronoun "him" — resolved via the pronoun path
            # below from the FIRST message) -> "Did you mean Dr. Ali Raza in
            # General Medicine?" -> "yes". The single-most-recent-message lookup
            # this used to do only checked "can u book...him?" itself, which
            # names no doctor either, so this recovery silently found nothing —
            # resolved_match stayed None and the model got zero doctor context,
            # producing a generic "book online yourself" answer instead of
            # actually offering to show real slots. Reuses the same bounded
            # lookback _most_recently_named_doctor already does for a pronoun
            # reference, so it can see past an intermediate pronoun-only message
            # to the one that actually named the doctor.
            prior_match = _most_recently_named_doctor(db, ctx.clinic_id, history)
            if prior_match is not None:
                resolved_match = prior_match
                # The patient's "yes" IS the confirmation — treat exactly like
                # an already-shown card so the model proceeds directly instead
                # of asking the same confirming question a second time.
                doctor_already_shown = True
                confirmed_via_affirmative = True
        if resolved_match is None and _message_references_a_doctor_by_pronoun(message):
            # "show me his available slots on fri" names no doctor of its own, just
            # a pronoun — recover the real doctor (and their real department_name,
            # not a specialization or other prose the patient/KB text used) from
            # the patient's own earlier message. Still asks a confirming question
            # below (doctor_already_shown stays False) since, unlike the bare-"yes"
            # case above, the patient never actually confirmed this identity yet.
            resolved_match = _most_recently_named_doctor(db, ctx.clinic_id, history)
        if resolved_match is not None and not doctor_already_shown:
            doctor_already_shown = _doctor_already_shown(
                history, resolved_match.full_name, resolved_match.department_name
            )

    # A patient narrowing an already-shown multi-doctor card down to just one of
    # those doctors ("only show me Dr. X's slots") gets a real, filtered card built
    # directly from the DB — never routed through the LLM/tool-calling loop, which
    # has no way to filter get_department_availability's result at all.
    #
    # Reported live: "is dr farhan malik available on sat?" -> confirmed -> shown
    # his real Saturday slots -> "is he available on mon?" — "he" was correctly
    # resolved back to Dr. Farhan Malik (doctor_already_shown True), but this
    # message has no "only"/"just" narrowing word, so the short-circuit below
    # didn't fire and the LLM was left to call get_department_availability for
    # the whole department on Monday — which has no doctor-filtering option at
    # all, so it silently showed Dr. Ahmed Farooq's Monday slots instead of ever
    # answering the actual question asked ("is HE available"). Once a message
    # refers to a doctor only by pronoun, it can only be asking about that one
    # doctor specifically — treated the same as an explicit narrowing word.
    #
    # Reported live (2nd report): "show me available slots for dr farhan rehman"
    # after a two-doctor card had already shown him named no narrowing word at
    # all ("only"/"just"), so this short-circuit didn't fire even though a single
    # real doctor was directly, unambiguously named — the LLM was left to call
    # get_department_availability for the whole department again. Naming one
    # specific doctor by name and asking about their availability always means
    # just that doctor, never "and everyone else in their department too" — so a
    # direct name match this turn is now treated the same as an explicit
    # narrowing word (direct_name_match_this_turn).
    #
    # Reported live (3rd report): the far more common two-turn shape — "show me
    # available slots for dr farhan rehman" (first mention, unconfirmed) -> "Did
    # you mean Dr. Farhan Rehman in Family Medicine?" -> "yes" — fell through
    # this same gap from the other direction: "yes" names no doctor and has no
    # narrowing word of its own, so even after doctor_already_shown correctly
    # became True via the bare-affirmative recovery above, this short-circuit
    # still didn't fire, and the LLM's own get_department_availability call
    # showed the entire department again. The whole reason that confirming
    # question exists is to check availability for THAT one doctor — the "yes"
    # confirming it is never a request to broaden back out to the department, so
    # confirmed_via_affirmative is now treated the same way.
    #
    # Reported live (4th report): a doctor+department already shown, then "book
    # with dr shahid sheikh on sat aug 22 at 8 am" — a real, explicit slot pick
    # (doctor, date, AND time all named, plus the word "book") — matched
    # direct_name_match_this_turn and fired this short-circuit anyway, which
    # re-fetched and re-displayed the SAME availability card instead of ever
    # booking anything: this deterministic path has no way to resolve a
    # specific slot_id from natural language at all, only to filter/re-show
    # availability, so a message asking to actually BOOK a specific slot needs
    # the LLM/tool-calling path below instead — that's the one place
    # PREVIOUSLY SHOWN OPTIONS gets matched against the patient's wording to
    # find the real slot_id and call book_appointment (see this module's own
    # top-of-file docstring, step 2). Recognizing "book" here and stepping
    # aside is a narrow, additive exception — every other narrowing trigger
    # above is unaffected for a plain "show me his slots"/"yes" style message.
    if (
        resolved_match is not None
        and doctor_already_shown
        and resolved_appointment is None
        and not _message_has_explicit_slot_id(message)
        and not _message_expresses_a_specific_slot_pick_intent(message)
        and (
            _message_or_pending_name_disambiguation_asks_to_narrow(message, history)
            or _message_references_a_doctor_by_pronoun(message)
            or direct_name_match_this_turn
            or confirmed_via_affirmative
        )
    ):
        # Reuses the same bare-weekday resolution the LLM tool path relies on
        # (see resolve_bare_weekday_window's own docstring) — "is he available
        # on mon?" must be checked against Monday specifically, not just
        # whatever slots happen to be earliest.
        forced_window = resolve_bare_weekday_window(message)
        earliest_date = date.fromisoformat(forced_window[0]) if forced_window else None
        latest_date = date.fromisoformat(forced_window[1]) if forced_window else None
        # Reported live: "only show me available slots of dr farhan rehman
        # after 12 pm on monday" and "show me his available slots after 12
        # pm" both silently ignored the time-of-day request and returned the
        # same top-5-earliest-of-the-day slots — this was never resolved or
        # applied anywhere in this deterministic path at all.
        time_window = resolve_time_of_day_window(message)
        earliest_time, latest_time = time_window if time_window else (None, None)
        availability = get_department_availability(
            db,
            ctx.clinic_id,
            resolved_match.department_name,
            doctor_id=resolved_match.doctor_id,
            earliest_date=earliest_date,
            latest_date=latest_date,
            earliest_time=earliest_time,
            latest_time=latest_time,
        )
        if not availability.doctors:
            if availability.next_available_when is not None:
                # Named to the one doctor actually checked, not the whole
                # department — reusing the department-wide "no one in X" wording
                # here would misleadingly imply every doctor in the department
                # was checked for this window, when only resolved_match was.
                tz = _clinic_timezone(db, ctx.clinic_id)
                return (
                    f"{resolved_match.full_name} doesn't have any open slots in that window. "
                    f"Their earliest available slot is {_format_when(availability.next_available_when, tz)}. "
                    "Would you like me to book that instead, or check a different day?"
                )
            return f"{resolved_match.full_name} doesn't have any open slots right now. Please check back later."
        return _doctor_options_payload(db, ctx.clinic_id, availability)

    # Instructed live: booking now gets the same code-enforced confirm-then-act
    # gate as cancel/reschedule above. This is the fresh-booking counterpart —
    # resolved_appointment is None here (no cancel/reschedule action in play), and
    # the message carries an explicit slot_id from a shown DOCTOR_OPTIONS card's
    # slot-pick button. Composes a real, DB-backed confirming question via
    # get_slot_summary rather than letting the LLM call book_appointment
    # immediately. Falls through to the normal LLM/tool-calling path only if the
    # slot lookup fails (e.g. a stale/tampered slot_id).
    if resolved_appointment is None and _message_has_explicit_slot_id(message):
        picked_slot_id = _extract_slot_id(message)
        picked_slot = get_slot_summary(db, ctx.clinic_id, picked_slot_id) if picked_slot_id else None
        if picked_slot is not None:
            if _booking_cap_reached(db, ctx, picked_slot):
                return _booking_cap_reply(picked_slot["department_name"])
            return _book_confirmation_reply(picked_slot)

    # Handoff step 2: recover the most recent card (if any) and render it into the
    # prompt so the model can resolve a natural-language reference against it.
    marker_payload = _most_recent_availability_marker(history)

    language_name = llm._LANGUAGE_NAMES.get(language, "English")
    system_prompt = _build_system_prompt(
        language_name,
        marker_payload,
        resolved_match,
        doctor_already_shown,
        resolved_appointment,
        resolved_appointment_action,
    )

    reschedule_redirect_id = (
        resolved_appointment["appointment_id"]
        if resolved_appointment is not None and resolved_appointment_action == "reschedule"
        else None
    )
    cancel_redirect_id = (
        resolved_appointment["appointment_id"]
        if resolved_appointment is not None and resolved_appointment_action == "cancel"
        else None
    )

    forced_date_window = resolve_bare_weekday_window(message)
    forced_time_window = resolve_time_of_day_window(message)
    tools = [
        t
        for t in build_tools(
            db,
            ctx,
            reschedule_redirect_appointment_id=reschedule_redirect_id,
            cancel_redirect_appointment_id=cancel_redirect_id,
            forced_date_window=forced_date_window,
            forced_time_window=forced_time_window,
        )
        if t.name in _APPOINTMENT_AGENT_TOOL_NAMES
    ]
    return llm.run_tool_calling_agent(system_prompt, message, history, tools)
