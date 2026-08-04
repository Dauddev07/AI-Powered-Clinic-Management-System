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

from sqlalchemy.orm import Session

from app.core.tenancy import ClinicContext
from app.models.conversation_memory import ConversationMemory
from app.services import llm
from app.services.chat_markers import DEPARTMENT_LIST_MARKER, DOCTOR_DISAMBIGUATION_MARKER, DOCTOR_OPTIONS_MARKER
from app.services.chat_tools import build_tools, list_upcoming_appointments, resolve_bare_weekday_window
from app.services.department_availability import find_doctors_by_name
from app.services.message_classifier import _preceding_assistant_turn_looks_like_a_question

_APPOINTMENT_AGENT_TOOL_NAMES = {
    "book_appointment",
    "reschedule_appointment",
    "cancel_appointment",
    "get_my_appointments",
    "get_department_availability",
}

_INTRO = (
    "You are a clinic assistant chatbot for a hospital management system, helping "
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

_CANCEL_KEYWORDS = frozenset({"cancel", "cancellation", "cancelling", "canceling", "cancelled"})
_RESCHEDULE_KEYWORDS = frozenset({"reschedule", "rescheduling", "postpone", "postponing"})

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


def _most_recent_user_message(history: list[ConversationMemory]) -> str | None:
    for row in reversed(history):
        if getattr(row, "role", None) == "user":
            return getattr(row, "content", "") or None
    return None


def _detect_action_intent(message: str) -> str | None:
    """Cheap keyword check for a fresh cancel/reschedule request — deliberately
    narrow (unlike the broad, over-inclusive keyword lists elsewhere in this
    system): a false positive here would trigger the appointment-ambiguity handoff
    below for an unrelated message, and a false negative just falls through to the
    LLM exactly as before this handoff existed, so there's no safety asymmetry
    pushing this one broader."""
    words = set(re.findall(r"[a-z0-9']+", message.lower()))
    if words & _CANCEL_KEYWORDS:
        return "cancel"
    if words & _RESCHEDULE_KEYWORDS:
        return "reschedule"
    return None


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
    """True when this exact doctor+department already appeared in a real
    DOCTOR_OPTIONS_MARKER/DEPARTMENT_LIST_MARKER card shown earlier in `history` —
    scans every such card, not just the most recent one, since the card in question
    may be several turns back (e.g. a symptom-triage card from earlier the same
    conversation). Used to skip the RESOLVED DOCTOR confirming question when the
    patient is plainly just referencing something the assistant itself already told
    them, not naming a doctor cold."""
    target_name = doctor_name.strip().lower()
    target_department = department_name.strip().lower()
    for row in history:
        if getattr(row, "role", None) != "assistant":
            continue
        content = getattr(row, "content", "") or ""
        department_groups = None
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
        if not department_groups:
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
    names = ", ".join(f"{c['doctor_name']} ({c['department_name']}, {c['when']})" for c in candidates)
    question = f"You have more than one upcoming appointment — which one would you like to {verb}: {names}?"
    payload = {"kind": "appointment", "action": action, "question": question, "candidates": candidates}
    return DOCTOR_DISAMBIGUATION_MARKER + json.dumps(payload)


def _pending_appointment_disambiguation(history: list[ConversationMemory]) -> dict | None:
    """True only when the assistant's OWN last turn was
    _appointment_disambiguation_reply's own question — a genuine reply to it, not
    just any later message, so an unrelated message sent after being asked doesn't
    get misread as answering a stale question."""
    if not history:
        return None
    last = history[-1]
    if getattr(last, "role", None) != "assistant":
        return None
    content = getattr(last, "content", "") or ""
    if not content.startswith(DOCTOR_DISAMBIGUATION_MARKER):
        return None
    try:
        payload = json.loads(content[len(DOCTOR_DISAMBIGUATION_MARKER):])
    except (ValueError, TypeError):
        return None
    if payload.get("kind") != "appointment":
        return None
    return payload


def _match_candidate(message: str, candidates: list[dict]) -> dict | None:
    """Matches the patient's reply against exactly one candidate by doctor name
    (full name, or its last word as a plain surname reference like "the Sheikh
    one") — deliberately simple and exact rather than fuzzy, since an ambiguous or
    failed match should re-ask (see run_appointment_agent), never guess."""
    lowered = message.lower()
    matches = [
        c for c in candidates
        if c["doctor_name"].strip().lower() in lowered
        or c["doctor_name"].strip().split()[-1].lower() in lowered
    ]
    return matches[0] if len(matches) == 1 else None


def _build_system_prompt(
    language_name: str,
    patient_memory: str,
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
        action_sentence = (
            "This is a RESCHEDULE: once the patient picks a new time from the slots you "
            "show them, call reschedule_appointment with this exact appointment_id and the "
            "new slot_id — never book_appointment, even if their wording sounds like "
            'booking a fresh appointment (e.g. "I\'d like to book the appointment with Dr. '
            'X").\n\n'
            if resolved_appointment_action == "reschedule"
            else "Call cancel_appointment with this exact appointment_id.\n\n"
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
    tail = llm._TAIL_STYLE_AND_MEMORY_RULES.format(
        language_name=language_name,
        patient_memory=patient_memory.strip() if patient_memory and patient_memory.strip() else "(none)",
    )
    return (
        f"{_INTRO}\n\n"
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
    patient_memory: str = "",
) -> str:
    # Appointment-ambiguity handoff: resolved BEFORE the doctor-name handoff below,
    # so a reply naming a doctor to answer "which appointment?" (e.g. "the one with
    # Dr. Sheikh") resolves the APPOINTMENT, not a fresh RESOLVED DOCTOR confirmation
    # for that same doctor.
    resolved_appointment: dict | None = None
    resolved_appointment_action: str | None = None

    pending = _pending_appointment_disambiguation(history)
    if pending is not None:
        candidate = _match_candidate(message, pending["candidates"])
        if candidate is None:
            return _appointment_disambiguation_reply(pending["action"], pending["candidates"])
        resolved_appointment = candidate
        resolved_appointment_action = pending["action"]
    else:
        action = _detect_action_intent(message)
        if action is not None:
            active = list_upcoming_appointments(db, ctx)
            if len(active) > 1:
                name_matches = find_doctors_by_name(db, ctx.clinic_id, message)
                narrowed = (
                    [a for a in active if a["doctor_name"] == name_matches[0].full_name]
                    if len(name_matches) == 1
                    else []
                )
                if len(narrowed) == 1:
                    resolved_appointment = narrowed[0]
                    resolved_appointment_action = action
                else:
                    return _appointment_disambiguation_reply(action, active)
            elif len(active) == 1:
                resolved_appointment = active[0]
                resolved_appointment_action = action

    # Handoff step 1: deterministic ambiguity check against the real doctor table.
    # A message with no doctor-name reference at all (e.g. "cancel my appointment")
    # simply returns zero matches here and falls through normally below. Skipped
    # when the appointment-ambiguity handoff above already resolved this turn, so a
    # doctor name used only to answer "which appointment?" doesn't also trigger its
    # own separate RESOLVED DOCTOR confirmation question.
    resolved_match = None
    doctor_already_shown = False
    if resolved_appointment is None:
        matches = find_doctors_by_name(db, ctx.clinic_id, message)
        if len(matches) > 1:
            return _disambiguation_marker_reply(matches)
        resolved_match = matches[0] if matches else None
        if (
            resolved_match is None
            and _is_short_affirmative_reply(message)
            and _preceding_assistant_turn_looks_like_a_question(history)
        ):
            # A bare "yes" to the assistant's own confirming question names no
            # doctor itself — re-resolve from the patient's own prior message
            # (the one that actually named the doctor) instead of leaving the
            # model with nothing and letting it re-ask from scratch.
            prior_message = _most_recent_user_message(history)
            if prior_message:
                prior_matches = find_doctors_by_name(db, ctx.clinic_id, prior_message)
                if len(prior_matches) == 1:
                    resolved_match = prior_matches[0]
                    # The patient's "yes" IS the confirmation — treat exactly like
                    # an already-shown card so the model proceeds directly instead
                    # of asking the same confirming question a second time.
                    doctor_already_shown = True
        if resolved_match is not None and not doctor_already_shown:
            doctor_already_shown = _doctor_already_shown(
                history, resolved_match.full_name, resolved_match.department_name
            )

    # Handoff step 2: recover the most recent card (if any) and render it into the
    # prompt so the model can resolve a natural-language reference against it.
    marker_payload = _most_recent_availability_marker(history)

    language_name = llm._LANGUAGE_NAMES.get(language, "English")
    system_prompt = _build_system_prompt(
        language_name,
        patient_memory,
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
    tools = [
        t
        for t in build_tools(
            db,
            ctx,
            reschedule_redirect_appointment_id=reschedule_redirect_id,
            cancel_redirect_appointment_id=cancel_redirect_id,
            forced_date_window=forced_date_window,
        )
        if t.name in _APPOINTMENT_AGENT_TOOL_NAMES
    ]
    return llm.run_tool_calling_agent(system_prompt, message, history, tools)
