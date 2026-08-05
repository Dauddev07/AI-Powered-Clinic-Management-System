"""Symptom-triage specialist agent (app.services.orchestrator architecture).

No KB retrieval — reuses the existing logic exactly as it worked in the original
single-pipeline system: pulls the clinic's real active department names and passes
them as context, since there was never a reliable way to guarantee a KB document
covered every symptom/department combination (see
app.services.department_availability.list_active_department_names). Do not add a KB
collection or retrieval step here.

Tools bound: get_department_availability, find_doctors_by_name only.

System prompt = the triage rules (_TRIAGE_ALWAYS + conditional _TRIAGE_PATH2, reused
verbatim from app.services.llm/message_classifier.needs_path2_screening) + the
TOOL USE RULES governing these two tools specifically + the shared style rules —
all sliced verbatim from the original single-pipeline _AGENT_SYSTEM_PROMPT, never
retyped. The original prompt's STRICT GROUNDING RULE and
"`note` ALSO doubles as..." paragraphs are deliberately NOT included here: both are
about grounding an answer in KB-retrieved "Retrieved context," which this agent no
longer has (see the no-KB-retrieval note above) — a real, deliberate capability
change from the original single-pipeline behavior, not an oversight. A patient's
inline factual aside ("what's your address, also do you have anyone free in
Cardiology?") is no longer answered in the same reply by this agent; it needs its
own message, which the router sends to general_info_agent instead. Note-composition
for the ROUTING-REASONING purpose (explaining why a department was chosen) is
already fully self-contained inside _TRIAGE_ALWAYS's own tail rules and doesn't
depend on either dropped paragraph.
"""
import json
import re

from sqlalchemy.orm import Session

from app.core.tenancy import ClinicContext
from app.models.conversation_memory import ConversationMemory
from app.services import llm
from app.services.chat_markers import DEPARTMENT_LIST_MARKER, DOCTOR_OPTIONS_MARKER, NO_SLOTS_MARKER
from app.services.chat_tools import (
    _get_department_availability_impl,
    build_tools,
    combine_department_availability_results,
    resolve_bare_weekday_window,
)
from app.services.department_availability import list_active_department_names
from app.services.diagnosis_guard import violates_no_diagnosis_rule
from app.services.message_classifier import is_department_recommendation_request, needs_path2_screening
from app.services.orchestrator.symptom_hints import departments_hinted_by_patient_symptom_words

_SYMPTOM_AGENT_TOOL_NAMES = {"get_department_availability", "find_doctors_by_name"}

_INTRO = (
    "You are a clinic assistant chatbot for a hospital management system, helping "
    "patients with symptom triage and department routing."
)

# Mirrors PATH 3's own stated exception in llm.py ("if the patient's own message
# already gives a body part/area, duration, AND severity together, you may call
# the tool sooner"). A symptom keyword having already matched is what routed the
# message here at all (see message_classifier.is_symptom_message), so it stands
# in for "body part/area" — only duration and severity need checking here.
_DURATION_HINT_RE = re.compile(
    r"\b(day|days|week|weeks|month|months|hour|hours|year|years|since)\b", re.IGNORECASE
)
_SEVERITY_HINT_RE = re.compile(
    r"\b(mild|moderate|severe|bearable|unbearable|sharp|dull|intense|excruciating|slight)\b", re.IGNORECASE
)


def _message_already_gives_duration_and_severity(message: str) -> bool:
    return bool(_DURATION_HINT_RE.search(message) and _SEVERITY_HINT_RE.search(message))


# Self-diagnosis claims ("i have brain tumor", "i think i have cancer") are
# deliberately EXCLUDED from the first-message backstop below — the established,
# separately-reported fix for this category (see message_classifier._SYMPTOM_KEYWORDS'
# own comment) was that the model should respond with a CONCISE department redirect,
# not a longer structured breakdown. Asking "is that mild, moderate, or severe?" in
# response to "I have a brain tumor" reads as dismissive/deflecting for something a
# patient is already alarmed about, not genuinely clarifying — this category needs a
# fast redirect (handled by the LLM itself, backstopped by the diagnosis-recovery net
# below if it still free-texts the condition instead of calling the tool), not an
# extra screening question first. Same word list as _SYMPTOM_KEYWORDS' self-diagnosis
# entries, kept local here since this is a narrower, different purpose (excluding from
# a screening gate, not routing).
_SELF_DIAGNOSIS_WORDS = frozenset({"tumor", "tumour", "cancer", "diabetes", "diabetic"})


def _is_self_diagnosis_claim(message: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", message.lower()))
    return bool(words & _SELF_DIAGNOSIS_WORDS)


def _build_system_prompt(language_name: str, department_names: list[str], include_path2: bool) -> str:
    context_line = (
        f"Active departments at this clinic: {', '.join(department_names)}."
        if department_names
        else "Active departments at this clinic: (none configured)."
    )
    tail = llm._TAIL_STYLE_RULES.format(language_name=language_name)
    return (
        f"{_INTRO}\n\n"
        f"{llm._triage_section(include_path2)}\n"
        f"TOOL USE RULES:\n{llm._TOOL_RULES_SHARED}{llm._TOOL_RULES_FIND_DOCTORS_BY_NAME}\n"
        f"{tail}"
        f"Today's date is {llm._current_date_str()}.\n\n"
        f"{context_line}\n"
    )


def _departments_named_in_note_but_not_covered(
    note: str | None, department_names: list[str], already_covered: str
) -> list[str]:
    """Reported live: a patient describing two distinct complaints (ear pain + chest
    pain) got a reply whose own `note` correctly said "this could be evaluated by ENT
    for the ear pain and Cardiology for the chest pain" — the MULTIPLE DISTINCT
    SYMPTOMS/COMPLAINTS rule was followed in reasoning, but the model only actually
    called get_department_availability once (ENT), never for Cardiology, silently
    dropping the second department instead of calling the tool for it too. Same
    "prompt instruction alone isn't a reliable guarantee for a mutating/tool-calling
    step" pattern already fixed elsewhere in this codebase (reschedule-vs-book,
    cancel/reschedule disambiguation) — this is the equivalent deterministic
    safety net: scan the model's OWN note for any other real, active department name
    it already named but never actually queried, using word-boundary matching so a
    short department name can't spuriously match inside an unrelated word."""
    if not note:
        return []
    lowered = note.lower()
    missing = []
    for name in department_names:
        if name == already_covered:
            continue
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            missing.append(name)
    return missing


# Symptom-vocabulary -> department-name hint table moved to
# app.services.orchestrator.symptom_hints (see that module's own docstring) —
# appointment_agent needs the exact same mapping now, to check a directly-named
# department against what the patient's own symptoms actually support.


def run_symptom_agent(
    db: Session,
    ctx: ClinicContext,
    message: str,
    language: str,
    history: list[ConversationMemory],
) -> str:
    department_names = list_active_department_names(db, ctx.clinic_id)

    # Reported live: "isn't neurologist a better idea?" / "what do you think based on
    # my symptoms" got either a blind department switch with no reasoning, or a
    # free-text reply that HALLUCINATED a symptom never actually mentioned. A
    # recommendation request is answered ENTIRELY from the real symptom-to-department
    # mapping over what the patient has actually said — never handed to the LLM to
    # freehand a reason, same "never trust the model to compose grounded clinical
    # reasoning" principle as every other deterministic card in this system.
    #
    # Reported live (2nd instance): "I have fever and body aches, what do you
    # recommend?" — symptom description and the recommendation phrase in the SAME
    # message — short-circuited straight to a department, skipping the normal PATH 2
    # screening questions (severity/duration) that should always come first for a
    # symptom nobody has triaged yet. Scanning only PRIOR history (message="") fixes
    # this: a symptom mentioned for the first time in THIS message still goes through
    # the normal triage flow below; only a recommendation asked about symptoms
    # already established in earlier turns is answered deterministically here.
    if is_department_recommendation_request(message):
        hinted = departments_hinted_by_patient_symptom_words("", history, department_names, set())
        if hinted:
            results = [
                _get_department_availability_impl(
                    db, ctx, name, note=f"Based on the {label} you described, {name} would be a good fit."
                )
                for name, label in hinted.items()
            ]
            return results[0] if len(results) == 1 else combine_department_availability_results(results)
        # No real symptom described yet this session to base a recommendation on —
        # fall through to the normal triage flow below, which will ask what's wrong.

    include_path2 = needs_path2_screening(message, history)

    # DETERMINISTIC BACKSTOP for PATH 3's "never zero questions" rule — live testing
    # showed the model following this instruction most of the time, but not always:
    # two separate reports ("I am having nausea..", "pain in my jaw" — see
    # tests/test_orchestrator.py) each skipped straight to a department card with
    # ZERO clarifying questions on the very first message of a brand new session,
    # despite an explicit prompt paragraph (llm.py's PATH 3 section) specifically
    # calling out this exact shape of message as the most common way the rule gets
    # broken. A prompt instruction alone was not a reliable enough guarantee for
    # this correctness-critical step, same conclusion reached for every other
    # deterministic backstop in this file — so the highest-confidence case (a
    # symptom mentioned for the very first time in a brand new session) is now
    # enforced in code instead of hoped for. Deliberately narrow in scope: only
    # fires when history is completely empty (this really is message #1), PATH 2
    # doesn't already apply (that path already reliably asks first), the patient
    # hasn't already given duration+severity together (PATH 3's own stated
    # exception — genuine clarification already present, not a quota), and this
    # isn't itself a recommendation request (handled separately above). A symptom
    # raised for the first time LATER in an ongoing conversation is intentionally
    # NOT covered here — the model's own MULTIPLE DISTINCT SYMPTOMS handling
    # already has real conversation context to reason from in that case, which a
    # brand new session has none of.
    if (
        not include_path2
        and not history
        and not is_department_recommendation_request(message)
        and not _message_already_gives_duration_and_severity(message)
        and not _is_self_diagnosis_claim(message)
    ):
        return (
            "Could you tell me how severe this is (mild, moderate, or severe) and how "
            "long you've had it? Any other symptoms alongside it, like swelling, fever, "
            "or numbness?"
        )

    language_name = llm._LANGUAGE_NAMES.get(language, "English")
    system_prompt = _build_system_prompt(language_name, department_names, include_path2)

    forced_date_window = resolve_bare_weekday_window(message)
    tools = [
        t
        for t in build_tools(db, ctx, forced_date_window=forced_date_window)
        if t.name in _SYMPTOM_AGENT_TOOL_NAMES
    ]

    reply = llm.run_tool_calling_agent(system_prompt, message, history, tools)

    # Reported live: "fasting blood sugar 200+" then "excessive thirst and frequent
    # urination" (the textbook diabetes triad) got a free-text reply naming the
    # condition outright instead of calling get_department_availability — a real
    # double failure: PATH 3's "ALWAYS ENDS WITH THE TOOL CALL, NEVER FREE-TEXT
    # ADVICE INSTEAD" rule was skipped, AND the no-diagnosis rule was violated in
    # the same breath. app.services.diagnosis_guard would already strip a plain
    # diagnostic reply down to its generic "tell me more" redirect — correct for
    # safety, but it has no department to route to, so the patient's real turn
    # (they DID tell it their symptoms) is lost entirely, right back to square one.
    # This is the deterministic recovery: when the model diagnosed instead of
    # routing, use the same symptom-word hint table above to resolve a real
    # department from what the patient actually said and call the tool ourselves,
    # so the turn still ends with a real department/doctor option on the table —
    # the diagnostic free text itself is discarded, never shown to the patient.
    if not reply.startswith((DOCTOR_OPTIONS_MARKER, DEPARTMENT_LIST_MARKER, NO_SLOTS_MARKER)) and violates_no_diagnosis_rule(
        reply
    ):
        hinted = departments_hinted_by_patient_symptom_words(message, history, department_names, set())
        if hinted:
            results = [
                _get_department_availability_impl(
                    db, ctx, name, note=f"Based on the {label} described, this should be evaluated by a doctor."
                )
                for name, label in hinted.items()
            ]
            reply = results[0] if len(results) == 1 else combine_department_availability_results(results)

    if reply.startswith(DOCTOR_OPTIONS_MARKER):
        payload = json.loads(reply[len(DOCTOR_OPTIONS_MARKER):])
        covered_department = payload.get("department_name")
        original_note = payload.get("note")
        missing = _departments_named_in_note_but_not_covered(original_note, department_names, covered_department)
        # The model's own note already explains both departments by name in one
        # sentence (that's how this list was found) — reuse it verbatim rather than
        # inventing separate text, so the extra card explains itself instead of
        # showing up with no reasoning at all (see combine_department_availability_
        # results' own docstring for why this matters — a reason must always be
        # shown when routing was inferred from symptoms, never left blank).
        extra_notes = {name: original_note for name in missing}
        already_covered = {covered_department, *missing}
        # Names the SPECIFIC symptom driving this extra department (e.g. "the
        # chest pain could also be evaluated by Cardiology") instead of a
        # boilerplate "this could also be evaluated by X" with no reasoning
        # attached — reported live: in a head pain + chest pain case, General
        # Medicine's note named the symptom (model-composed) but Cardiology's
        # generic fallback note didn't, reading as an inconsistent, half-explained
        # reply between the two cards.
        for name, label in departments_hinted_by_patient_symptom_words(
            message, history, department_names, already_covered
        ).items():
            if name not in missing:
                missing.append(name)
                extra_notes[name] = f"Based on the {label} described, this could also be evaluated by {name}."
        if missing:
            extra_results = [
                _get_department_availability_impl(db, ctx, name, note=extra_notes[name]) for name in missing
            ]
            reply = combine_department_availability_results([reply, *extra_results])

    return reply
