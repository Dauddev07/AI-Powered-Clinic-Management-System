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
from app.services.message_classifier import (
    is_department_recommendation_request,
    is_symptom_message,
    needs_path2_screening,
)
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


def _no_symptom_described_yet(history: list[ConversationMemory]) -> bool:
    """True when nothing in `history` is itself a symptom-shaped message — i.e. the
    CURRENT message is effectively the first real symptom description of the
    session, even if `history` isn't literally empty. Reported live: "hey, my name
    is daud" -> greeting reply -> "i am having pain in my joints all along the
    body" — the PATH-3 backstop below used to require history to be completely
    empty, so this small-talk exchange (itself harmless, no symptom content at
    all) was enough to disqualify the backstop from firing on what was genuinely
    the patient's first-ever symptom message. Without the backstop, the LLM was
    free to skip the required tool call entirely and returned a long free-text
    disclaimer instead (never calling get_department_availability) — a failure
    shape that didn't trip the diagnosis-guard or faked-payload recovery nets
    either, since it named no specific diagnosis and faked no JSON, just gave
    advice-and-ask text. Small talk preceding a symptom must not count against
    this backstop the same way an earlier REAL symptom turn correctly does."""
    return not any(
        getattr(row, "role", None) == "user" and is_symptom_message(getattr(row, "content", "") or "")
        for row in history
    )


# Reported live: after Orthopedics was correctly resolved for leg pain (screened
# with 2 real questions), "i am also having pain in my eyes as well" and then
# "...in my ear as well" both skipped straight to a department card with ZERO
# clarifying questions — the PATH-3 backstop above deliberately only covers a
# session's FIRST-EVER symptom (see its own docstring: "a symptom raised for the
# first time LATER in an ongoing conversation is intentionally NOT covered here
# — the model's own MULTIPLE DISTINCT SYMPTOMS handling already has real
# conversation context to reason from"). That assumption is exactly what failed
# here — same "prompt instruction alone isn't reliable" lesson as every other
# backstop in this file, just for a case this file's own comments had previously
# argued didn't need one. Detects a genuinely NEW, not-yet-discussed symptom
# category (not just more detail about the one already being screened) by
# comparing what the CURRENT MESSAGE ALONE hints (history=[], so escalation
# words like "weight" that only mean something combined with an EARLIER turn's
# anatomy word correctly hint nothing by themselves) against what PRIOR history
# alone already hinted — anything in the former but not the latter is new.
def _introduces_a_new_symptom_category(
    message: str, history: list[ConversationMemory], department_names: list[str]
) -> bool:
    prior_hints = set(departments_hinted_by_patient_symptom_words("", history, department_names, set()))
    message_alone_hints = set(departments_hinted_by_patient_symptom_words(message, [], department_names, set()))
    return bool(message_alone_hints - prior_hints)


# Reported live: "pain in my chest and in testies as well" (mild, bearable, 2 days)
# got a reply that never actually called get_department_availability — the model
# free-texted a recommendation paragraph, then hand-typed a fragment that MIMICS the
# tool's own JSON payload shape ({"department_name": ..., "note": ...}) as plain
# text instead of a real tool call. No real doctor/slot data ever existed behind it,
# and only ONE department (missing the chest-pain->Cardiology hint) was named. The
# existing recovery net below only fired for diagnosis-language violations
# ("you have X") — this reply contains no diagnostic phrasing at all, so it slipped
# through untouched. This pattern (a literal `"department_name"` key typed in prose)
# is a reliable, low-false-positive signal that the model faked the tool's output
# instead of calling it, and is checked independently of the diagnosis-language rule.
_FAKED_TOOL_PAYLOAD_RE = re.compile(r'"department_name"\s*:', re.IGNORECASE)


def _looks_like_a_faked_tool_payload(reply: str) -> bool:
    return bool(_FAKED_TOOL_PAYLOAD_RE.search(reply))


# Reported live: "i am having pain in my joints all along the body" (a PATH-2-
# eligible message, so the PATH-3 "zero questions" backstop above deliberately
# doesn't apply here — that backstop assumes PATH 2 already reliably asks a real
# question, which is exactly what failed) got a long free-text disclaimer
# instead of either a real clarifying question or a tool call: a red-flag bullet
# list plus a permission-seeking "would you like me to help you find a doctor?"
# tail. It named no diagnosis and faked no JSON, so neither existing recovery
# trigger caught it either. A REAL single clarifying question in this system is
# always plain prose ending in "?", never a formatted list — a reply containing
# 2+ bullet/numbered lines is a reliable, low-false-positive structural signal
# that the model dumped generic advice instead of doing its actual job, checked
# independently of the other two triggers.
_LIST_LINE_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+", re.MULTILINE)


def _looks_like_an_advice_dump_instead_of_routing(reply: str) -> bool:
    return len(_LIST_LINE_RE.findall(reply)) >= 2


# Reported live (2nd instance of the same failure category): "issue in chewing any
# solid thing" got a reply in plain PROSE, not a bulleted list — "I recommend
# scheduling an appointment with a dentist or an oral-maxillofacial specialist" —
# so it slipped past the bullet-line detector above too, plus contained no
# diagnosis language and no faked JSON. A real clarifying question in this system
# NEVER names a specific type of doctor/specialist — it only ever asks about the
# symptom itself (severity, duration, associated signs) — so a non-marker reply
# that names one is a reliable, low-false-positive signal the model recommended
# in free text instead of actually calling the tool, regardless of prose shape.
_SPECIALIST_TITLE_WORDS = frozenset({
    "dentist", "cardiologist", "dermatologist", "neurologist", "psychiatrist",
    "orthopedist", "pediatrician", "paediatrician", "gynecologist", "gynaecologist",
    "ophthalmologist", "pulmonologist", "otolaryngologist", "physician", "specialist",
    "surgeon", "practitioner",
})
_SPECIALIST_TITLE_RE = re.compile(r"\b(?:" + "|".join(_SPECIALIST_TITLE_WORDS) + r")\b", re.IGNORECASE)


def _recommends_a_specialist_in_free_text(reply: str) -> bool:
    return bool(_SPECIALIST_TITLE_RE.search(reply))


def _patient_named_this_department(department_name: str, message: str, history: list[ConversationMemory]) -> bool:
    """True when the patient themselves (never the assistant) said this
    department's name somewhere in the conversation — used to tell "the patient
    explicitly asked for this department" apart from "the model picked this
    department on its own," since only the latter should ever get overridden by
    the hint table below."""
    patient_texts = [message] + [
        getattr(row, "content", "") or "" for row in history if getattr(row, "role", None) == "user"
    ]
    combined = " ".join(patient_texts).lower()
    return department_name.lower() in combined


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
    # deterministic backstop in this file. Two independent triggers, both gated by
    # the same three universal exceptions (a recommendation request, duration+
    # severity already given, a self-diagnosis claim):
    #   1. The session's first-ever symptom (_no_symptom_described_yet) — kept
    #      scoped to `not include_path2` since PATH 2 already reliably asks its
    #      own first question in THIS specific case.
    #   2. A genuinely NEW symptom category introduced LATER in an ongoing
    #      conversation (_introduces_a_new_symptom_category) — this one was
    #      PREVIOUSLY, deliberately excluded (the model's own "MULTIPLE DISTINCT
    #      SYMPTOMS" handling was assumed to reliably screen it, same as PATH 2
    #      above). Reported live: after Orthopedics was correctly resolved for leg
    #      pain, "i am also having pain in my eyes as well" and then "...in my
    #      ear as well" both skipped straight to a card with zero questions — that
    #      assumption failed too, so this trigger deliberately does NOT check
    #      `include_path2` at all (one of the two live-failing messages was
    #      itself PATH-2-eligible, so that guard wouldn't have helped anyway).
    no_symptom_yet = _no_symptom_described_yet(history)
    if (
        not is_department_recommendation_request(message)
        and not _message_already_gives_duration_and_severity(message)
        and not _is_self_diagnosis_claim(message)
        and (
            (not include_path2 and no_symptom_yet)
            # Guarded by `not no_symptom_yet` so this stays mutually exclusive
            # with trigger 1 above — without it, a brand-new session's FIRST
            # symptom would always "introduce a new category" too (nothing in
            # empty prior history to diff against), silently reapplying this
            # trigger to PATH-2-eligible first messages trigger 1 deliberately
            # exempts.
            or (not no_symptom_yet and _introduces_a_new_symptom_category(message, history, department_names))
        )
    ):
        return (
            "Could you tell me how severe this is (mild, moderate, or severe) and how "
            "long you've had it?"
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
    # Also covers the faked-tool-payload, advice-dump, and specialist-recommendation
    # cases above — same recovery, wider trigger.
    if not reply.startswith((DOCTOR_OPTIONS_MARKER, DEPARTMENT_LIST_MARKER, NO_SLOTS_MARKER)) and (
        violates_no_diagnosis_rule(reply)
        or _looks_like_a_faked_tool_payload(reply)
        or _looks_like_an_advice_dump_instead_of_routing(reply)
        or _recommends_a_specialist_in_free_text(reply)
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

        # Reported live: "pain in my legs" -> screened (mild, since morning) ->
        # explicitly denied swelling/difficulty bearing weight -> the model's OWN
        # tool call still named Orthopedics ("...without swelling or difficulty
        # bearing weight, Orthopedics can evaluate this") — a direct contradiction
        # of this exact symptom shape's own established rule (bare limb pain, no
        # injury signal, is General Medicine territory; see the limb/joint if/else
        # in symptom_hints.py). Every recovery net above only intercepts replies
        # that DON'T call the tool — this one did call it, just with the wrong
        # department, so none of them caught it. When the hint table has a
        # confident answer for what the patient actually said and the model's own
        # chosen department isn't even IN that answer, the deterministic table
        # wins: the primary card is rebuilt against the hinted department instead.
        # Skipped when the PATIENT themselves asked for this department by name
        # (see _patient_named_this_department) — that's a real, deliberate patient
        # choice, not the model freehanding one.
        hinted_for_primary = departments_hinted_by_patient_symptom_words(message, history, department_names, set())
        if (
            hinted_for_primary
            and covered_department not in hinted_for_primary
            and not _patient_named_this_department(covered_department, message, history)
        ):
            corrected_department, corrected_label = next(iter(hinted_for_primary.items()))
            reply = _get_department_availability_impl(
                db, ctx, corrected_department,
                note=f"Based on the {corrected_label} described, {corrected_department} would be appropriate.",
            )
            # The corrected department might have no open slots at all (a real,
            # different outcome shape — NO_SLOTS_MARKER, not DOCTOR_OPTIONS_MARKER)
            # — the "missing"/extra-department logic below only applies to the
            # doctor-options shape, so this returns as-is rather than crashing on
            # a payload that was never produced.
            if not reply.startswith(DOCTOR_OPTIONS_MARKER):
                return reply
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
