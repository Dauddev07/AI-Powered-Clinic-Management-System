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
TOOL USE RULES governing these two tools specifically + the shared style/patient-
memory rules — all sliced verbatim from the original single-pipeline
_AGENT_SYSTEM_PROMPT, never retyped. The original prompt's STRICT GROUNDING RULE and
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
from app.services.message_classifier import needs_path2_screening

_SYMPTOM_AGENT_TOOL_NAMES = {"get_department_availability", "find_doctors_by_name"}

_INTRO = (
    "You are a clinic assistant chatbot for a hospital management system, helping "
    "patients with symptom triage and department routing."
)


def _build_system_prompt(
    language_name: str, department_names: list[str], patient_memory: str, include_path2: bool
) -> str:
    context_line = (
        f"Active departments at this clinic: {', '.join(department_names)}."
        if department_names
        else "Active departments at this clinic: (none configured)."
    )
    tail = llm._TAIL_STYLE_AND_MEMORY_RULES.format(
        language_name=language_name,
        patient_memory=patient_memory.strip() if patient_memory and patient_memory.strip() else "(none)",
    )
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


# Reported live (2nd instance of the same underlying gap): a patient describing ear
# pain AND itchy skin, then confirming "chest pain mild, itchy skin from past 1 day",
# got a reply that routed to Cardiology and never mentioned Dermatology at all — not
# even in its own `note` this time, so _departments_named_in_note_but_not_covered
# above (which only scans the note's own text) has nothing to find. The model simply
# dropped the second complaint entirely rather than reasoning about it and failing to
# call the tool. This is the deeper backstop: map the symptom vocabulary actually
# used by the PATIENT (this message + their own earlier turns, never the assistant's)
# to common hospital-department-name substrings, and if a real active department at
# this clinic matches a symptom category that's still uncovered, fetch it too — same
# "don't trust the model's tool-calling completeness" principle, one level earlier
# than the note-scan above (before the model ever reasons about it, not just after).
_SYMPTOM_DEPARTMENT_HINTS: tuple[tuple[frozenset[str], tuple[str, ...], str], ...] = (
    (frozenset({"skin", "itchy", "itching", "rash", "allergic", "allergy", "eczema", "hives"}), ("derma",), "skin symptoms"),
    (frozenset({"ear", "earache", "hearing"}), ("ent", "otolaryn", "ear"), "ear pain"),
    (frozenset({"eye", "eyes", "vision"}), ("ophthal", "eye"), "eye symptoms"),
    (frozenset({"tooth", "teeth", "toothache", "dental"}), ("dent",), "tooth pain"),
    (
        frozenset({"bone", "fracture", "fractured", "sprain", "sprained", "joint", "dislocated", "dislocation"}),
        ("ortho",),
        "bone/joint injury",
    ),
    (frozenset({"chest", "heart", "palpitations", "cardiac"}), ("cardio",), "chest pain"),
    (
        frozenset({"stomach", "abdominal", "abdomen", "digestive", "diarrhea", "diarrhoea", "vomit", "vomiting"}),
        ("gastro", "internal medicine", "general medicine"),
        "stomach symptoms",
    ),
    (frozenset({"anxiety", "depression", "mental", "stress"}), ("psych",), "mental health symptoms"),
    (
        frozenset({"cough", "breathless", "wheeze", "wheezing", "lung", "respiratory"}),
        ("pulmon", "respiratory"),
        "respiratory symptoms",
    ),
    # Reported live: "fasting blood sugar 200+" plus "excessive thirst and frequent
    # urination" is the textbook symptom triad for diabetes — the model named the
    # actual condition in free text instead of calling the tool, which is exactly
    # what _reply_forced_to_a_department_when_the_model_diagnosed_instead_of_routing
    # below exists to catch, but it still needs a department hint to route to.
    (
        frozenset({
            "sugar", "diabetes", "diabetic", "thirst", "thirsty", "urination", "urinate",
            "urinating",
        }),
        ("endocrin", "internal medicine", "general medicine"),
        "blood sugar symptoms",
    ),
    # Reported live: "i have brain tumor" — a self-diagnosis claim naming a
    # suspected condition rather than raw symptoms — needs a department hint too,
    # same as the diabetes-triad case above.
    (frozenset({"brain", "tumor", "tumour", "cancer", "seizure", "seizures"}), ("neuro", "oncol"), "neurological symptoms"),
)


def _departments_hinted_by_patient_symptom_words(
    message: str, history: list[ConversationMemory], department_names: list[str], already_covered: set[str]
) -> dict[str, str]:
    """Returns {department_name: symptom_label} for every real, active,
    not-yet-covered department a symptom category the PATIENT mentioned points to —
    the label (e.g. "chest pain", "ear pain") lets callers compose a note that
    names the SPECIFIC symptom driving this department, rather than a generic
    "this could also be evaluated by X" with no reasoning attached to it at all.
    Reported live: "head pain and chest pain" routed correctly to both General
    Medicine and Cardiology, but Cardiology's note read as boilerplate ("this could
    also be evaluated by Cardiology") with no mention of chest pain specifically,
    while General Medicine's model-composed note did name the symptom — an
    inconsistent, half-explained reply. First matching category wins if a
    department's name happens to match more than one hint substring (rare)."""
    patient_texts = [message] + [
        getattr(row, "content", "") or "" for row in history if getattr(row, "role", None) == "user"
    ]
    words = set(re.findall(r"[a-z0-9]+", " ".join(patient_texts).lower()))
    hinted_substrings: dict[str, str] = {}
    for keywords, hints, label in _SYMPTOM_DEPARTMENT_HINTS:
        if words & keywords:
            for hint in hints:
                hinted_substrings.setdefault(hint, label)
    if not hinted_substrings:
        return {}
    # Reported live: "ear and neck pain" + "difficulty swallowing" (an ENT case,
    # nothing dental) also pulled in Dentistry — the "ear" keyword's hint substring
    # "ent" is a plain `in` substring check, and "ent" happens to occur inside
    # "dentistry" (d-ENT-istry) too. Every one of these hints is a department-name
    # PREFIX by design (e.g. "cardio" leads "Cardiology", "otolaryn" leads
    # "Otolaryngology") — anchoring the match to the start of a word, rather than
    # letting it match anywhere inside one, keeps "ent" matching "ENT" itself
    # without also matching mid-word inside unrelated names like "Dentistry".
    missing: dict[str, str] = {}
    for name in department_names:
        if name in already_covered:
            continue
        lowered_name = name.lower()
        for hint, label in hinted_substrings.items():
            if re.search(rf"\b{re.escape(hint)}", lowered_name):
                missing[name] = label
                break
    return missing


def run_symptom_agent(
    db: Session,
    ctx: ClinicContext,
    message: str,
    language: str,
    history: list[ConversationMemory],
    patient_memory: str = "",
) -> str:
    include_path2 = needs_path2_screening(message, history)
    department_names = list_active_department_names(db, ctx.clinic_id)
    language_name = llm._LANGUAGE_NAMES.get(language, "English")
    system_prompt = _build_system_prompt(language_name, department_names, patient_memory, include_path2)

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
        hinted = _departments_hinted_by_patient_symptom_words(message, history, department_names, set())
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
        for name, label in _departments_hinted_by_patient_symptom_words(
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
