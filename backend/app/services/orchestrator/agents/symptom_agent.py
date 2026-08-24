"""Symptom-triage specialist agent (app.services.orchestrator architecture).

No KB retrieval — reuses the existing logic exactly as it worked in the original
single-pipeline system: pulls the clinic's real active department names and passes
them as context, since there was never a reliable way to guarantee a KB document
covered every symptom/department combination (see
app.services.department_availability.list_active_department_names). Do not add a KB
collection or retrieval step here.

Tools bound: get_department_availability, find_doctors_by_name only.

System prompt = llm._OFF_TOPIC_AND_INTEGRITY_RULES (off-topic/general-knowledge
refusal + instruction-integrity, shared with appointment_agent — see llm.py's own
comment on why this is a separate constant from STRICT GROUNDING RULE below) + the
triage rules (_TRIAGE_ALWAYS + conditional _TRIAGE_PATH2, reused verbatim from
app.services.llm/message_classifier.needs_path2_screening) + the TOOL USE RULES
governing these two tools specifically + the shared style rules — all sliced
verbatim from the original single-pipeline _AGENT_SYSTEM_PROMPT, never retyped. The
original prompt's STRICT GROUNDING RULE and "`note` ALSO doubles as..." paragraphs
are deliberately NOT included here: both are about grounding an answer in
KB-retrieved "Retrieved context," which this agent no longer has (see the
no-KB-retrieval note above) — a real, deliberate capability change from the
original single-pipeline behavior, not an oversight. A patient's inline factual
aside ("what's your address, also do you have anyone free in Cardiology?") is no
longer answered in the same reply by this agent; it needs its own message, which
the router sends to general_info_agent instead. Note-composition for the
ROUTING-REASONING purpose (explaining why a department was chosen) is already
fully self-contained inside _TRIAGE_ALWAYS's own tail rules and doesn't depend on
either dropped paragraph.
"""
import json
import re
from dataclasses import dataclass
from types import SimpleNamespace

from sqlalchemy.orm import Session

from app.core.tenancy import ClinicContext
from app.models.conversation_memory import ConversationMemory
from app.services import llm
from app.services.chat_markers import DEPARTMENT_LIST_MARKER, DOCTOR_OPTIONS_MARKER, NO_SLOTS_MARKER
from app.services.chat_tools import (
    _get_department_availability_impl,
    build_tools,
    combine_department_availability_results,
    ensure_slot_pick_example_has_a_date,
    resolve_date_window,
    resolve_time_of_day_window,
)
from app.services.department_availability import list_active_department_names
from app.services.diagnosis_guard import violates_no_diagnosis_rule
from app.services.message_classifier import (
    is_department_recommendation_request,
    is_symptom_message,
    needs_path2_screening,
)
from app.services.orchestrator.symptom_hints import (
    departments_hinted_by_patient_symptom_words,
    symptom_words_with_no_matching_department,
    unsupported_symptom_labels,
)

_SYMPTOM_AGENT_TOOL_NAMES = {"get_department_availability", "find_doctors_by_name"}

_INTRO = (
    "You are Cura, a clinic assistant chatbot for a hospital management system, helping "
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

# Requested: a stated duration well past the normal severity-screening window
# (100 days, 50 days, "since childhood", etc.) should not change the normal
# triage questioning at all — severity is still asked exactly as usual if not
# already given — but by the time a real department/slots card is eventually
# shown, it should carry an extra note encouraging the patient to get seen
# soon even if it isn't severe, since something that's been going on this long
# is worth timely attention regardless of severity. See
# _append_long_duration_note below (mirrors _append_emergency_downgrade_safety_
# note's exact mechanism/scoping) and _session_stated_long_duration_for_the_
# current_symptom (mirrors _session_had_an_earlier_emergency_flag_for_the_
# current_symptom's exact same-symptom-episode tracking, including clearing on
# a genuinely new complaint).
_DURATION_NUMBER_RE = re.compile(
    r"\b(\d{1,4})\s*(day|days|week|weeks|month|months|year|years)\b", re.IGNORECASE
)
_DURATION_UNIT_TO_DAYS = {
    "day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30, "year": 365, "years": 365,
}
_LONG_DURATION_DAY_THRESHOLD = 5

_CHRONIC_DURATION_PHRASE_RE = re.compile(
    r"\bsince\s+(?:childhood|birth|i\s+was\s+(?:a\s+)?(?:child|kid|little|young))\b"
    r"|\b(?:for|since)\s+(?:many|several)\s+years\b"
    r"|\ba\s+long\s+time\b"
    r"|\bmy\s+(?:whole|entire)\s+life\b"
    r"|\blifelong\b",
    re.IGNORECASE,
)


def _message_states_long_duration(message: str) -> bool:
    if _CHRONIC_DURATION_PHRASE_RE.search(message):
        return True
    match = _DURATION_NUMBER_RE.search(message)
    if match is None:
        return False
    count = int(match.group(1))
    unit_days = _DURATION_UNIT_TO_DAYS[match.group(2).lower()]
    return count * unit_days > _LONG_DURATION_DAY_THRESHOLD


# "no no its not that long, its been 2/3/4 days" — the patient walking back an
# earlier long-duration statement. Deliberately its own check rather than
# reusing _message_states_long_duration's own negative — a plain SHORT duration
# stated later in the SAME episode without any explicit correction wording
# (e.g. answering a differentiator question with "2 days" for an unrelated
# reason) is not the same signal as the patient actively correcting themselves,
# so this only fires on the correction phrasing itself.
_DURATION_CORRECTION_RE = re.compile(
    r"\bnot\b.{0,20}\b(?:that|so)\s+long\b"
    r"|\bisn'?t\b.{0,20}\b(?:that|so)\s+long\b",
    re.IGNORECASE,
)


def _message_corrects_long_duration(message: str) -> bool:
    return bool(_DURATION_CORRECTION_RE.search(message))
# "frequent"/"frequently"/"excessive"/"excessively" added alongside the mild/
# moderate/severe words above — how OFTEN or how MUCH a symptom occurs is a real
# severity signal in its own right (e.g. "frequent urination", "excessive
# thirst"), so a message already describing it this way should skip the severity
# question the same way one already saying "mild"/"severe" does. Scoped narrowly
# to just these two word families, not every frequency-adjacent word.
_SEVERITY_HINT_RE = re.compile(
    r"\b(mild|moderate|severe|bearable|unbearable|sharp|dull|intense|excruciating|slight"
    r"|frequent|frequently|excessive|excessively)\b",
    re.IGNORECASE,
)

# Narrow subset of _SEVERITY_HINT_RE that unambiguously means "already at
# emergency severity" per llm.py's own PATH 1 wording ("SEVERITY ALREADY
# STATED... SKIPS THE SEVERITY QUESTION: ... stated 'severe' goes to PATH
# 1") — used below to stop the PATH-3 "never zero questions" backstop from
# forcing a clarifying question onto a message that should get zero further
# questions and go straight to the emergency reply instead. Deliberately
# excludes "mild"/"moderate"/"bearable"/"slight"/"sharp"/"dull"/"intense" —
# those genuinely still need the duration follow-up this backstop exists to
# guarantee; only wording that reads as already-at-emergency skips it.
_HIGH_SEVERITY_HINT_RE = re.compile(r"\b(severe|unbearable|excruciating)\b", re.IGNORECASE)

# Requested: "i am continuously thirsty"/"i have continuous thirst" must be
# treated as severity-already-given the same way "excessive thirst"/"frequent
# thirst" already are via _SEVERITY_HINT_RE above — only duration should be
# asked, never severity. Deliberately NOT added to _SEVERITY_HINT_RE itself
# (which applies to every symptom, not just thirst): "continuous"/"constant"
# describe a DURATION pattern (always present vs. comes-and-goes), not
# intensity, for most symptoms — symptom_hints.py's own low-mood-persistence-
# words comment already draws this exact distinction ("constant"/"persistent"/
# "ongoing" are pure duration modifiers with zero severity meaning alone).
# Broadening the general regex would wrongly skip the severity question for
# something like "continuous headache" too. Scoped narrowly to thirst instead,
# where "continuous"/"constant" genuinely is the clinically relevant
# diabetes-triad framing (alongside "frequent"/"excessive", which already work).
_THIRST_WORDS = frozenset({"thirst", "thirsty"})
_THIRST_PERSISTENCE_WORDS = frozenset({"continuous", "continuously", "constant", "constantly"})


def _message_states_persistent_thirst(message: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", message.lower()))
    return bool(words & _THIRST_WORDS and words & _THIRST_PERSISTENCE_WORDS)


def _message_already_gives_duration_and_severity(
    message: str, history: list[ConversationMemory] | None = None
) -> bool:
    return bool(
        _DURATION_HINT_RE.search(message)
        and (
            _SEVERITY_HINT_RE.search(message)
            or _message_states_a_numeric_vital_reading(message)
            or _extract_stated_pain_scale_number(message, history or []) is not None
            or _message_states_persistent_thirst(message)
        )
    )


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


# Reported live: "i think i have a brain tumour" correctly got a routine
# department card (per this module's own established design above — a
# self-diagnosis claim should get a concise redirect, never treated as an
# automatic emergency), but "i have a brain tumour" — the exact same claim,
# just without the "i think" hedge — got a full PATH 1 emergency reply
# instead. Nothing in the prompt actually distinguishes stated-as-fact from
# hedged self-diagnosis language the way it explicitly does for major-bone
# fractures (see EXCEPTION — A MAJOR/WEIGHT-BEARING BONE... in llm.py); the
# model's own "obviously severe description" judgment was simply inconsistent
# between the two phrasings for the same underlying claim. Scoped narrowly to
# a BARE claim — nothing else describing severity or a real red-flag sign
# alongside it — so a self-diagnosis claim genuinely accompanied by severe
# pain, bleeding, a seizure, etc. still gets to go through PATH 1 normally;
# only the claim-alone shape gets forced to the same deterministic outcome
# regardless of which way the hedge word happens to tip the model.
_OTHER_EMERGENCY_SIGNAL_WORDS = frozenset({
    "severe", "unbearable", "extreme", "worst", "bleeding", "seizure", "seizures",
    "unconscious", "confusion", "confused", "vomiting", "collapse", "collapsed",
    "faint", "fainted", "fainting", "numb", "numbness", "paralysis", "paralyzed",
})


def _bare_self_diagnosis_claim(message: str) -> bool:
    if not _is_self_diagnosis_claim(message):
        return False
    if _SEVERITY_HINT_RE.search(message):
        return False
    words = set(re.findall(r"[a-z0-9]+", message.lower()))
    return not (words & _OTHER_EMERGENCY_SIGNAL_WORDS)


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


def _preceding_assistant_turn_asked_a_screening_question(history: list[ConversationMemory]) -> bool:
    """True when the last assistant turn was a plain free-text clarifying question
    (severity/duration, body location, associated symptoms, temperature, etc.) —
    as opposed to a structured DOCTOR_OPTIONS/DEPARTMENT_LIST/NO_SLOTS card, which
    means something else entirely (awaiting a slot pick, not more symptom detail).
    Used by _introduces_a_new_symptom_category below to tell "the patient is
    answering the question we just asked" apart from "the patient is volunteering
    an unprompted new complaint."""
    if not history:
        return False
    last = history[-1]
    if getattr(last, "role", None) != "assistant":
        return False
    content = (getattr(last, "content", "") or "").strip()
    if not content or content.startswith((DOCTOR_OPTIONS_MARKER, DEPARTMENT_LIST_MARKER, NO_SLOTS_MARKER)):
        return False
    # Reported live: this returned False for a turn that WAS genuinely a
    # question ("Do you have any fever, difficulty swallowing, or swollen
    # glands?"), purely because _append_emergency_downgrade_safety_note (see
    # its own docstring) appends its reminder sentence AFTER the question
    # text whenever an earlier emergency reply happened earlier in this same
    # symptom episode — moving the string's true trailing character from "?"
    # to the note's own closing period. The very next reply ("yes a bit of
    # fever and difficulty swallowing" — a direct answer to that exact
    # question) then got silently misread as an unprompted new complaint
    # instead of an answer, re-triggering the "never zero questions"
    # backstop's severity/duration question from scratch as if screening had
    # never started. Only happens once an earlier emergency reply exists in
    # the session — with none, no note gets appended, "?" stays the true
    # trailing character, and this already worked correctly, matching
    # exactly what was reported live. Strip that specific, known suffix
    # before checking what the question itself actually ended with.
    if content.endswith(_EMERGENCY_DOWNGRADE_SAFETY_NOTE):
        content = content[: -len(_EMERGENCY_DOWNGRADE_SAFETY_NOTE)].rstrip()
    return content.endswith("?") or content.endswith("؟")


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
#
# Reported live (2nd report): "im sick" -> asked severity/duration -> answered
# ("3 days, mild") -> asked "where exactly / what type of problem" -> answered
# "fever, body ache" -> asked severity/duration AGAIN -> re-answered the same
# thing -> asked "any other symptoms (cough, sore throat...)" -> answered "sore
# throat" -> asked severity/duration a THIRD time. Each answer happened to hint a
# department the prior turns hadn't (fever/ache -> General Medicine, then sore
# throat -> ENT), so this function correctly detected a "new category" each
# time — but the patient wasn't volunteering an unprompted new complaint, they
# were directly answering the assistant's own immediately-preceding question.
# Severity/duration answered once for the overall episode shouldn't be re-asked
# just because a later ANSWER happens to touch a different department's
# keywords. Guarded by checking whether the preceding assistant turn was itself
# a screening question first: if so, this message is elaboration on an existing,
# already-screened complaint, not a new one, regardless of what it hints.
def _introduces_a_new_symptom_category(
    message: str, history: list[ConversationMemory], department_names: list[str]
) -> bool:
    if _preceding_assistant_turn_asked_a_screening_question(history):
        return False
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


# Reported live: "i am having pain in my leg" -> screened (severe, 5 days) -> a
# PATH 1 emergency reply -> patient declined ER -> "i want to book an
# appointment here" -> the model asked "which department...(e.g. Orthopedics
# or a general physician)" in free text instead of calling the tool with the
# department the hint table already supports from the patient's own words
# (severity+duration were already established). This is the product-level
# behavior this clinic wants: once real symptom info exists, always SHOW the
# actual department card, never ask the patient to pick or free-text a
# guess — the chatbot's whole job is to compute this, not defer it back to the
# patient. Only a genuinely blank hint table (nothing usable yet) should ever
# produce a real clarifying question, and that path is handled separately by
# the screening backstops above, before the LLM is even called.
_DEPARTMENT_QUESTION_RE = re.compile(
    r"which (?:department|dept|doctor|specialist)|let (?:you\s+)?(?:me\s+)?know which",
    re.IGNORECASE,
)


def _asks_the_patient_to_pick_a_department(reply: str) -> bool:
    return bool(_DEPARTMENT_QUESTION_RE.search(reply))


# Reported live (same conversation, next turn): "i dont know what dept can be
# best fir for it" got "the Orthopedics department is usually the most
# appropriate" — free-text naming a REAL department by its exact name, not a
# specialist title (so _recommends_a_specialist_in_free_text above doesn't
# catch it), without ever calling get_department_availability. This directly
# contradicted the hint table's own answer for bare "leg pain, no injury
# signal" (General Medicine, see symptom_hints.py's limb/joint if/else) — the
# LATER symptom-vs-department mismatch check in appointment_agent.py then
# fired against the bot's OWN prior recommendation, reading as the bot
# confusing itself. Since a real clarifying question in this system never
# names a specific department (same principle as the specialist-title
# detector above), a non-marker reply that does is a reliable signal the
# model free-texted a recommendation instead of calling the tool.
def _names_a_real_department_in_free_text(reply: str, department_names: list[str]) -> bool:
    lowered = reply.lower()
    return any(re.search(rf"\b{re.escape(name.lower())}\b", lowered) for name in department_names)


# Reported live: a genuine PATH 1 emergency reply ("chest pain is mild and
# accompanied my sweating" — a real emergency-consistent combination the model
# correctly caught via the EMERGENCY BACKSTOP secondary layer, since no literal
# "severe" or other word-level trigger fired) got silently discarded and replaced
# with a Cardiology booking card — the model did EXACTLY what PATH 1 requires (a
# one-sentence emergency statement, no tool call, first-aid steps formatted as a
# numbered "1) ... 2) ..." list per the STRUCTURE RULE), but that numbered-list
# formatting is indistinguishable from _looks_like_an_advice_dump_instead_of_
# routing's own trigger (2+ list-formatted lines) — every correctly-formatted
# PATH 1 reply necessarily trips it. "1122" is the one thing PATH 1 always
# contains and nothing else in this prompt ever produces (it's this clinic's
# specific emergency number, explicitly never substituted for a different
# country's), so checking for it first is a reliable, low-false-positive way to
# recognize "this is a valid emergency reply, don't touch it" before any of the
# other recovery triggers below get a chance to override it.
_VALID_EMERGENCY_REPLY_RE = re.compile(r"\b1122\b")


def _looks_like_a_valid_emergency_reply(reply: str) -> bool:
    return bool(_VALID_EMERGENCY_REPLY_RE.search(reply))


# Reported live: "very severe" headache -> PATH 1 emergency reply -> "i dont
# wanna goto er" (a REFUSAL of the action — correctly got the urgency
# reiterated) -> "but its not severe" (a CORRECTION of the earlier severity
# claim, a completely different kind of message) -> the model kept looping on
# a THIRD variation of the same emergency wording instead of ever accepting
# the correction and returning to normal screening. Nothing forced this
# transition — it was left entirely to the model's own in-context judgment,
# and it simply didn't make the call reliably. Deliberately narrow wording
# (mild/moderate/"not severe"/"not that bad") so a refusal-to-go message
# ("i dont wanna goto er" — no severity word in it at all) never matches this.
_SEVERITY_DOWNGRADE_RE = re.compile(
    r"\b(mild|moderate)\b"
    r"|\bnot\b.{0,20}\bsevere\b"
    r"|\bisn'?t\b.{0,20}\bsevere\b"
    r"|\bnot\s+that\s+bad\b",
    re.IGNORECASE,
)


# Requested: move the "map a 0-10 pain-scale number to mild/moderate/severe,
# never re-ask, never repeat the same emergency reply on a downward correction"
# behavior out of the system prompt and into a code-level check, same "never
# trust the LLM to reason about a numeric threshold on its own, hand it the
# real mapping" principle as the fever/BP/blood-sugar fixes above. The prompt
# still tells the model never to GENERATE a 0-10 scale question in the first
# place (that's a generation-time wording choice with no code-level
# equivalent) — this only covers the reactive half: the patient answering with
# a number anyway.
_PAIN_SCALE_QUESTION_RE = re.compile(
    r"scale of\s*0\s*to\s*10|rate\b.{0,25}\bpain\b|\bout of 10\b", re.IGNORECASE
)


def _preceding_assistant_turn_asked_a_numeric_pain_scale_question(history: list[ConversationMemory]) -> bool:
    if not history:
        return False
    last = history[-1]
    if getattr(last, "role", None) != "assistant":
        return False
    return bool(_PAIN_SCALE_QUESTION_RE.search(getattr(last, "content", "") or ""))


# An explicit "6/10"/"6 out of 10" is unambiguous on its own, trusted regardless
# of what the preceding turn asked. A BARE number ("6", "a 6") is only trusted
# as a pain-scale answer when the preceding turn actually asked a scale-shaped
# question — otherwise a bare "6" could mean anything (a date, a quantity) and
# must not be misread as severity.
_EXPLICIT_PAIN_SCALE_RE = re.compile(r"\b(\d{1,2})\s*(?:/\s*10|out of\s*10)\b", re.IGNORECASE)
_BARE_NUMBER_ONLY_RE = re.compile(r"^\s*(?:it'?s\s+|its\s+|a\s+)*(\d{1,2})\s*\.?\s*$", re.IGNORECASE)
# "not 6 but 5", "make that a 4" — a correction to a previously-stated pain-scale
# number. Checked independently of the preceding-question gate above since the
# correction phrasing itself is unambiguous regardless of what was asked.
_PAIN_SCALE_CORRECTION_RE = re.compile(
    r"\bnot\s+\d{1,2}\b.{0,15}\bbut\s+(\d{1,2})\b|\bmake\s+(?:that|it)\s+a\s+(\d{1,2})\b", re.IGNORECASE
)


def _extract_stated_pain_scale_number(message: str, history: list[ConversationMemory]) -> int | None:
    match = _EXPLICIT_PAIN_SCALE_RE.search(message)
    if match is None and _preceding_assistant_turn_asked_a_numeric_pain_scale_question(history):
        match = _BARE_NUMBER_ONLY_RE.match(message)
    if match is None:
        return None
    value = int(match.group(1))
    return value if 0 <= value <= 10 else None


def _extract_pain_scale_correction(message: str) -> int | None:
    match = _PAIN_SCALE_CORRECTION_RE.search(message)
    if match is None:
        return None
    value = int(match.group(1) or match.group(2))
    return value if 0 <= value <= 10 else None


def _pain_scale_number_to_severity_word(value: int) -> str:
    if value <= 3:
        return "mild"
    if value <= 6:
        return "moderate"
    return "severe"


_PAIN_SCALE_SEVERITY_INSTRUCTION_TEMPLATE = (
    "\n\nIMPORTANT — STATED PAIN-SCALE NUMBER: the patient's message states a pain level "
    "of {value}/10. Map this yourself: 0-3 = mild, 4-6 = moderate, 7-10 = severe. Treat it "
    "exactly as if the patient had said \"{word}\" — do not ask \"is it mild, moderate, or "
    "severe\" again for this; proceed with screening/routing from that word."
)


def _message_downgrades_severity(message: str) -> bool:
    if _SEVERITY_DOWNGRADE_RE.search(message):
        return True
    corrected = _extract_pain_scale_correction(message)
    return corrected is not None and _pain_scale_number_to_severity_word(corrected) != "severe"


def _last_reply_was_an_emergency_reply(history: list[ConversationMemory]) -> bool:
    if not history:
        return False
    last = history[-1]
    if getattr(last, "role", None) != "assistant":
        return False
    return _looks_like_a_valid_emergency_reply(getattr(last, "content", "") or "")


# Appended to the system prompt ONLY for the one turn where this correction is
# detected — an explicit, unambiguous instruction rather than trusting the
# model to infer this nuance on its own from raw history, the same
# "never trust a prompt-only inference for something this consequential"
# principle used throughout this module.
_SEVERITY_DOWNGRADE_INSTRUCTION = (
    "\n\nIMPORTANT — THE PATIENT JUST CORRECTED THEIR OWN EARLIER SEVERITY CLAIM: their "
    "last message states this is NOT severe (mild/moderate) after an earlier 'severe' "
    "answer already triggered PATH 1. Accept this correction. Do NOT repeat the "
    "emergency/ER message again. Continue normal screening from here exactly as PATH 2/3 "
    "specify, treating the current stated severity as mild/moderate — ask about duration "
    "and any other relevant differentiator, or proceed to PATH 3 and call "
    "get_department_availability if enough is already known."
)


# Reported live: "fever with 103" got treated as an emergency (PATH 1), but per the
# clinic's own fever guidance table a reading of 103°F is still "high fever," not an
# emergency on its own — only below 97°F (hypothermia-range) or above 103°F ("very
# high") should read as fever-related emergency severity by itself. Nothing in the
# prompt gave the model an actual numeric table to reason from, so it fell back to
# its own general-knowledge judgment call, which skewed too conservative for a stated
# number. Scoped to a NUMBER actually stated alongside "fever"/"temperature" in the
# CURRENT message — a bare "i have high fever" with no number is unaffected and still
# goes through PATH 2's normal fever screening as before.
_FEVER_NUMBER_BEFORE_RE = re.compile(
    r"\b(\d{2,3}(?:\.\d)?)\s*(?:°\s*f\.?|deg(?:rees?)?\s*f\.?|f\.?)?\s*(?:degrees?\s*)?"
    r"(?:fever|temp(?:erature)?)\b",
    re.IGNORECASE,
)
_FEVER_NUMBER_AFTER_RE = re.compile(
    r"\b(?:fever|temp(?:erature)?)\b(?:\s+\w+){0,2}?\s*"
    r"(\d{2,3}(?:\.\d)?)\s*(?:°\s*f\.?|deg(?:rees?)?\s*f\.?|f\b)?",
    re.IGNORECASE,
)


def _extract_stated_fever_temperature(message: str) -> float | None:
    match = _FEVER_NUMBER_BEFORE_RE.search(message) or _FEVER_NUMBER_AFTER_RE.search(message)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    # Sanity bound to plausible human body-temperature readings in Fahrenheit —
    # rejects an unrelated number ("fever for 3 days" has no digit here anyway,
    # but this guards against something like "temp 5" from a different context).
    if not (90.0 <= value <= 112.0):
        return None
    return value


def _fever_temperature_is_within_the_clinic_safe_range(temp: float) -> bool:
    return 97.0 <= temp <= 103.0


_FEVER_SAFE_RANGE_INSTRUCTION = (
    "\n\nIMPORTANT — STATED FEVER TEMPERATURE, CLINIC GUIDANCE TABLE: the patient's message "
    "states a specific fever/temperature reading. This clinic's own guidance table is: "
    "below 97°F = emergency-level low; 97-99°F = normal; 99-100.9°F = mild fever; "
    "101-103°F = high fever; above 103°F = emergency-level very high. A reading between "
    "97°F and 103°F (inclusive) — mild OR high fever — is NOT an emergency by itself, no "
    "matter how high it feels to the patient. Do NOT go to PATH 1 for this reading alone. "
    "Screen it normally per PATH 2 (how long, and any stiff neck/rash/confusion/"
    "breathlessness/persistent vomiting alongside it) and only escalate to PATH 1 if a "
    "genuine red-flag symptom is present alongside the fever, never purely because of the "
    "number itself."
)


# Requested: a stated blood pressure or blood sugar READING is itself the severity —
# the clinic's own guidance tables classify it directly from the numbers, so the
# patient should never be asked "is it mild, moderate, or severe" for these; only
# duration is still worth asking if it isn't already given. Same "never trust the
# LLM to reason about a clinical threshold on its own, always hand it the real
# table" principle as the fever fix above.
_BP_READING_RE = re.compile(r"\b(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b", re.IGNORECASE)


def _extract_stated_blood_pressure(message: str) -> tuple[int, int] | None:
    match = _BP_READING_RE.search(message)
    if match is None:
        return None
    try:
        systolic, diastolic = int(match.group(1)), int(match.group(2))
    except ValueError:
        return None
    # Sanity bounds to plausible human BP readings — rejects an unrelated ratio-
    # shaped fragment (a date, a fraction) from being misread as a BP reading.
    if not (60 <= systolic <= 260 and 30 <= diastolic <= 200):
        return None
    return systolic, diastolic


def _blood_pressure_category(systolic: int, diastolic: int) -> str:
    if systolic >= 180 or diastolic >= 120:
        return "hypertensive crisis"
    if systolic >= 140 or diastolic >= 90:
        return "very high"
    if systolic >= 130 or diastolic >= 80:
        return "high"
    if systolic >= 120:
        return "elevated"
    if systolic < 90 or diastolic < 60:
        return "low"
    return "normal"


_BLOOD_PRESSURE_SEVERITY_INSTRUCTION = (
    "\n\nIMPORTANT — STATED BLOOD PRESSURE READING, CLINIC GUIDANCE TABLE: the patient's "
    "message states a specific blood pressure reading (systolic/diastolic). This clinic's "
    "own guidance table is: below 90/60 = low; below 120/80 = normal; 120-129 systolic "
    "(diastolic still below 80) = elevated; 130-139 systolic or 80-89 diastolic = high; "
    "140+ systolic or 90+ diastolic = very high; 180+ systolic and/or 120+ diastolic = "
    "hypertensive crisis. The reading itself IS the severity — do NOT ask the patient 'is "
    "it mild, moderate, or severe' for this; work out the category yourself from the "
    "numbers using this table instead. Only ask about duration if it isn't already stated."
)

_BLOOD_SUGAR_NUMBER_BEFORE_RE = re.compile(
    r"\b(\d{2,3})\s*(?:mg\s*/?\s*dl)?\s*(?:blood\s*)?(?:sugar|glucose)\b", re.IGNORECASE
)
_BLOOD_SUGAR_NUMBER_AFTER_RE = re.compile(
    r"\b(?:blood\s*)?(?:sugar|glucose)\b(?:\s+\w+){0,3}?\s*(\d{2,3})\s*(?:mg\s*/?\s*dl)?", re.IGNORECASE
)


def _extract_stated_blood_sugar(message: str) -> int | None:
    match = _BLOOD_SUGAR_NUMBER_BEFORE_RE.search(message) or _BLOOD_SUGAR_NUMBER_AFTER_RE.search(message)
    if match is None:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    # Sanity bound to plausible human blood-sugar readings in mg/dL.
    if not (30 <= value <= 600):
        return None
    return value


_POSTPRANDIAL_CONTEXT_RE = re.compile(
    r"\bafter\s+(?:a\s+)?(?:meal|eating|food)\b|\bpost[\s-]?(?:meal|prandial)\b|\b2\s*hours?\s*after\b",
    re.IGNORECASE,
)
_FASTING_CONTEXT_RE = re.compile(
    r"\bfasting\b|\bbefore\s+(?:a\s+)?(?:meal|eating|food)\b|\bempty\s+stomach\b", re.IGNORECASE
)


def _blood_sugar_category(value: int, message: str) -> str:
    # Ambiguous readings (no fasting/post-meal context stated) default to the
    # fasting thresholds — the more conservative of the two tables (a lower bar
    # for "diabetes range"), rather than assuming the more permissive one.
    is_postprandial = bool(_POSTPRANDIAL_CONTEXT_RE.search(message)) and not _FASTING_CONTEXT_RE.search(message)
    if is_postprandial:
        if value < 70:
            return "low"
        if value < 140:
            return "normal"
        if value < 200:
            return "high/prediabetes"
        return "diabetes range"
    if value < 70:
        return "low"
    if value < 100:
        return "normal"
    if value < 126:
        return "high/prediabetes"
    return "diabetes range"


_BLOOD_SUGAR_SEVERITY_INSTRUCTION = (
    "\n\nIMPORTANT — STATED BLOOD SUGAR READING, CLINIC GUIDANCE TABLE: the patient's "
    "message states a specific blood sugar/glucose reading. This clinic's own guidance "
    "table is: FASTING (8+ hrs) — below 70 = low, 70-99 = normal, 100-125 = high/"
    "prediabetes, 126+ = diabetes range. 2 HOURS AFTER A MEAL — below 70 = low, below "
    "140 = normal, 140-199 = high/prediabetes, 200+ = diabetes range. If the patient "
    "didn't say whether this was fasting or after a meal, use the fasting thresholds as "
    "the default read. The reading itself IS the severity — do NOT ask the patient 'is it "
    "mild, moderate, or severe' for this; work out the category yourself from the number "
    "using this table instead. Only ask about duration if it isn't already stated. "
    "Abnormal results still need confirmation by a healthcare professional — state the "
    "category plainly, don't assert a diagnosis."
)

# Severe hypoglycemia (<54 mg/dL) or a reading high enough to carry a real
# DKA/HHS risk (>400 mg/dL) — standard clinical emergency cutoffs, same "hand
# the model/code the real numeric threshold, never make it reason about one
# itself" principle as every other vital-reading table in this file. Used by
# _vitals_reading_backstop below, the only caller — unlike fever/BP, blood
# sugar had no code-level emergency threshold defined anywhere before this.
_BLOOD_SUGAR_EMERGENCY_LOW = 54
_BLOOD_SUGAR_EMERGENCY_HIGH = 400


def _blood_sugar_is_emergency_range(value: int) -> bool:
    return value < _BLOOD_SUGAR_EMERGENCY_LOW or value > _BLOOD_SUGAR_EMERGENCY_HIGH


# Topic vocabulary for _vitals_reading_backstop below — deliberately just the
# vital's own name/abbreviation, not full symptom vocabulary, since this only
# needs to recognize "the patient is talking about fever/BP/blood sugar AT
# ALL," not classify the message's overall symptom category.
_FEVER_TOPIC_RE = re.compile(r"\b(fever|temp|temperature)\b", re.IGNORECASE)
_BP_TOPIC_RE = re.compile(r"\b(bp|blood\s*pressure|hypertension|hypotension)\b", re.IGNORECASE)
_SUGAR_TOPIC_RE = re.compile(r"\b(sugar|glucose|diabet\w*)\b", re.IGNORECASE)


_VITALS_EMERGENCY_REPLY_TEMPLATE = (
    "This sounds like an emergency — a {reading_description} needs urgent medical attention. "
    "Please call 1122 or go to the nearest emergency department right away.\n\n"
    "1) Stay as calm as possible and avoid unnecessary movement.\n"
    "2) Do not take any medication unless a medical professional tells you to.\n"
    "3) If possible, don't go alone — have someone accompany you or call an ambulance."
)

# Every canned reply _vitals_reading_backstop itself can produce, as prefixes
# — used to recognize "the preceding assistant turn was THIS backstop asking
# for the reading/duration" so answering it is never mistaken for spending an
# unrelated PATH-2-style screening round (see the "PATH 2 is exactly one
# round" guard inside _vitals_reading_backstop for the reported-live bug this
# closes).
_VITALS_BACKSTOP_QUESTION_PREFIXES = (
    "Could you tell me your temperature reading",
    "Could you tell me your blood pressure reading",
    "Could you tell me your blood sugar reading",
    "Got it — and how long have you had it?",
)


# Requested: "for fever/sugar/bp, never ask severity — just ask for the
# reading" as a CODE-LEVEL check, not a prompt instruction the model might or
# might not follow. Reported live, full transcript: "fever of 104" correctly
# skipped straight to asking duration on the FIRST message (the
# gives_severity check in the general PATH-3 backstop above already covered
# turn 1 specifically) — but the very next turn ("since yesterday", no number
# restated) fell through to the LLM's own judgment with zero deterministic
# guidance (the _FEVER_SAFE_RANGE_INSTRUCTION prompt injection near
# _build_system_prompt only ever scanned the CURRENT message for a number,
# never prior turns), and the model re-asked "mild, moderate, or severe" for
# a fever it had already been given a number for. Then, several turns later,
# it wrongly escalated a documented-safe 103°F reading (this clinic's own
# table: 97-103°F inclusive is NOT an emergency) to a false emergency alert —
# the exact failure this whole guidance-table system exists to prevent,
# simply because the guidance wasn't reliably present on that turn either.
#
# Unlike the general PATH-3 backstop above (gated by no_symptom_yet, i.e.
# first message of the session only), this runs on EVERY turn of a fever/BP/
# blood-sugar-topic episode, deterministically, before the LLM is ever
# called — a stated numeric reading for these three vitals never needs the
# mild/moderate/severe question on ANY turn (the reading itself always IS
# the severity, same principle the existing prompt instructions already
# state), and an emergency-range reading is answered immediately, in code,
# rather than relying on the model to notice and act on it correctly.
#
# Scoped to the CURRENT symptom episode only (same "clears on a genuinely new
# symptom label" convention as _session_had_an_earlier_emergency_flag_for_
# the_current_symptom/_session_stated_long_duration_for_the_current_symptom
# above) so switching to an unrelated complaint doesn't keep this backstop
# firing on stale vitals from an earlier, resolved episode. Supports a
# correction ("no no its 103" after "104") the same way every other reading
# in this function already does — later readings simply overwrite earlier
# ones as the scan walks forward, so the LATEST stated number always wins.
#
# Reported live: "no no its 103" (correcting an earlier "fever of 104") still
# read back as 104 — _extract_stated_fever_temperature requires the literal
# word "fever"/"temp"/"temperature" adjacent to the number, which a natural
# follow-up correction usually doesn't repeat at all, relying entirely on the
# already-established topic for context the way a human reader would. The
# bare-number fallback below only fires once EXACTLY ONE of the three vital
# topics is already active from earlier in the episode (never firing at all
# ambiguously across two, e.g. a fever+BP episode) and the message itself is
# short — the same two guardrails _BARE_NUMBER_ONLY_RE already uses for pain-
# scale corrections elsewhere in this file.
_BARE_NUMBER_RE = re.compile(r"\b(\d{1,3}(?:\.\d)?)\b")


def _extract_bare_number(text: str) -> float | None:
    if len(text.split()) > 6:
        return None
    match = _BARE_NUMBER_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


@dataclass
class _VitalsEpisodeState:
    fever_temp: float | None = None
    bp_reading: tuple[int, int] | None = None
    blood_sugar: int | None = None
    saw_fever_topic: bool = False
    saw_bp_topic: bool = False
    saw_sugar_topic: bool = False
    saw_duration: bool = False
    saw_severity_word: bool = False

    def started(self) -> bool:
        return self.fever_temp is not None or self.bp_reading is not None or self.blood_sugar is not None or (
            self.saw_fever_topic or self.saw_bp_topic or self.saw_sugar_topic
        )

    def has_reading(self) -> bool:
        return self.fever_temp is not None or self.bp_reading is not None or self.blood_sugar is not None

    def reset(self) -> None:
        self.fever_temp = self.bp_reading = self.blood_sugar = None
        self.saw_fever_topic = self.saw_bp_topic = self.saw_sugar_topic = False
        self.saw_duration = self.saw_severity_word = False


# Shared by _vitals_reading_backstop below AND the fever/BP/blood-sugar
# guidance-table prompt injections near _build_system_prompt's call site —
# reported live: "its 103 since 6 days" (answering this backstop's OWN "give
# me your reading" question) still got asked "mild, moderate, or severe"
# again, then a bare correction "its 103" wrongly escalated — because once
# _vitals_reading_backstop itself defers to the LLM (see its own "PATH 2 is
# exactly one round" guard below), the OLD prompt-injection mechanism only
# ever scanned the CURRENT message for a number, never the reading given one
# or two turns earlier in the SAME episode, leaving the LLM with zero
# deterministic guidance for a reading it had already been given. Extracting
# the episode scan into its own function lets both call sites see the exact
# same up-to-date picture of "what reading, if any, is this episode's".
#
# A bare topic WORD ("fever"/"bp"/"sugar") with no accompanying number only
# counts toward starting/continuing the episode when the message ISN'T
# purely answering a screening question about a DIFFERENT complaint — same
# differentiator-vs-new-complaint distinction
# _hinted_departments_excluding_a_screening_answer/_session_had_an_earlier_
# emergency_flag_for_the_current_symptom already draw elsewhere in this file
# (reported live: "sore throat" -> "is it accompanied by fever or chills?" ->
# "yes fever and chills" must never start a fever-topic episode of its own).
# An actual stated NUMBER always counts regardless of that distinction — a
# real reading's clinical significance can't be dismissed just because it
# happened to come up while answering a differentiator question.
def _scan_vitals_episode(
    message: str, history: list[ConversationMemory], department_names: list[str]
) -> _VitalsEpisodeState:
    state = _VitalsEpisodeState()

    def _scan(text: str, answering_screening_question: bool) -> None:
        topic_before = (state.saw_fever_topic, state.saw_bp_topic, state.saw_sugar_topic)
        if not answering_screening_question:
            if _FEVER_TOPIC_RE.search(text):
                state.saw_fever_topic = True
            if _BP_TOPIC_RE.search(text):
                state.saw_bp_topic = True
            if _SUGAR_TOPIC_RE.search(text):
                state.saw_sugar_topic = True
        if _DURATION_HINT_RE.search(text):
            state.saw_duration = True
        if _SEVERITY_HINT_RE.search(text):
            state.saw_severity_word = True
        temp = _extract_stated_fever_temperature(text)
        if temp is not None:
            state.fever_temp = temp
            state.saw_fever_topic = True
        bp = _extract_stated_blood_pressure(text)
        if bp is not None:
            state.bp_reading = bp
            state.saw_bp_topic = True
        sugar = _extract_stated_blood_sugar(text)
        if sugar is not None:
            state.blood_sugar = sugar
            state.saw_sugar_topic = True
        if temp is None and bp is None and sugar is None and sum(topic_before) == 1:
            bare = _extract_bare_number(text)
            if bare is not None:
                if topic_before[0] and 90.0 <= bare <= 112.0:
                    state.fever_temp = bare
                elif topic_before[2] and 30 <= bare <= 600:
                    state.blood_sugar = int(bare)

    running_history: list[ConversationMemory] = []
    for row in history:
        if getattr(row, "role", None) == "user":
            content = getattr(row, "content", "") or ""
            if state.started() and _message_introduces_a_new_symptom_label(content, running_history, department_names):
                state.reset()
            _scan(content, _preceding_assistant_turn_asked_a_screening_question(running_history))
        running_history.append(row)
    if state.started() and _message_introduces_a_new_symptom_label(message, history, department_names):
        state.reset()
    _scan(message, _preceding_assistant_turn_asked_a_screening_question(history))
    return state


def _vitals_reading_backstop(
    message: str, history: list[ConversationMemory], department_names: list[str]
) -> str | None:
    state = _scan_vitals_episode(message, history, department_names)
    fever_temp, bp_reading, blood_sugar = state.fever_temp, state.bp_reading, state.blood_sugar
    saw_duration, saw_severity_word = state.saw_duration, state.saw_severity_word

    if not (state.saw_fever_topic or state.saw_bp_topic or state.saw_sugar_topic):
        return None

    # Self-diagnosis claims ("i have diabetes") get the SAME fast, concise
    # LLM-composed redirect as the general backstop above (see
    # _is_self_diagnosis_claim's own comment) rather than this backstop's own
    # "please give me your reading" question, which would read as dismissive
    # for something the patient is already alarmed about — but only when no
    # actual reading was ever stated; a self-diagnosis claim genuinely
    # accompanied by a real emergency-range number must still escalate.
    if fever_temp is None and bp_reading is None and blood_sugar is None and _is_self_diagnosis_claim(message):
        return None

    if fever_temp is not None and not _fever_temperature_is_within_the_clinic_safe_range(fever_temp):
        return _VITALS_EMERGENCY_REPLY_TEMPLATE.format(reading_description=f"fever of {fever_temp:g}°F")
    if bp_reading is not None and _blood_pressure_category(*bp_reading) == "hypertensive crisis":
        return _VITALS_EMERGENCY_REPLY_TEMPLATE.format(
            reading_description=f"blood pressure reading of {bp_reading[0]}/{bp_reading[1]}"
        )
    if blood_sugar is not None and _blood_sugar_is_emergency_range(blood_sugar):
        return _VITALS_EMERGENCY_REPLY_TEMPLATE.format(reading_description=f"blood sugar reading of {blood_sugar}")

    # Everything below only decides whether THIS backstop should ask its own
    # canned follow-up (reading, or duration) — never the emergency check
    # above, which always applies regardless. Two independent reasons to
    # instead defer entirely to the LLM (return None, same as "this doesn't
    # apply at all"):
    #
    # 1. `_preceding_assistant_turn_asked_a_screening_question` — mirrors
    #    PATH 2's own "exactly one round" rule elsewhere in this file: once
    #    the immediately preceding assistant turn already asked SOMETHING
    #    (duration, a differentiator, anything) and the patient answered it,
    #    that round is spent — reported live: "my blood sugar level is
    #    200+" -> LLM asked about accompanying symptoms -> "yes excessive
    #    thirst and frequent urination" (the diabetes triad, answering that
    #    question) still got asked "how long have you had it?" by this
    #    backstop, when the patient had already answered a real screening
    #    question and the diabetes-triad answer should route immediately.
    #    Scoped to the reading/duration CANNED QUESTIONS only — the
    #    emergency check two paragraphs up is never subject to this, an
    #    emergency-range reading always escalates regardless of how many
    #    rounds have already happened.
    #    Reported live (2nd instance): guard 1 was too broad in the OTHER
    #    direction too — "i am having fever" -> this backstop asked "Could
    #    you tell me your temperature reading (in °F), and how long you've
    #    had it?" -> "its 103 since 6 days" (directly answering THIS
    #    backstop's OWN question, reading AND duration both given in one
    #    reply) still got deferred to the LLM by guard 1, since the
    #    preceding turn technically did ask a question — the LLM then
    #    re-asked "mild, moderate, or severe" anyway (no deterministic
    #    guidance for a reading two turns back — see _scan_vitals_episode's
    #    own docstring on why the prompt-injection call site needed the
    #    exact same episode-wide scan), and a later bare correction wrongly
    #    escalated. Guard 1 must only defer when the round that was just
    #    spent belongs to something OTHER than this backstop's own question
    #    — answering this backstop's own question is never a "round" in the
    #    PATH-2 sense, it's this backstop completing its own exchange, and
    #    must always be processed by the checks above/below.
    # 2. Ordinary severity+duration language already given across the
    #    episode (`saw_severity_word and saw_duration`) — reported live: "i
    #    have a mild sore throat since 2 days and i am also having fever and
    #    chills" (fever mentioned as a second, genuinely new complaint in the
    #    very FIRST message, so guard 1 above doesn't apply) got asked for a
    #    temperature reading instead of proceeding straight to the normal
    #    multi-department card — but "mild"/"since 2 days" already convey
    #    exactly the same severity+duration information a reading would,
    #    same "when in doubt, the patient's own ordinary words already
    #    answered it" principle as _message_already_gives_duration_and_
    #    severity elsewhere in this file (that helper only checks the
    #    CURRENT message; this checks the whole episode the same way this
    #    backstop's other state already does).
    preceding_was_this_backstops_own_question = (
        _preceding_assistant_turn_asked_a_screening_question(history)
        and bool(history)
        and (getattr(history[-1], "content", "") or "").startswith(_VITALS_BACKSTOP_QUESTION_PREFIXES)
    )
    if (
        _preceding_assistant_turn_asked_a_screening_question(history)
        and not preceding_was_this_backstops_own_question
    ) or (saw_duration and saw_severity_word):
        return None

    have_reading = fever_temp is not None or bp_reading is not None or blood_sugar is not None
    if not have_reading:
        if state.saw_fever_topic:
            reading_prompt = "your temperature reading (in °F)"
        elif state.saw_bp_topic:
            reading_prompt = "your blood pressure reading (e.g. 130/85)"
        else:
            reading_prompt = "your blood sugar reading (mg/dL), and whether it was fasting or after a meal"
        if saw_duration:
            return f"Could you tell me {reading_prompt}?"
        return f"Could you tell me {reading_prompt}, and how long you've had it?"

    if not saw_duration:
        return "Got it — and how long have you had it?"

    # Reading + duration both known and the reading is in the safe range —
    # nothing further for this backstop to do; fall through to normal PATH 3
    # routing exactly as it would for any other routine symptom.
    return None


# Loosely matches a reply that asks the literal "mild, moderate, or severe"
# question in either word order — used by the recovery net in
# _run_symptom_agent_body below to catch the model asking this AGAIN despite
# a reading that already answers it (see that net's own comment for the
# reported live transcript). Deliberately loose (all three words appearing
# near each other, not a single fixed phrase) since the model's own wording
# varies turn to turn ("Is the fever severe, moderate, or mild?", "mild,
# moderate, or severe?", etc.) — a false positive here only means an
# already-known-safe reading's episode routes a turn early, never mid-
# screening for a symptom with no reading involved at all (this check is
# only ever consulted alongside _episode_has_a_known_safe_vital_reading).
_ASKS_SEVERITY_QUESTION_RE = re.compile(
    r"\bmild\b.{0,30}\bmoderate\b.{0,30}\bsevere\b|\bsevere\b.{0,30}\bmoderate\b.{0,30}\bmild\b",
    re.IGNORECASE,
)


def _reply_asks_the_severity_question_again(reply: str) -> bool:
    return bool(_ASKS_SEVERITY_QUESTION_RE.search(reply))


def _episode_has_a_known_safe_vital_reading(
    message: str, history: list[ConversationMemory], department_names: list[str]
) -> bool:
    state = _scan_vitals_episode(message, history, department_names)
    if state.fever_temp is not None and _fever_temperature_is_within_the_clinic_safe_range(state.fever_temp):
        return True
    if state.bp_reading is not None and _blood_pressure_category(*state.bp_reading) != "hypertensive crisis":
        return True
    if state.blood_sugar is not None and not _blood_sugar_is_emergency_range(state.blood_sugar):
        return True
    return False


def _message_states_a_numeric_vital_reading(message: str) -> bool:
    # Reported live: "i am having fever and its 103" still got asked "how severe
    # is this... and how long" — this function is what the PATH-3 "never zero
    # questions" backstop's gives_severity check relies on to recognize a numeric
    # reading as severity-already-given (same as it already does for BP/blood
    # sugar), but it never called _extract_stated_fever_temperature at all, even
    # though that function already existed in this same file for the clinic's own
    # fever guidance table. A stated fever number is exactly as much "the reading
    # itself IS the severity" as a stated BP/sugar reading is (see this function's
    # existing callers' own comments) — there was never a reason fever should have
    # been left out.
    return (
        _extract_stated_blood_pressure(message) is not None
        or _extract_stated_blood_sugar(message) is not None
        or _extract_stated_fever_temperature(message) is not None
    )


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
        f"{llm._OFF_TOPIC_AND_INTEGRITY_RULES}"
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


# Reported live: a patient described a headache as "very severe" (a real PATH 1
# emergency reply, twice — once initially, once again after they said "but i
# dont wana goto er"), then said "no no its not severe its mild" — the very
# next reply accepted this with ZERO acknowledgment of the earlier severe
# claim, screened normally, and ended with a plain "General Medicine would be
# appropriate" card as if the severe episode had never happened. Nothing
# challenges or double-checks a downgrade like this today — it's entirely up
# to the model's own in-context judgment, unlike every other safety-critical
# decision in this codebase, which has a real code backstop. This still
# RESPECTS the patient's own self-correction (never repeats the scary message
# again, never blocks the routine booking) — it only makes sure the earlier
# claim isn't silently erased, by attaching a one-time reminder to whatever
# routine reply eventually follows.
_EMERGENCY_DOWNGRADE_SAFETY_NOTE = (
    "Earlier in this conversation this was described as very severe. If it becomes that severe "
    "again, or you notice any concerning new symptoms, please seek emergency care immediately."
)


# Reported live: a genuine emergency flag for SEVERE STOMACH pain correctly attached
# the downgrade note to the immediate stomach-pain follow-ups — but then kept
# attaching to completely unrelated LATER symptoms in the same session (a fresh mild
# headache, then a fresh chest-pain report) that were never themselves described as
# severe at all. _session_had_an_earlier_emergency_flag only checks "did ANY
# emergency reply happen anywhere in this session," with no concept of the patient
# having moved on to a different complaint entirely.
#
# _introduces_a_new_symptom_category (used elsewhere in this file) can't be reused
# as-is here: it diffs DEPARTMENT names, and headache/stomach both hint "General
# Medicine" at a clinic with no separate Neurology/Gastro department, so a
# department-name diff sees no change at all between them. Diffing the symptom
# LABEL instead ("head pain" vs "stomach symptoms" vs "chest pain") correctly tells
# these apart even when they'd route to the very same real department.
def _message_introduces_a_new_symptom_label(
    message: str, history: list[ConversationMemory], department_names: list[str]
) -> bool:
    if _preceding_assistant_turn_asked_a_screening_question(history):
        return False
    prior_labels = set(departments_hinted_by_patient_symptom_words("", history, department_names, set()).values())
    message_labels = set(departments_hinted_by_patient_symptom_words(message, [], department_names, set()).values())
    return bool(message_labels - prior_labels)


def _session_had_an_earlier_emergency_flag_for_the_current_symptom(
    message: str, history: list[ConversationMemory], department_names: list[str]
) -> bool:
    """Same underlying signal as _session_had_an_earlier_emergency_flag, but scoped
    to the CURRENT symptom episode only — the flag is cleared the moment a genuinely
    new symptom label is introduced (by the patient, unprompted), same as how
    _introduces_a_new_symptom_category tracks a fresh complaint elsewhere in this
    file, just comparing symptom labels rather than department names (see comment
    above)."""
    running_history: list[ConversationMemory] = []
    saw_emergency = False
    for row in history:
        role = getattr(row, "role", None)
        content = getattr(row, "content", "") or ""
        if role == "user":
            if saw_emergency and _message_introduces_a_new_symptom_label(content, running_history, department_names):
                saw_emergency = False
        elif role == "assistant":
            if getattr(row, "red_flag", False) or _looks_like_a_valid_emergency_reply(content):
                saw_emergency = True
        running_history.append(row)
    if saw_emergency and _message_introduces_a_new_symptom_label(message, history, department_names):
        saw_emergency = False
    return saw_emergency


def _append_emergency_downgrade_safety_note(
    reply: str, message: str, history: list[ConversationMemory], department_names: list[str]
) -> str:
    if _looks_like_a_valid_emergency_reply(reply):
        # This reply IS itself a fresh PATH 1 emergency reply — nothing to
        # append, the emergency is already being addressed head-on.
        return reply
    if not _session_had_an_earlier_emergency_flag_for_the_current_symptom(message, history, department_names):
        return reply
    if reply.startswith(DOCTOR_OPTIONS_MARKER):
        payload = json.loads(reply[len(DOCTOR_OPTIONS_MARKER):])
        existing_note = payload.get("note")
        payload["note"] = f"{existing_note} {_EMERGENCY_DOWNGRADE_SAFETY_NOTE}" if existing_note else _EMERGENCY_DOWNGRADE_SAFETY_NOTE
        return DOCTOR_OPTIONS_MARKER + json.dumps(payload, default=str)
    if reply.startswith((DEPARTMENT_LIST_MARKER, NO_SLOTS_MARKER)):
        # Multi-department/no-slots shapes are rarer right after a downgrade
        # and more involved to rewrite structurally — left untouched rather
        # than risk corrupting a real card; the common single-department case
        # above is what the reported transcript actually showed.
        return reply
    return f"{reply}\n\n{_EMERGENCY_DOWNGRADE_SAFETY_NOTE}"


_LONG_DURATION_NOTE = (
    "This has been going on for a while — even if it isn't severe, it's best to get it "
    "checked soon so treatment isn't delayed."
)


def _session_stated_long_duration_for_the_current_symptom(
    message: str, history: list[ConversationMemory], department_names: list[str]
) -> bool:
    """Same same-symptom-episode tracking as
    _session_had_an_earlier_emergency_flag_for_the_current_symptom above,
    just for a long/chronic duration instead of an emergency flag — cleared
    the moment either a genuinely new symptom is introduced (same signal as
    that function) or the patient explicitly walks the duration back (see
    _message_corrects_long_duration)."""
    running_history: list[ConversationMemory] = []
    long_duration = False
    for row in history:
        role = getattr(row, "role", None)
        content = getattr(row, "content", "") or ""
        if role == "user":
            if long_duration and (
                _message_corrects_long_duration(content)
                or _message_introduces_a_new_symptom_label(content, running_history, department_names)
            ):
                long_duration = False
            elif _message_states_long_duration(content):
                long_duration = True
        running_history.append(row)
    if long_duration and (
        _message_corrects_long_duration(message)
        or _message_introduces_a_new_symptom_label(message, history, department_names)
    ):
        long_duration = False
    elif _message_states_long_duration(message):
        long_duration = True
    return long_duration


def _append_long_duration_note(
    reply: str, message: str, history: list[ConversationMemory], department_names: list[str]
) -> str:
    if _looks_like_a_valid_emergency_reply(reply):
        # Already being told to seek emergency care right now — a "get it
        # checked soon" note would read as a confusing downgrade.
        return reply
    if not _session_stated_long_duration_for_the_current_symptom(message, history, department_names):
        return reply
    if reply.startswith(DOCTOR_OPTIONS_MARKER):
        payload = json.loads(reply[len(DOCTOR_OPTIONS_MARKER):])
        existing_note = payload.get("note")
        payload["note"] = f"{existing_note} {_LONG_DURATION_NOTE}" if existing_note else _LONG_DURATION_NOTE
        return DOCTOR_OPTIONS_MARKER + json.dumps(payload, default=str)
    # Multi-department/no-slots shapes left untouched — same conservative
    # scoping as _append_emergency_downgrade_safety_note above, for the same
    # reason (rarer shape, more involved to rewrite structurally without risk).
    return reply


def run_symptom_agent(
    db: Session,
    ctx: ClinicContext,
    message: str,
    language: str,
    history: list[ConversationMemory],
) -> str:
    department_names = list_active_department_names(db, ctx.clinic_id)
    reply = _append_emergency_downgrade_safety_note(
        _run_symptom_agent_body(db, ctx, message, language, history), message, history, department_names
    )
    reply = _append_long_duration_note(reply, message, history, department_names)
    return ensure_slot_pick_example_has_a_date(reply)


def _run_symptom_agent_body(
    db: Session,
    ctx: ClinicContext,
    message: str,
    language: str,
    history: list[ConversationMemory],
) -> str:
    department_names = list_active_department_names(db, ctx.clinic_id)

    # Checked before every other deterministic check below, including the
    # unsupported-department apology and recommendation-request handling —
    # see _vitals_reading_backstop's own docstring for why a fever/BP/blood-
    # sugar reading's severity/emergency status must never be left to the LLM
    # on ANY turn, not just the first message.
    vitals_reply = _vitals_reading_backstop(message, history, department_names)
    if vitals_reply is not None:
        return vitals_reply

    # Instructed live: a symptom this clinic genuinely has no matching specialist
    # for (e.g. a growth/height concern with no Endocrinology department, or a
    # urinary/testicular complaint with no Urology department) used to silently
    # fall back to General Medicine just because SOME department existed — that's
    # misleading, since General Medicine isn't actually equipped for it. Checked
    # before anything else, so it applies whether the patient described the
    # symptom plainly or asked for a recommendation, and answered with an honest,
    # gentle apology instead — never a wrong-department card. Only fires when
    # NOTHING real is hinted at all for the whole conversation so far: a compound
    # complaint that also names something this clinic DOES treat (e.g. "urinary
    # pain and chest pain") still gets that real department normally; the
    # unsupported part simply isn't separately called out in that case.
    #
    # Instructed live (general case): the hardcoded _ORPHAN_SYMPTOM_CATEGORIES
    # list above only ever covers categories someone happened to add — any OTHER
    # symptom this clinic can't treat used to be left to the LLM to freehand a
    # guess (the same class of bug as the old brain-tumor->Neurology mistake).
    # symptom_words_with_no_matching_department checks the WHOLE symptom-hint
    # table generically, so any symptom category it already knows about gets
    # this same honest treatment — and, unlike the hardcoded list, it also names
    # the specialty the patient should look for elsewhere (e.g. "Oncology"),
    # since knowing that only matters once you already know it isn't here.
    #
    # Reported live: after this apology fired for "I think I have cancer," a
    # completely unrelated follow-up ("what's 2+2") got the EXACT SAME apology
    # again instead of the normal off-topic redirect. Root cause: both
    # unsupported_symptom_labels and symptom_words_with_no_matching_department
    # scan message+history TOGETHER — a symptom word from three turns ago never
    # leaves that combined word-set, so the check kept firing on every later
    # turn regardless of what the CURRENT message actually said. This gate
    # requires the CURRENT message itself to still be symptom-shaped or an
    # explicit "based on my symptoms" recommendation request — a message with
    # neither (small talk, math, an unrelated factual question) skips this
    # apology entirely and falls through to the normal LLM flow below, which
    # has its own off-topic guardrail (_OFF_TOPIC_AND_INTEGRITY_RULES) to
    # handle it instead of repeating stale department-availability context.
    current_message_still_about_a_symptom = (
        is_symptom_message(message) or is_department_recommendation_request(message)
    )
    hinted = departments_hinted_by_patient_symptom_words(message, history, department_names, set())
    unsupported = unsupported_symptom_labels(message, history, department_names)
    unmatched = symptom_words_with_no_matching_department(message, history, department_names)
    if current_message_still_about_a_symptom and (unsupported or unmatched) and not hinted:
        seen_labels: set[str] = set()
        clauses_en: list[str] = []
        clauses_ur: list[str] = []
        for label in unsupported:
            if label in seen_labels:
                continue
            seen_labels.add(label)
            clauses_en.append(f"a specialist department for {label}")
            clauses_ur.append(f"{label} کے لیے کوئی مخصوص شعبہ")
        for label, canonical_name in unmatched:
            if label in seen_labels:
                continue
            seen_labels.add(label)
            clauses_en.append(f"a {canonical_name} department for {label}")
            clauses_ur.append(f"{label} کے لیے {canonical_name} کا شعبہ")
        clauses_text_en = " or ".join(clauses_en)
        clauses_text_ur = " یا ".join(clauses_ur)
        return (
            f"I'm sorry, this clinic doesn't have {clauses_text_en}. I'd recommend visiting another "
            "hospital or clinic for this. Is there anything else I can help you with?"
            if language != "ur"
            else (
                f"معذرت، اس کلینک میں {clauses_text_ur} دستیاب نہیں ہے۔ "
                "براہ کرم اس کے لیے کسی دوسرے ہسپتال یا کلینک سے رجوع کریں۔ کیا میں کسی اور معاملے میں "
                "آپ کی مدد کر سکتا ہوں؟"
            )
        )

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
    # the same four universal exceptions (a recommendation request, duration+
    # severity already given, a self-diagnosis claim, high-severity wording):
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
        and not _message_already_gives_duration_and_severity(message, history)
        and not _is_self_diagnosis_claim(message)
        # Reported live: "i am having severe sore throat" (sore throat isn't in
        # any PATH 2 keyword list, so include_path2 was False) got "Got it —
        # and how long have you had it?" from this exact block below instead
        # of going straight to PATH 1 with zero further questions — the
        # gives_severity check a few lines down only asks WHETHER a severity
        # word matched, never WHICH one, so "severe" got the same
        # ask-for-duration treatment as "mild"/"moderate". This backstop
        # exists to guarantee AT LEAST ONE clarifying question is asked; it
        # must never force one in on top of wording that means the correct
        # number of further questions is actually zero.
        and not _HIGH_SEVERITY_HINT_RE.search(message)
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
        # Reported live: this backstop always asked BOTH severity and duration, even
        # when the patient's own message already volunteered one of the two — e.g.
        # "it's been going on for 3 days" got asked "how severe...and how long" again,
        # re-asking something already answered. `_message_already_gives_duration_and_severity`
        # above only gates the case where BOTH are present (and skips this whole block
        # then); this narrows the question itself to whichever piece is still missing.
        # A stated blood pressure/blood sugar reading counts as severity already
        # given the same way "mild"/"frequent"/etc. do — the reading itself IS the
        # severity per the clinic's own guidance tables (see
        # _message_states_a_numeric_vital_reading's own comment), so this must
        # never re-ask "how severe" for it either.
        gives_severity = (
            bool(_SEVERITY_HINT_RE.search(message))
            or _message_states_a_numeric_vital_reading(message)
            or _extract_stated_pain_scale_number(message, history) is not None
            or _message_states_persistent_thirst(message)
        )
        gives_duration = bool(_DURATION_HINT_RE.search(message))
        if gives_severity and not gives_duration:
            return "Got it — and how long have you had it?"
        if gives_duration and not gives_severity:
            return "Got it — and how severe is it (mild, moderate, or severe)?"
        return (
            "Could you tell me how severe this is (mild, moderate, or severe) and how "
            "long you've had it?"
        )

    language_name = llm._LANGUAGE_NAMES.get(language, "English")
    system_prompt = _build_system_prompt(language_name, department_names, include_path2)
    if _last_reply_was_an_emergency_reply(history) and _message_downgrades_severity(message):
        system_prompt += _SEVERITY_DOWNGRADE_INSTRUCTION
    # Episode-wide (not just the CURRENT message) — see _scan_vitals_episode's
    # own docstring: a reading given one or two turns back (e.g. answering
    # _vitals_reading_backstop's own "what's your reading?" question, which
    # this call site never sees since that backstop already returned before
    # the LLM was ever reached) must still get this deterministic guidance on
    # every later turn of the same episode, not just the turn the number was
    # first stated in.
    vitals_episode = _scan_vitals_episode(message, history, department_names)
    stated_fever_temp = vitals_episode.fever_temp
    if stated_fever_temp is not None and _fever_temperature_is_within_the_clinic_safe_range(stated_fever_temp):
        system_prompt += _FEVER_SAFE_RANGE_INSTRUCTION
    stated_bp = vitals_episode.bp_reading
    if stated_bp is not None:
        system_prompt += _BLOOD_PRESSURE_SEVERITY_INSTRUCTION
    stated_blood_sugar = vitals_episode.blood_sugar
    if stated_blood_sugar is not None:
        system_prompt += _BLOOD_SUGAR_SEVERITY_INSTRUCTION
    stated_pain_scale = _extract_stated_pain_scale_number(message, history)
    if stated_pain_scale is not None:
        system_prompt += _PAIN_SCALE_SEVERITY_INSTRUCTION_TEMPLATE.format(
            value=stated_pain_scale, word=_pain_scale_number_to_severity_word(stated_pain_scale)
        )

    forced_date_window = resolve_date_window(message)
    forced_time_window = resolve_time_of_day_window(message)
    tools = [
        t
        for t in build_tools(db, ctx, forced_date_window=forced_date_window, forced_time_window=forced_time_window)
        if t.name in _SYMPTOM_AGENT_TOOL_NAMES
    ]

    reply = llm.run_tool_calling_agent(system_prompt, message, history, tools)

    # Reported live: "i have a brain tumour" got a full PATH 1 emergency reply
    # while "i think i have a brain tumour" (the identical claim, just hedged)
    # got the routine department redirect this module's design calls for — see
    # _bare_self_diagnosis_claim's own docstring. Applied even to an
    # otherwise-valid-looking PATH 1 reply (the one exception the recovery net
    # below deliberately never touches), so both phrasings land on the same
    # deterministic outcome.
    if _looks_like_a_valid_emergency_reply(reply) and _bare_self_diagnosis_claim(message):
        hinted = departments_hinted_by_patient_symptom_words(message, history, department_names, set())
        if hinted:
            results = [
                _get_department_availability_impl(
                    db, ctx, name, note=f"Based on the {label} described, this should be evaluated by a doctor."
                )
                for name, label in hinted.items()
            ]
            reply = results[0] if len(results) == 1 else combine_department_availability_results(results)

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
    # cases above — same recovery, wider trigger. Excludes a valid PATH 1 emergency
    # reply (see _looks_like_a_valid_emergency_reply's own docstring) — that reply
    # is meant to end the turn exactly as written, never routed to a department.
    if (
        not reply.startswith((DOCTOR_OPTIONS_MARKER, DEPARTMENT_LIST_MARKER, NO_SLOTS_MARKER))
        and not _looks_like_a_valid_emergency_reply(reply)
        and (
            violates_no_diagnosis_rule(reply)
            or _looks_like_a_faked_tool_payload(reply)
            or _looks_like_an_advice_dump_instead_of_routing(reply)
            or _recommends_a_specialist_in_free_text(reply)
            or _names_a_real_department_in_free_text(reply, department_names)
            or _asks_the_patient_to_pick_a_department(reply)
            # Reported live: "its 103 since 10 days" (safe range, duration
            # given) -> "Is the fever severe, moderate, or mild? Also..." ->
            # "no such symptoms" -> "Is the fever mild, moderate, or
            # severe?" — TWICE the model re-asked the exact question this
            # clinic's own guidance table says a stated reading already
            # answers, despite _run_symptom_agent_body injecting the safe-
            # range instruction on every one of these turns (see
            # _scan_vitals_episode's own docstring — that part already works
            # correctly). _vitals_reading_backstop above only takes full,
            # guaranteed control for the FIRST round of a vitals episode;
            # once one real screening round has happened it deliberately
            # hands off to the model for the rest (see that backstop's own
            # "PATH 2 is exactly one round" comment, and the diabetes-triad
            # bug that guard was built to fix) — but handing off doesn't mean
            # the model can be trusted to get it right every time, same
            # "prompt instruction alone isn't a guarantee" lesson as every
            # other recovery net in this if-block. Same recovery as the rest
            # of this block: discard the redundant question and route from
            # the real symptom-hint table instead.
            or (
                _reply_asks_the_severity_question_again(reply)
                and _episode_has_a_known_safe_vital_reading(message, history, department_names)
            )
        )
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

    # Requested: "sore throat" -> "is it accompanied by fever or chills?" -> "yes
    # fever" (or "fever and chills") got BOTH ENT (for the sore throat) AND
    # General Medicine (for "fever"/"chills") shown as separate departments — but
    # "fever"/"chills" here is a DIFFERENTIATOR ANSWER about the sore throat
    # complaint, not an independent new complaint of its own. The same word,
    # raised as its OWN standalone complaint ("i have fever and chills"), still
    # correctly hints General Medicine as a real, separate thing. The
    # distinguishing signal is exactly the one _introduces_a_new_symptom_category
    # already uses for the identical "answering vs volunteering" question:
    # whether the immediately preceding assistant turn was a plain screening
    # question.
    #
    # Reported live (2nd instance): the first fix here only excluded the CURRENT
    # message's own words when IT was answering a screening question — but
    # "sore throat" -> "any fever/swallowing trouble/swollen glands?" -> "yes a
    # bit of fever" -> "any cough or runny nose?" -> "yes a runny nose" still
    # pulled General Medicine in, because "fever" came from an EARLIER
    # differentiator answer still sitting in `history`, which the hint scanner
    # (departments_hinted_by_patient_symptom_words) always scans regardless of
    # which turn it came from. This walks the WHOLE history turn by turn and
    # excludes EVERY user turn that was itself answering a screening question at
    # the time it was said, not just the current one — a differentiator answered
    # three turns ago is excluded exactly the same way as one answered just now.
    def _hinted_departments_excluding_a_screening_answer(already_covered: set[str]) -> dict[str, str]:
        filtered_history: list[ConversationMemory] = []
        for index, row in enumerate(history):
            if getattr(row, "role", None) == "user" and _preceding_assistant_turn_asked_a_screening_question(
                history[:index]
            ):
                filtered_history.append(SimpleNamespace(role="user", content=""))
            else:
                filtered_history.append(row)
        effective_message = "" if _preceding_assistant_turn_asked_a_screening_question(history) else message
        # corroboration_message/corroboration_history deliberately pass the REAL,
        # unfiltered message/history — see departments_hinted_by_patient_symptom_
        # words' and _matched_hint_groups' own docstrings for why a screening
        # answer's corroboration-only words (e.g. "swelling") must still count even
        # though its GATING words are excluded here.
        return departments_hinted_by_patient_symptom_words(
            effective_message, filtered_history, department_names, already_covered,
            corroboration_message=message, corroboration_history=history,
        )

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
        for name, label in _hinted_departments_excluding_a_screening_answer(already_covered).items():
            if name not in missing:
                missing.append(name)
                extra_notes[name] = f"Based on the {label} described, this could also be evaluated by {name}."
        if missing:
            extra_results = [
                _get_department_availability_impl(db, ctx, name, note=extra_notes[name]) for name in missing
            ]
            reply = combine_department_availability_results([reply, *extra_results])

    # Requested: a code-level check (not a prompt instruction, to avoid the extra
    # prompt tokens) for when the model's OWN tool-calling loop already called
    # get_department_availability more than once THIS turn — that's combined into a
    # DEPARTMENT_LIST_MARKER reply before this function ever sees it (see
    # llm._finalize_reply), so the single-department correction block above never
    # runs at all for this shape. Reported live: a patient describing ONLY chest
    # pain and sweating got a SECOND department (General Medicine) with a note
    # reading "Based on the fever/body aches described" — neither was ever
    # mentioned; the second department and its whole justification were
    # fabricated. The first/primary department is always kept as-is (same
    # "primary is whatever the model concluded first" convention as the
    # single-department correction above); every ADDITIONAL department must be
    # grounded in a real symptom hint from the patient's own words (or directly
    # named by the patient) or it's dropped — never shown on the strength of the
    # model's own freehand reasoning alone.
    elif reply.startswith(DEPARTMENT_LIST_MARKER):
        payload = json.loads(reply[len(DEPARTMENT_LIST_MARKER):])
        departments = payload.get("departments") or []
        if len(departments) > 1:
            hinted = _hinted_departments_excluding_a_screening_answer(set())
            kept = [departments[0]]
            for entry in departments[1:]:
                name = entry.get("department_name")
                if name in hinted or _patient_named_this_department(name, message, history):
                    kept.append(entry)
            if len(kept) < len(departments):
                if len(kept) == 1:
                    sole = kept[0]
                    reply = DOCTOR_OPTIONS_MARKER + json.dumps(
                        {
                            "department_name": sole["department_name"],
                            "note": sole.get("note"),
                            "doctors": sole["doctors"],
                        },
                        default=str,
                    )
                else:
                    reply = DEPARTMENT_LIST_MARKER + json.dumps(
                        {"departments": kept, "unavailable": payload.get("unavailable", [])}, default=str
                    )

    return reply
