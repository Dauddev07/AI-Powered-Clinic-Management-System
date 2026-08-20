"""Intent layer for the orchestrator architecture — classifies each patient message
into exactly one specialist agent: SYMPTOM_GENERAL, APPOINTMENT, or GENERAL_INFO.

Mirrors app.services.message_classifier.classify_message_intent's shape exactly: a
free, deterministic heuristic decides first, and only genuinely ambiguous messages
fall through to one GROQ_HELPER_MODEL call — most messages never pay for an LLM call
here, same as today's classifier. Built entirely from existing signals already in
app.services.message_classifier (is_symptom_message, needs_booking_action_tools,
_preceding_assistant_turn_looks_like_a_question) — none of that logic is
reimplemented here.

RECENCY FIX: a bare reply to a clarifying question ("very severe", "the Monday one")
is ambiguous on wording alone — it could be continuing symptom screening (route back
to SYMPTOM_GENERAL) or a booking/disambiguation reply (route to APPOINTMENT). Rather
than defaulting one way always, this compares WHICH happened more recently in the
conversation — a symptom mentioned once several turns ago must not outrank a
conversation that has since clearly moved into booking territory. See
_symptom_context_more_recent_than_booking_context.
"""
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.core.api_keys import api_key_manager
from app.core.config import settings
from app.services.chat_markers import (
    BOOKING_MARKER,
    DEPARTMENT_LIST_MARKER,
    DOCTOR_DISAMBIGUATION_MARKER,
    DOCTOR_OPTIONS_MARKER,
)
from app.services.message_classifier import (
    _heuristic_classify as _classic_heuristic_classify,
)
from app.services.message_classifier import (
    _CANCEL_ACTION_WORDS,
    _RESCHEDULE_ACTION_WORDS,
    _preceding_assistant_turn_looks_like_a_question,
    is_department_list_request,
    is_department_recommendation_request,
    is_department_scope_question,
    is_doctor_count_or_listing_request,
    is_symptom_message,
    needs_booking_action_tools,
)
from app.services.orchestrator.agents.appointment_agent import (
    _message_asks_for_most_recent_appointment_by_status,
)

logger = logging.getLogger(__name__)

SYMPTOM_GENERAL = "symptom_general"
APPOINTMENT = "appointment"
GENERAL_INFO = "general_info"


def _most_recent_matching_index(history, predicate) -> int:
    """Index (0-based, into `history`, most-recent-last) of the LAST row for which
    predicate(row) is True, or -1 if none match."""
    if not history:
        return -1
    for i in range(len(history) - 1, -1, -1):
        if predicate(history[i]):
            return i
    return -1


def _most_recent_symptom_turn_index(history) -> int:
    return _most_recent_matching_index(
        history, lambda row: getattr(row, "role", None) == "user" and is_symptom_message(row.content or "")
    )


def _most_recent_booking_context_index(history) -> int:
    def _is_booking_signal(row) -> bool:
        role = getattr(row, "role", None)
        content = getattr(row, "content", "") or ""
        if role == "assistant":
            return content.startswith(
                (DOCTOR_OPTIONS_MARKER, DEPARTMENT_LIST_MARKER, BOOKING_MARKER, DOCTOR_DISAMBIGUATION_MARKER)
            )
        if role == "user":
            # history=None here is deliberate: we only want THIS row's own
            # keyword/phrase signal in isolation while scanning, not its
            # "preceding turn" check re-evaluated per historical row (that would
            # be a different question — asked and answered by rule 2 already).
            return needs_booking_action_tools(content, None)
        return False

    return _most_recent_matching_index(history, _is_booking_signal)


def _symptom_context_more_recent_than_booking_context(history) -> bool:
    return _most_recent_symptom_turn_index(history) > _most_recent_booking_context_index(history)


_MAX_NO_SIGNAL_WORDS = 12


def _looks_like_signal_free_short_statement(message: str) -> bool:
    """True for a short, plain-Latin-script statement with no recognizable
    symptom/booking/knowledge signal at all — used only as rule 6's last-resort
    default to GENERAL_INFO instead of paying for _llm_classify below, whose own
    prompt is deliberately biased toward SYMPTOM_GENERAL "when in doubt" for safety.
    That bias is the right call for a genuinely ambiguous message, but by the time
    rule 6 runs, rules 3/4 have already ruled out booking and — via a deliberately
    broad ~80-term keyword list — symptom signal, so a short, keyword-free
    statement doesn't need to pay for, or risk, that biased guess; it's already
    been shown to have nothing to do with either. Deliberately more generous than
    message_classifier._heuristic_classify's own 3-word conversational cap: that
    function has no idea whether booking/symptom signal was already ruled out by
    the time it runs, so it has to stay conservative on its own; this rule already
    knows both were ruled out, so it can afford a wider net."""
    words = re.findall(r"[a-z0-9']+", message.lower())
    if not words or len(words) > _MAX_NO_SIGNAL_WORDS:
        return False
    non_word_chars = re.sub(r"[a-z0-9\s,'!.]", "", message.lower())
    return not non_word_chars


# The rule cascade below used to be a straight if/elif chain, numbered 0.5, 0.6, 1,
# 1.5, 1.8, 2, 3, 4, 5, 6 in the order they run. Those fractional numbers are a
# record of rules getting wedged between existing ones over time as new
# misrouting cases were reported live — which worked, but made the actual
# evaluation order something you had to read the whole function to reconstruct,
# and gave a new rule no natural place to declare where it belongs. Restructured
# into an explicit, ordered table instead: same rules, same order, same
# first-match-wins semantics — just declared as data instead of implied by
# where an `if` happens to sit in the function body. Every rule keeps its
# original name/number and full "reported live" rationale so nothing about WHY
# each rule exists is lost, and every existing test (test_router_ruleN_...)
# still exercises the exact same behavior since only the plumbing changed.
#
# Each rule function has the same signature — (message, history, last_role,
# last_content) -> str | None — even when it only needs a subset of those
# arguments, so the cascade below can treat every rule identically.


def _rule_0_5_department_recommendation(message, history, last_role, last_content):
    """The patient is asking for a department RECOMMENDATION ("what do you think",
    "isn't X a better idea"), checked before even Rule 1's marker continuity.
    Reported live: after a DOCTOR_OPTIONS_MARKER card was shown, this kind of
    message fell to Rule 1 and went straight to appointment_agent, which has no
    symptom awareness at all — it just showed availability for whatever
    department got named next, then hallucinated a "reason" for a follow-up
    question instead of reasoning from the patient's REAL described symptoms. A
    recommendation request must always reach symptom_agent's deterministic
    symptom-to-department check, regardless of what card was shown last turn."""
    if is_department_recommendation_request(message):
        return SYMPTOM_GENERAL
    return None


def _rule_0_6_department_scope_question(message, history, last_role, last_content):
    """A general "what does this specialty treat" scope question, checked before
    Rule 1's marker continuity and Rule 2's screening continuity so it wins
    regardless of what card or clarifying question came right before it. Reported
    live: "so what symptoms does dermatologist treats?" followed a clarifying
    question from symptom_agent, so screening continuity (Rule 2) kept routing it
    right back to symptom_agent, which has no way to answer a scope question and
    just produced its generic dead-end redirect. This is informational, not a
    symptom description or a recommendation request — belongs with
    general_info_agent, same as any other clinic-info question."""
    if is_department_scope_question(message):
        return GENERAL_INFO
    return None


# Deliberately narrower than message_classifier._LOGISTICS_KEYWORDS, which also
# includes "doctor"/"appointment"/"book"/"schedule"/"slot"/"clinic"/"department" —
# words that overlap with genuine booking/symptom vocabulary and would misfire this
# rule on a real mid-screening booking message. Scoped to words that are ONLY ever
# about clinic logistics (hours, location, fees, policies, contact info) and have no
# plausible symptom/booking-action meaning at all.
_CLINIC_LOGISTICS_WORDS = frozenset({
    "hour", "hours", "timing", "timings", "open", "opens", "opening", "opened",
    "close", "closes", "closing", "closed",
    "location", "located", "address", "direction", "directions", "parking",
    "fee", "fees", "price", "prices", "cost", "costs", "charge", "charges", "insurance",
    "policy", "policies", "contact", "phone", "email", "holiday", "holidays", "weekend", "weekends",
})


def _message_is_clinic_logistics_question(message: str) -> bool:
    words = set(re.findall(r"[a-z0-9']+", message.lower()))
    return bool(words & _CLINIC_LOGISTICS_WORDS)


def _rule_0_7_clinic_logistics_question(message, history, last_role, last_content):
    """A plain clinic-logistics question (hours, location, fees, policies, contact
    info...), checked before rule 1's marker continuity and rule 2's screening
    continuity — same positioning/reasoning as rule 0.6 just above, for the same
    class of problem, one category over.

    Reported live: "i have headache" -> "how severe, how long?" -> "mild" ->
    "are you experiencing nausea, visual changes, fever?" -> "where is this clinic
    located" still got routed to SYMPTOM_GENERAL and refused as off-topic ("I don't
    have that information... contact the clinic directly"). Root cause: rule 2's
    screening-continuity check only asks "did the preceding assistant turn look
    like a question" (true for ANY question-shaped reply, including its own
    screening question) and "is the symptom context still more recent than booking
    context" (trivially true here — nothing booking-related happened at all) — it
    has no way to recognize the CURRENT message is an unrelated, self-contained
    clinic question rather than an answer to the screening question at all. A
    logistics question must always win here, regardless of what clarifying
    question came right before it, same principle rule 0.6 already applies for
    department-scope questions."""
    if _message_is_clinic_logistics_question(message) and not is_symptom_message(message):
        return GENERAL_INFO
    return None


def _message_states_a_cancel_or_reschedule_action(message: str) -> bool:
    """Deliberately just the raw keyword check (_CANCEL_ACTION_WORDS/
    _RESCHEDULE_ACTION_WORDS, the same shared vocabulary appointment_agent's own
    _detect_action_intent scans) — NOT needs_booking_action_tools(), which also has
    its own "preceding assistant turn looks like a question" fallback trigger. That
    fallback is exactly why rule 3 (which reuses needs_booking_action_tools) sits
    AFTER rule 2 in the first place (see rule 3's own docstring) — reusing it here,
    before rule 2, would make a plain screening-answer like "mild" (a reply to a
    question) misfire this rule too. Only the CURRENT message's own explicit
    cancel/reschedule wording should be able to jump the queue this early."""
    words = set(re.findall(r"[a-z0-9']+", message.lower()))
    return bool(words & (_CANCEL_ACTION_WORDS | _RESCHEDULE_ACTION_WORDS))


def _rule_0_8_cancel_reschedule_action_overrides_screening_continuity(message, history, last_role, last_content):
    """An explicit cancel/reschedule request, checked before rule 2's screening
    continuity — same positioning/reasoning as rules 0.6/0.7 just above, one more
    category over.

    Reported live: "i have headache" -> "how severe, how long?" -> "mild" ->
    "are you experiencing nausea, visual changes, fever?" -> "cancel my
    appointment" / "i want to reschedule my appointment" all still got routed to
    SYMPTOM_GENERAL instead of appointment_agent. Root cause: rule 2 fires purely
    because the preceding assistant turn looks like a question and nothing
    booking-related has happened yet (trivially true with zero prior booking
    turns) — it never checks whether the CURRENT message itself is an explicit
    booking action. Rule 3 already exists to catch exactly this (via
    needs_booking_action_tools), but sits AFTER rule 2 in the cascade — and can't
    simply be reordered ahead of it, since needs_booking_action_tools' own
    "preceding turn looks like a question" fallback would then misfire on a plain
    screening-answer reply like "mild" and send it to appointment_agent instead of
    continuing triage. This narrower rule uses only the CURRENT message's own
    explicit keyword instead, so it's safe to check this early."""
    if _message_states_a_cancel_or_reschedule_action(message):
        return APPOINTMENT
    return None


def _rule_0_9_appointment_status_history_query_overrides_screening_continuity(
    message, history, last_role, last_content
):
    """A "when was my most recent cancelled/missed/completed appointment"-style
    query, checked before rule 2's screening continuity — same category as rule
    0.8 just above (an appointment_agent-only capability getting swallowed by
    screening continuity), but a genuinely different underlying signal: a status
    HISTORY question, not an action word, so it needs its own detector rather than
    extending rule 0.8's keyword check.

    Reported live (follow-up to the rule 0.8 report): "when was my last cancelled
    appointment" happened to already work after rule 0.8 shipped, purely by
    coincidence — "cancelled" is also a literal member of _CANCEL_ACTION_WORDS, so
    it tripped rule 0.8 despite that rule being about action words, not status
    queries. "what was my last completed appointment" / "when was my most recent
    missed appointment" — same category of question, no cancel/reschedule keyword
    at all — still fell through to rule 2 and got routed to SYMPTOM_GENERAL.
    Reuses appointment_agent's own _message_asks_for_most_recent_appointment_by_status
    (the exact function that answers this deterministically from real DB rows once
    it reaches appointment_agent) rather than re-implementing the recency+status
    detection here — same "one place to define what this question shape means"
    principle every other reused-signal rule in this cascade already follows."""
    if _message_asks_for_most_recent_appointment_by_status(message) is not None:
        return APPOINTMENT
    return None


def _rule_1_marker_continuity(message, history, last_role, last_content):
    """The preceding turn was a slot-pick card or an appointment-agent
    disambiguation question — normally unambiguous, "continue the appointment
    flow," EXCEPT when the CURRENT message itself plainly describes a symptom.
    Reported live: a Cardiology DOCTOR_OPTIONS card was shown for chest pain,
    then "i am also having some skin related issues" — an unprompted, genuinely
    NEW symptom, unrelated to any slot-pick — matched this rule and went
    straight to appointment_agent, which has no symptom-triage logic at all and
    just showed Dermatology availability with ZERO clarifying questions,
    skipping symptom_agent's screening backstop entirely (that backstop can't
    help if the message never reaches symptom_agent in the first place). The
    rule's own old assumption — "a reply to THAT is never itself a fresh
    symptom statement in practice" — is exactly what failed here; Rule 1.8
    below already exists for this exact case ("current message plainly states
    a symptom") but could never fire for a marker-preceded turn, since this
    rule always won first. A genuine slot-pick reply ("the 9am one", "Dr.
    Farooq please") contains no symptom keyword, so is_symptom_message stays
    False for it and this rule still applies normally.

    Now checked AFTER rule 1.5 (department-list request) rather than before it —
    see that rule's own docstring for the live report explaining why an explicit
    "how many departments" question must win over marker continuity too."""
    if (
        last_role == "assistant"
        and last_content.startswith((DOCTOR_OPTIONS_MARKER, DOCTOR_DISAMBIGUATION_MARKER))
        and not is_symptom_message(message)
    ):
        return APPOINTMENT
    return None


def _rule_1_5_department_list_request(message, history, last_role, last_content):
    """Explicit request for the full department list. Reported live (1st report):
    "what are the available depts" wasn't recognized at all — it contains
    "available", one of needs_booking_action_tools' own keywords, so without this
    rule it would fall to rule 3 below and reach appointment_agent, which has no
    tool that can list every department (get_department_availability requires one
    specific name).

    Reported live (2nd report): checked before screening/booking continuity but
    NOT before rule 1 (marker continuity) — "how many total depts are there in
    this clinic" asked right after a DOCTOR_OPTIONS card was shown still matched
    rule 1 first (not a symptom message) and got routed to appointment_agent,
    which has no department-LIST capability at all and just re-showed the same
    stale card instead of answering. Moved ahead of rule 1 in the cascade so an
    explicit, unambiguous new department-list request wins regardless of what
    card was shown a moment ago — same principle rules 0.5/0.6 already apply for
    their own categories."""
    if is_department_list_request(message):
        return GENERAL_INFO
    return None


def _rule_1_6_doctor_count_or_listing_request(message, history, last_role, last_content):
    """A "how many doctors/cardiologists are there" question about a specific
    department or specialty. Reported live: "how many cardiologist are there in
    this clinic??" contains no booking-action keyword and isn't a symptom, so it
    fell through to general_info_agent, which answered from static KB prose
    instead of the real, current doctor count — and its follow-up, "show me
    their information", then got an incomplete answer since the KB document
    didn't happen to mention the second doctor. appointment_agent has the real,
    live per-department doctor lookup this needs (get_department_availability);
    general_info_agent only has KB text search, which can't be guaranteed
    complete or current for something admin-managed like the doctor roster."""
    if is_doctor_count_or_listing_request(message):
        return APPOINTMENT
    return None


def _rule_1_8_current_message_states_a_symptom(message, history, last_role, last_content):
    """The CURRENT message itself plainly describes a symptom. Reported live:
    General Medicine was resolved for dizziness/fainting, then small talk
    ("okay thank you" / "one more question"), then the assistant's own generic
    filler reply ("Sure, what would you like to know? ... doctors...") — its
    trailing "?" plus mentioning "doctors" tripped rule 2/3's booking-continuity
    heuristics (booking context was more recent than the last symptom turn), so
    "i am also having pain in my chest" — an unambiguous NEW symptom statement —
    was sent to appointment_agent instead of symptom_agent, skipping screening
    entirely. A message that itself plainly states a symptom must win regardless
    of what the immediately preceding turn's phrasing happened to look like.
    Rule 1 above now already excludes this case (a marker-preceded turn that
    itself plainly describes a symptom no longer matches Rule 1 at all), so in
    practice this rule is what actually catches it; kept as its own named rule
    since it also covers every OTHER non-marker "preceding turn" shape."""
    if is_symptom_message(message):
        return SYMPTOM_GENERAL
    return None


def _rule_2_screening_continuity(message, history, last_role, last_content):
    """Screening continuity (item 5): a reply to a plain clarifying question (not
    one of the markers above) stays with symptom_agent ONLY when the most recent
    relevant context is still symptom-side, not booking-side."""
    if _preceding_assistant_turn_looks_like_a_question(history) and _symptom_context_more_recent_than_booking_context(
        history
    ):
        return SYMPTOM_GENERAL
    return None


def _rule_3_booking_action_signal(message, history, last_role, last_content):
    """Explicit booking-action signal. Reuses needs_booking_action_tools() exactly
    as-is, including its own "preceding turn is a question" trigger — safe to
    reuse here because rule 2 already carved out the one case where a
    question-reply should NOT default to appointment."""
    if needs_booking_action_tools(message, history):
        return APPOINTMENT
    return None


def _rule_4_symptom_signal_anywhere(message, history, last_role, last_content):
    """Symptom signal: the CURRENT message, or — only when that symptom mention is
    still more recent than any booking-context turn since (see
    _symptom_context_more_recent_than_booking_context, the exact same
    recency-comparison infra Rule 2 already uses, see the module's own "RECENCY
    FIX" docstring) — a turn earlier in the conversation.

    Reported live: "i am having pain in chest" (turn 1) -> symptom screening ->
    "cancel my upcoming appointment" / "reschedule my upcoming appointment" (turns
    2-4, clearly booking-related, correctly handled by appointment_agent) ->
    "whats the clinic hours?" (turn 5) — a plain, self-contained, keyword-clean
    informational question with nothing to do with symptoms or booking — still got
    routed to SYMPTOM_GENERAL and refused as off-topic ("I don't have that
    information... contact the clinic directly"), because this rule's history scan
    had NO recency bound at all: once any symptom mention was on record, every
    later turn that rules 0.5-3 didn't otherwise claim fell to symptom_agent
    forever, even after the conversation had since moved through several unrelated
    booking turns. Rule 2 solved exactly this class of problem for its own
    (screening-continuity) case years ago; this rule just never got the same
    treatment. Only guards against booking context specifically (not e.g. several
    turns of unrelated small talk with no booking signal at all) — narrowly scoped
    to the reported failure mode rather than a broader rewrite."""
    if is_symptom_message(message):
        return SYMPTOM_GENERAL
    if _most_recent_symptom_turn_index(history) >= 0 and _symptom_context_more_recent_than_booking_context(history):
        return SYMPTOM_GENERAL
    return None


def _rule_5_classic_conversational_heuristic(message, history, last_role, last_content):
    """Plain conversational/personal-recall/logistics phrasing: reuse
    message_classifier's own heuristic; anything it can decide maps to
    GENERAL_INFO (small talk and personal-recall are both handled fine by
    general_info_agent's prompt, same as get_chat_reply did before it)."""
    if _classic_heuristic_classify(message, history) is not None:
        return GENERAL_INFO
    return None


def _rule_6_signal_free_short_statement(message, history, last_role, last_content):
    """Signal-free short statement: reported live, "my name is daud" (4 words, no
    symptom/booking keyword) fell through every rule above and got classified
    SYMPTOM_GENERAL by the biased LLM fallback below, attaching the entire
    triage prompt to a message with nothing to do with symptoms. See
    _looks_like_signal_free_short_statement's docstring for why this is safe."""
    if _looks_like_signal_free_short_statement(message):
        return GENERAL_INFO
    return None


# First-match-wins, in this exact order — see each rule function's docstring for
# why it sits where it does relative to its neighbors.
_RULE_CASCADE = (
    ("0.5", _rule_0_5_department_recommendation),
    ("0.6", _rule_0_6_department_scope_question),
    ("0.7", _rule_0_7_clinic_logistics_question),
    ("0.8", _rule_0_8_cancel_reschedule_action_overrides_screening_continuity),
    ("0.9", _rule_0_9_appointment_status_history_query_overrides_screening_continuity),
    ("1.5", _rule_1_5_department_list_request),
    ("1", _rule_1_marker_continuity),
    ("1.6", _rule_1_6_doctor_count_or_listing_request),
    ("1.8", _rule_1_8_current_message_states_a_symptom),
    ("2", _rule_2_screening_continuity),
    ("3", _rule_3_booking_action_signal),
    ("4", _rule_4_symptom_signal_anywhere),
    ("5", _rule_5_classic_conversational_heuristic),
    ("6", _rule_6_signal_free_short_statement),
)


def _heuristic_classify(message: str, history=None) -> str | None:
    """Fast, free pre-filter — returns a definitive classification for clear-cut
    cases, or None when genuinely ambiguous (falls through to _llm_classify)."""
    history = history or []
    last = history[-1] if history else None
    last_role = getattr(last, "role", None) if last else None
    last_content = (getattr(last, "content", "") or "") if last else ""

    for _name, rule in _RULE_CASCADE:
        result = rule(message, history, last_role, last_content)
        if result is not None:
            return result
    return None


_CLASSIFY_SYSTEM_PROMPT = """Classify the patient's message into exactly one of:

SYMPTOM_GENERAL: describes a symptom, injury, or health concern, or is part of an \
ongoing conversation about one — even if this specific message doesn't itself name \
a symptom.

APPOINTMENT: about booking, rescheduling, cancelling, or checking an existing \
appointment — including picking a specific doctor/slot already shown, or \
continuing an appointment-related conversation.

GENERAL_INFO: a general question about the clinic (hours, location, fees, \
policies), small talk, or a question about something the patient themselves said \
earlier — not about a symptom and not about an appointment action.

When in doubt, or if there's ANY chance this could be about a symptom, answer \
SYMPTOM_GENERAL — it is always safer to over-classify toward symptom triage than to \
miss a real health concern.

Respond with exactly one word: SYMPTOM_GENERAL, APPOINTMENT, or GENERAL_INFO."""


def _llm_classify(message: str) -> str:
    if not settings.LLM_API_KEY:
        return SYMPTOM_GENERAL

    try:
        llm = ChatGroq(model=settings.GROQ_HELPER_MODEL, api_key=api_key_manager.next_key(), temperature=0.0)
        raw = llm.invoke(
            [SystemMessage(content=_CLASSIFY_SYSTEM_PROMPT), HumanMessage(content=message)]
        ).content.strip().upper()
    except Exception:
        logger.exception("Intent classification failed, defaulting to symptom_general")
        return SYMPTOM_GENERAL

    if "APPOINTMENT" in raw and "SYMPTOM" not in raw:
        return APPOINTMENT
    if "GENERAL_INFO" in raw and "SYMPTOM" not in raw:
        return GENERAL_INFO
    return SYMPTOM_GENERAL


def classify_agent_intent(message: str, history=None) -> str:
    """Entry point: SYMPTOM_GENERAL / APPOINTMENT / GENERAL_INFO. Most messages are
    classified for free by _heuristic_classify; only genuine ambiguity pays for an
    LLM call, same shape as message_classifier.classify_message_intent."""
    heuristic_result = _heuristic_classify(message, history)
    if heuristic_result is not None:
        return heuristic_result
    return _llm_classify(message)
