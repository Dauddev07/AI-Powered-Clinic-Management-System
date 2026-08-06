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
    _preceding_assistant_turn_looks_like_a_question,
    is_department_list_request,
    is_department_recommendation_request,
    is_department_scope_question,
    is_symptom_message,
    needs_booking_action_tools,
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


def _heuristic_classify(message: str, history=None) -> str | None:
    """Fast, free pre-filter — returns a definitive classification for clear-cut
    cases, or None when genuinely ambiguous (falls through to _llm_classify)."""
    history = history or []
    last = history[-1] if history else None
    last_role = getattr(last, "role", None) if last else None
    last_content = (getattr(last, "content", "") or "") if last else ""

    # Rule 0.5 — the patient is asking for a department RECOMMENDATION ("what do you
    # think", "isn't X a better idea"), checked before even Rule 1's marker
    # continuity. Reported live: after a DOCTOR_OPTIONS_MARKER card was shown, this
    # kind of message fell to Rule 1 and went straight to appointment_agent, which
    # has no symptom awareness at all — it just showed availability for whatever
    # department got named next, then hallucinated a "reason" for a follow-up
    # question instead of reasoning from the patient's REAL described symptoms. A
    # recommendation request must always reach symptom_agent's deterministic
    # symptom-to-department check, regardless of what card was shown last turn.
    if is_department_recommendation_request(message):
        return SYMPTOM_GENERAL

    # Rule 0.6 — a general "what does this specialty treat" scope question, checked
    # before Rule 1's marker continuity and Rule 2's screening continuity so it wins
    # regardless of what card or clarifying question came right before it. Reported
    # live: "so what symptoms does dermatologist treats?" followed a clarifying
    # question from symptom_agent, so screening continuity (Rule 2) kept routing it
    # right back to symptom_agent, which has no way to answer a scope question and
    # just produced its generic dead-end redirect. This is informational, not a
    # symptom description or a recommendation request — belongs with
    # general_info_agent, same as any other clinic-info question.
    if is_department_scope_question(message):
        return GENERAL_INFO

    # Rule 1 — unambiguous: the preceding turn was a slot-pick card or an
    # appointment-agent disambiguation question. Either can only ever mean
    # "continue the appointment flow" — never inferred from wording.
    if last_role == "assistant" and last_content.startswith((DOCTOR_OPTIONS_MARKER, DOCTOR_DISAMBIGUATION_MARKER)):
        return APPOINTMENT

    # Rule 1.5 — explicit request for the full department list. Reported live:
    # "what are the available depts" wasn't recognized at all — it contains
    # "available", one of needs_booking_action_tools' own keywords, so without this
    # rule it would fall to rule 3 below and reach appointment_agent, which has no
    # tool that can list every department (get_department_availability requires one
    # specific name). Checked before screening/booking continuity since it's an
    # explicit, unambiguous new request that should win regardless of what the
    # conversation was doing a moment ago.
    if is_department_list_request(message):
        return GENERAL_INFO

    # Rule 1.8 — the CURRENT message itself plainly describes a symptom. Reported
    # live: General Medicine was resolved for dizziness/fainting, then small talk
    # ("okay thank you" / "one more question"), then the assistant's own generic
    # filler reply ("Sure, what would you like to know? ... doctors...") — its
    # trailing "?" plus mentioning "doctors" tripped rule 2/3's booking-continuity
    # heuristics (booking context was more recent than the last symptom turn), so
    # "i am also having pain in my chest" — an unambiguous NEW symptom statement —
    # was sent to appointment_agent instead of symptom_agent, skipping screening
    # entirely. A message that itself plainly states a symptom must win regardless
    # of what the immediately preceding turn's phrasing happened to look like.
    # Checked after Rule 1 (an actual slot-pick card is still unambiguous — a
    # reply to THAT is never itself a fresh symptom statement in practice).
    if is_symptom_message(message):
        return SYMPTOM_GENERAL

    # Rule 2 — screening continuity (item 5): a reply to a plain clarifying
    # question (not one of the markers above) stays with symptom_agent ONLY when
    # the most recent relevant context is still symptom-side, not booking-side.
    if _preceding_assistant_turn_looks_like_a_question(history) and _symptom_context_more_recent_than_booking_context(
        history
    ):
        return SYMPTOM_GENERAL

    # Rule 3 — explicit booking-action signal. Reuses needs_booking_action_tools()
    # exactly as-is, including its own "preceding turn is a question" trigger —
    # safe to reuse here because rule 2 already carved out the one case where a
    # question-reply should NOT default to appointment.
    if needs_booking_action_tools(message, history):
        return APPOINTMENT

    # Rule 4 — symptom signal, this message or earlier in the conversation.
    if is_symptom_message(message) or _most_recent_symptom_turn_index(history) >= 0:
        return SYMPTOM_GENERAL

    # Rule 5 — plain conversational/personal-recall/logistics phrasing: reuse
    # message_classifier's own heuristic; anything it can decide maps to
    # GENERAL_INFO (small talk and personal-recall are both handled fine by
    # general_info_agent's prompt, same as get_chat_reply did before it).
    if _classic_heuristic_classify(message, history) is not None:
        return GENERAL_INFO

    # Rule 6 — signal-free short statement: reported live, "my name is daud" (4
    # words, no symptom/booking keyword) fell through every rule above and got
    # classified SYMPTOM_GENERAL by the biased LLM fallback below, attaching the
    # entire triage prompt to a message with nothing to do with symptoms. See
    # _looks_like_signal_free_short_statement's docstring for why this is safe.
    if _looks_like_signal_free_short_statement(message):
        return GENERAL_INFO

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
