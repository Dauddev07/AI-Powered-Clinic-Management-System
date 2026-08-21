"""Orchestrates one chat turn: retrieve -> ground -> reply -> persist.

Memory is stored against the patient account from the very first message (not from
first booking, not after a threshold) — every turn, including the very first one in a
brand-new session, is written to conversation_memory before this function returns.

session_id IS a hard memory boundary: a continuing session (one that already has at
least one message) replays its OWN messages only, most-recent-first, capped at
HISTORY_MESSAGE_LIMIT — never another session's. Starting a genuinely new session gets
a completely empty transcript and no cross-session digest of any kind — "New Chat"
means a real fresh start with zero memory of anything said in a previous session, both
for a smaller/cheaper prompt and because a patient reasonably expects a new chat to not
carry old context forward. app.services.memory_summary still exists (its table/reset
path is used by delete_session below) but is no longer read from or written to here.
"""
import re
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.tenancy import ClinicContext
from app.models.conversation_memory import ConversationMemory
from app.services.chat_markers import (
    BOOKING_MARKER,
    DEPARTMENT_LIST_MARKER,
    DOCTOR_DISAMBIGUATION_MARKER,
    DOCTOR_OPTIONS_MARKER,
)
from app.services.diagnosis_guard import enforce_no_diagnosis
from app.services.language import detect_language
from app.services.memory_summary import reset_patient_summary
from app.services.message_classifier import is_symptom_message
from app.services.orchestrator.agents.appointment_agent import _detect_action_intent, run_appointment_agent
from app.services.orchestrator.agents.general_info_agent import run_general_info_agent
from app.services.orchestrator.agents.symptom_agent import _looks_like_a_valid_emergency_reply, run_symptom_agent
from app.services.orchestrator.router import (
    APPOINTMENT,
    GENERAL_INFO,
    SYMPTOM_GENERAL,
    _message_is_clinic_logistics_question,
    classify_agent_intent,
)
from app.services.red_flag import detect_red_flag, red_flag_message

# How many prior turns (user+assistant messages, not pairs) of THIS SAME SESSION are
# LOADED FROM THE DATABASE, most-recent-first. This is the outer bound everything else
# below trims from — kept generous because app.services.orchestrator.router's own
# recency-based rules (deciding whether a stale, long-ago symptom mention should still
# outrank an active booking flow — see router._symptom_context_more_recent_than_
# booking_context) need real visibility into the conversation, and that classification
# is a free heuristic (no LLM cost), so there's no token-cost reason to shrink it.
HISTORY_MESSAGE_LIMIT = 30

# Part C — per-agent history limits: once the router has decided which specialist
# handles this turn (using the FULL HISTORY_MESSAGE_LIMIT window above), the specialist
# itself is replayed a smaller, most-recent-only slice before its own LLM call —
# each agent's real working context is much shorter than what the router needs to see.
# Applied via _trim_history() right before each run_*_agent() call below.
#
# symptom_agent: PATH 2 screening is capped at exactly one round and PATH 3 at 3
# questions (see _TRIAGE_PATH2/_TRIAGE_ALWAYS in llm.py), so a single triage
# conversation rarely spans more than ~8-10 messages — 16 leaves headroom for the
# MULTIPLE DISTINCT SYMPTOMS/COMPLAINTS rule, which needs to recall an earlier-
# mentioned, not-yet-addressed complaint from a bit further back.
SYMPTOM_AGENT_HISTORY_LIMIT = 16

# appointment_agent: the handoff mechanism only ever needs the MOST RECENT
# DOCTOR_OPTIONS/DEPARTMENT_LIST marker card (see appointment_agent.
# _most_recent_availability_marker) — a booking flow that's still active is always
# recent, so 10 is ample without paying to replay a much longer transcript.
APPOINTMENT_AGENT_HISTORY_LIMIT = 10

# general_info_agent: KB question-answering doesn't depend on conversation history
# much (query_rewrite only rescues a raw retrieval miss), so this only needs enough
# for query_rewrite's own immediate-context rescue and general conversational
# continuity.
GENERAL_INFO_AGENT_HISTORY_LIMIT = 8


def _trim_history(history: list[ConversationMemory], limit: int) -> list[ConversationMemory]:
    """`history` is most-recent-LAST (see _load_recent_history) — the most recent
    `limit` messages are the last `limit` elements, not the first."""
    return history[-limit:] if len(history) > limit else history

# How many sessions (threads) are listed in the sidebar, most recently active first.
SESSION_LIST_LIMIT = 50

# How long a session's title (derived from its first user message) can be before
# truncation, so the sidebar row never wraps to more than one line.
SESSION_TITLE_MAX_LENGTH = 60


@dataclass
class ChatTurnResult:
    session_id: uuid.UUID
    reply: str
    red_flag: bool


@dataclass
class ChatSessionSummary:
    session_id: uuid.UUID
    title: str
    last_message_at: datetime


def _load_recent_history(db: Session, ctx: ClinicContext, session_id: uuid.UUID) -> list[ConversationMemory]:
    """Scoped to THIS session only — see module docstring."""
    stmt = (
        select(ConversationMemory)
        .where(
            ConversationMemory.clinic_id == ctx.clinic_id,
            ConversationMemory.user_id == ctx.user_id,
            ConversationMemory.session_id == session_id,
        )
        # created_at alone isn't a strict order: a single chat turn saves its user +
        # assistant rows in the same transaction, and Postgres now() returns the same
        # value for every statement within one transaction, so those two rows tie.
        # `seq` (a real auto-incrementing identity column) breaks the tie
        # deterministically, in true insertion order — same approach as
        # get_chat_history() in app/api/chat.py, against the same table.
        .order_by(ConversationMemory.created_at.desc(), ConversationMemory.seq.desc())
        .limit(HISTORY_MESSAGE_LIMIT)
    )
    rows = list(db.execute(stmt).scalars().all())
    rows.reverse()
    return rows


def _session_has_messages(db: Session, ctx: ClinicContext, session_id: uuid.UUID) -> bool:
    return (
        db.execute(
            select(ConversationMemory.id)
            .where(
                ConversationMemory.clinic_id == ctx.clinic_id,
                ConversationMemory.user_id == ctx.user_id,
                ConversationMemory.session_id == session_id,
            )
            .limit(1)
        ).first()
        is not None
    )


def _save_message(
    db: Session, ctx: ClinicContext, session_id: uuid.UUID, role: str, content: str, red_flag: bool = False
) -> None:
    db.add(
        ConversationMemory(
            clinic_id=ctx.clinic_id,
            session_id=session_id,
            user_id=ctx.user_id,
            role=role,
            content=content,
            red_flag=red_flag,
        )
    )
    db.flush()


def _is_marker_reply(reply: str) -> bool:
    """Marker-prefixed replies (booking cards, doctor-option cards, combined
    department-list cards, appointment_agent's disambiguation cards) are composed
    deterministically from real DB rows in app.services.chat_tools/
    app.services.orchestrator.agents.appointment_agent, never freehanded by the LLM
    — running the diagnosis-phrasing guard's regex over their raw JSON payload would
    risk corrupting a valid card for no safety benefit, so those are passed straight
    through untouched."""
    return reply.startswith((BOOKING_MARKER, DOCTOR_OPTIONS_MARKER, DEPARTMENT_LIST_MARKER, DOCTOR_DISAMBIGUATION_MARKER))


# GUARD PIPELINE — red_flag (checks the incoming message, can short-circuit the
# whole turn before any agent runs) and diagnosis_guard (checks/rewrites the
# agent's own reply afterward) are both "a prompt instruction alone is not a
# guarantee, so a plain deterministic check runs regardless of what the model
# would have done" — the same idea applied at two different points in the turn.
# They used to be two individually-wired calls inside handle_chat_message; now
# each stage is an ordered tuple of guard functions, so a second guard at either
# stage (e.g. a future abuse/rate-limit check before the agent runs) is a matter
# of adding one function to a tuple, not editing this function's control flow
# again. Behavior for the two guards that exist today is unchanged either way.
@dataclass
class GuardOutcome:
    reply: str
    red_flag: bool = False


def _red_flag_guard(message: str) -> GuardOutcome | None:
    """Emergency detection runs before anything else — a real emergency
    short-circuits triage, booking, and KB retrieval entirely and returns the
    urgent routing message immediately. A prompt instruction alone is not a
    guarantee, so this is a plain server-side check (regex, bare-severity, and
    semantic layers — see app.services.red_flag's own module docstring), not
    something the LLM is asked to decide."""
    if detect_red_flag(message):
        return GuardOutcome(reply=red_flag_message(), red_flag=True)
    return None


# Pre-agent guards: each takes the raw incoming message and can fully resolve
# the turn (return a GuardOutcome) before routing/the specialist agent ever
# runs. First non-None result wins; None means "not this guard's concern, try
# the next one" — same first-match-wins shape as router._RULE_CASCADE.
_PRE_AGENT_GUARDS: tuple = (_red_flag_guard,)


def _run_pre_agent_guards(message: str) -> GuardOutcome | None:
    for guard in _PRE_AGENT_GUARDS:
        outcome = guard(message)
        if outcome is not None:
            return outcome
    return None


# Requested: a single message stating two genuinely distinct things — a real
# symptom alongside an unrelated cancel/reschedule request, or a symptom
# alongside a clinic-logistics question — used to have one of the two
# silently vanish entirely, since the router picks exactly one bucket and the
# agent it hands off to has no awareness of the other thing at all (e.g. "my
# chest really hurts, please cancel my appointment tomorrow" landed on
# appointment_agent, which has zero symptom-screening code, so the chest pain
# was never looked at). Deliberately scoped to just these two concrete
# combinations reported live, not every conceivable pairing — booking a NEW
# appointment BECAUSE of a symptom ("I have a headache, can you book me an
# appointment") is one coherent ask, not two, and correctly stays untouched
# here. A genuine emergency is a separate, higher-priority guard (see
# _red_flag_guard above, which always runs first and short-circuits
# everything below regardless of what else is in the message) — this only
# ever engages for the non-emergency case.
_COMPOUND_INTENT_LABELS = {
    "symptom": "your symptom",
    "cancel_reschedule": "cancelling or rescheduling your appointment",
    "logistics": "your question about the clinic",
}
# Distinctive substring of _compound_intent_reply's own question — checked
# against the assistant's last turn to recognize a reply as answering THIS
# question specifically, same "fixed signature phrase" pattern
# general_info_agent's own _DEPARTMENT_LIST_REPLY_SIGNATURE uses, rather than
# a JSON marker: this question is plain conversational text with no
# candidate buttons to render, so a marker would only show as raw prefixed
# text to the patient with no frontend support for it.
_COMPOUND_INTENT_QUESTION_SIGNATURE = "which would you like me to help with first"


def _detect_compound_intent(message: str) -> tuple[str, str] | None:
    """Returns (label_a, label_b) naming the two distinct things detected in
    `message`, or None when only one (or zero) applies.

    Reported live: "no no not cancel, symptoms" (a patient correcting their
    own earlier "cancelling" answer) still saw a compound message and re-asked
    the same clarifying question — router._message_states_a_cancel_or_
    reschedule_action is a bare keyword check with no negation awareness (by
    its own docstring, deliberately, for its own different use case in the
    router cascade), so the literal word "cancel" inside "not cancel" still
    counted. appointment_agent._detect_action_intent is the negation-aware
    version already built for exactly this (walks tokens in order, skips a
    negated match via _action_word_is_negated) — used here instead so a
    correction/retraction is read the same way a human would."""
    if not is_symptom_message(message):
        return None
    if _detect_action_intent(message) in ("cancel", "reschedule"):
        return ("symptom", "cancel_reschedule")
    if _message_is_clinic_logistics_question(message):
        return ("symptom", "logistics")
    return None


def _compound_intent_reply(pair: tuple[str, str]) -> str:
    label_a, label_b = pair
    return (
        f"It looks like your message mentions more than one thing — {_COMPOUND_INTENT_LABELS[label_a]} "
        f"and {_COMPOUND_INTENT_LABELS[label_b]}. {_COMPOUND_INTENT_QUESTION_SIGNATURE.capitalize()}?"
    )


def _label_matches_choice(message: str, label: str) -> bool:
    """Reuses the exact same detectors that identified this label in the
    original message in the first place, rather than inventing a separate set
    of "choice" keywords that could drift out of sync with them — if the
    patient just repeats the relevant word ("cancel", "clinic hours") in
    their answer, that alone already means this label."""
    if label == "symptom":
        return is_symptom_message(message) or bool(re.search(r"\bsymptoms?\b", message.lower()))
    if label == "cancel_reschedule":
        # Negation-aware (see _detect_compound_intent's own comment) — "not
        # cancel, symptoms" must NOT match this label just because the word
        # "cancel" appears in it.
        return _detect_action_intent(message) in ("cancel", "reschedule") or bool(
            re.search(r"\bappointment\b", message.lower())
        )
    if label == "logistics":
        return _message_is_clinic_logistics_question(message) or bool(re.search(r"\bclinic\b", message.lower()))
    return False


def _resolve_pending_compound_intent_choice(
    message: str, history: list[ConversationMemory]
) -> tuple[str, str, list[ConversationMemory]] | None:
    """If the assistant's last turn was _compound_intent_reply's own question
    and this message clearly answers it, returns (forced_intent,
    original_compound_message, history_before_that_exchange) — the ORIGINAL
    message is re-run through just the chosen agent (ignoring the other thing
    it mentioned), rather than ever trying to treat "symptom first" itself as
    a real symptom description.

    Returns None whenever this message doesn't clearly answer that question —
    including a genuinely unrelated message, or the pending question having
    gone stale — so the caller falls through to completely normal
    classification of the CURRENT message. Deliberately NOT special-cased
    further than that: an unrelated reply superseding a stale question is the
    same principle every other pending-question check in this codebase
    already follows (see appointment_agent's own pending-disambiguation
    handling), not something unique to this guard."""
    if len(history) < 2:
        return None
    last = history[-1]
    if getattr(last, "role", None) != "assistant":
        return None
    if _COMPOUND_INTENT_QUESTION_SIGNATURE not in (getattr(last, "content", "") or "").lower():
        return None
    original = history[-2]
    if getattr(original, "role", None) != "user":
        return None
    original_message = getattr(original, "content", "") or ""
    pair = _detect_compound_intent(original_message)
    if pair is None:
        return None
    label_a, label_b = pair
    matched = [label for label in (label_a, label_b) if _label_matches_choice(message, label)]
    if len(matched) != 1:
        # Neither label matched (a genuinely unrelated reply) or both did
        # (still ambiguous) — either way, not a clear answer to this specific
        # question, so let it fall through to normal classification instead
        # of guessing.
        return None
    label_to_intent = {"symptom": SYMPTOM_GENERAL, "cancel_reschedule": APPOINTMENT, "logistics": GENERAL_INFO}
    return label_to_intent[matched[0]], original_message, history[:-2]


def _diagnosis_phrasing_guard(reply: str, language: str) -> str:
    """Server-side regex backstop that rewrites diagnostic-sounding phrasing out
    of the agent's own reply — skipped entirely for a marker-prefixed reply (see
    _is_marker_reply's docstring)."""
    if _is_marker_reply(reply):
        return reply
    return enforce_no_diagnosis(reply, language)


# Post-agent guards: each takes the specialist agent's reply and returns a
# (possibly rewritten) reply, applied in order. Today there's only one; a
# second post-agent guard is one more entry in this tuple.
_POST_AGENT_GUARDS: tuple = (_diagnosis_phrasing_guard,)


def _run_post_agent_guards(reply: str, language: str) -> str:
    for guard in _POST_AGENT_GUARDS:
        reply = guard(reply, language)
    return reply


def handle_chat_message(
    db: Session, ctx: ClinicContext, message: str, session_id: uuid.UUID | None
) -> ChatTurnResult:
    session_id = session_id or uuid.uuid4()
    language = detect_language(message)

    pre_guard_outcome = _run_pre_agent_guards(message)
    if pre_guard_outcome is not None:
        _save_message(db, ctx, session_id, "user", message)
        _save_message(db, ctx, session_id, "assistant", pre_guard_outcome.reply, red_flag=pre_guard_outcome.red_flag)
        db.commit()
        return ChatTurnResult(session_id=session_id, reply=pre_guard_outcome.reply, red_flag=pre_guard_outcome.red_flag)

    # Loaded before the current turn is saved, so the prompt's history never
    # double-includes the message being answered right now.
    #
    # Reported live: patient memory is scoped to THIS session/chat only, by
    # design — a brand new chat must start completely fresh, with no digest of
    # anything said in a previous session. This also cuts prompt token size:
    # no cross-session summary text, and no per-new-session summarization LLM
    # call (see app.services.memory_summary, still present but no longer
    # invoked here). A continuing session's own real transcript is still loaded
    # in full below (unchanged) — only the cross-session digest is gone.
    #
    # The cross-session PATIENT MEMORY prompt section itself (and the patient_memory
    # parameter threaded through every agent) was removed entirely, not just left
    # empty — reported live: leaving the LLM prompt's detailed "how to talk about
    # stored memory" instructions in place while always feeding them an empty value
    # caused the model to misfire on unrelated messages (e.g. replying "I don't have
    # any previous information stored about you" to a patient simply saying "my name
    # is daud"), since the instructions primed it to reason about "stored
    # information" even when nothing in the actual message asked about it.
    if _session_has_messages(db, ctx, session_id):
        history = _load_recent_history(db, ctx, session_id)
    else:
        history = []

    # A reply that clearly answers OUR OWN compound-intent clarifying question
    # (see _resolve_pending_compound_intent_choice's own comment) re-runs the
    # ORIGINAL two-things message through just the chosen agent — never the
    # current "symptom first"-style answer itself, which has no real content
    # of its own to screen/act on. effective_message/effective_history are
    # local substitutions used only for the agent call below; the REAL
    # current message ("symptom first") is still what gets saved as this
    # turn's user message further down, an honest transcript either way.
    compound_choice = _resolve_pending_compound_intent_choice(message, history)
    if compound_choice is not None:
        forced_intent, effective_message, effective_history = compound_choice
    else:
        forced_intent, effective_message, effective_history = None, message, history
        # A FRESH compound message (not answering any pending question) gets
        # asked which one to handle first — but only when the assistant's own
        # last turn isn't already some OTHER live pending question this
        # module knows nothing about (a doctor/slot disambiguation, a cancel/
        # book/reschedule confirmation, an availability card) — those need to
        # reach their own owning agent untouched, never hijacked here just
        # because this message also happens to contain a symptom word.
        last = history[-1] if history else None
        last_content = (getattr(last, "content", "") or "") if last else ""
        last_is_other_pending_question = getattr(last, "role", None) == "assistant" and last_content.startswith(
            (BOOKING_MARKER, DOCTOR_OPTIONS_MARKER, DEPARTMENT_LIST_MARKER, DOCTOR_DISAMBIGUATION_MARKER)
        )
        if not last_is_other_pending_question:
            pair = _detect_compound_intent(message)
            if pair is not None:
                reply = _compound_intent_reply(pair)
                _save_message(db, ctx, session_id, "user", message)
                _save_message(db, ctx, session_id, "assistant", reply)
                db.commit()
                return ChatTurnResult(session_id=session_id, reply=reply, red_flag=False)

    # The orchestrator's intent layer replaces classify_message_intent() + the old
    # single agent entirely: exactly one specialist handles this turn, decided by
    # app.services.orchestrator.router.classify_agent_intent (free heuristic first,
    # one GROQ_HELPER_MODEL call only on genuine ambiguity — see its own docstring).
    # general_info_agent covers what get_chat_reply used to (small talk,
    # personal-recall, and real hospital_info KB questions all handled by its own
    # prompt's CONVERSATIONAL EXCEPTION + retrieval, same as before).
    intent = forced_intent if forced_intent is not None else classify_agent_intent(message, history)
    if intent == SYMPTOM_GENERAL:
        reply = run_symptom_agent(
            db, ctx, effective_message, language, _trim_history(effective_history, SYMPTOM_AGENT_HISTORY_LIMIT)
        )
    elif intent == APPOINTMENT:
        reply = run_appointment_agent(
            db, ctx, effective_message, language, _trim_history(effective_history, APPOINTMENT_AGENT_HISTORY_LIMIT)
        )
    else:
        assert intent == GENERAL_INFO
        reply = run_general_info_agent(
            db, ctx, effective_message, language, _trim_history(effective_history, GENERAL_INFO_AGENT_HISTORY_LIMIT)
        )

    reply = _run_post_agent_guards(reply, language)

    # A PATH 1 emergency reply from symptom_agent's own screening (a patient's
    # stated severity, e.g. "very severe", not the red_flag.py regex layer
    # above) was never persisted as red_flag=True at all — only the pre-guard
    # path was. That gap meant _session_had_an_earlier_emergency_flag
    # (symptom_agent.py) had to fall back to scanning for the "1122" text
    # signature instead of this column; fixed at the source here so the
    # column is accurate regardless of which layer flagged the emergency.
    is_red_flag_reply = intent == SYMPTOM_GENERAL and _looks_like_a_valid_emergency_reply(reply)

    _save_message(db, ctx, session_id, "user", message)
    _save_message(db, ctx, session_id, "assistant", reply, red_flag=is_red_flag_reply)
    db.commit()

    return ChatTurnResult(session_id=session_id, reply=reply, red_flag=is_red_flag_reply)


def _session_title(db: Session, ctx: ClinicContext, session_id: uuid.UUID) -> str:
    # The session's own first user message doubles as its sidebar title — no separate
    # "title" ever needs to be generated or stored.
    first_message = db.execute(
        select(ConversationMemory.content)
        .where(
            ConversationMemory.clinic_id == ctx.clinic_id,
            ConversationMemory.user_id == ctx.user_id,
            ConversationMemory.session_id == session_id,
            ConversationMemory.role == "user",
        )
        .order_by(ConversationMemory.created_at.asc())
        .limit(1)
    ).scalar()

    title = (first_message or "New conversation").strip()
    if len(title) > SESSION_TITLE_MAX_LENGTH:
        title = title[: SESSION_TITLE_MAX_LENGTH - 1].rstrip() + "…"
    return title


def list_sessions(db: Session, ctx: ClinicContext) -> list[ChatSessionSummary]:
    rows = db.execute(
        select(ConversationMemory.session_id, func.max(ConversationMemory.created_at).label("last_at"))
        .where(ConversationMemory.clinic_id == ctx.clinic_id, ConversationMemory.user_id == ctx.user_id)
        .group_by(ConversationMemory.session_id)
        .order_by(func.max(ConversationMemory.created_at).desc())
        .limit(SESSION_LIST_LIMIT)
    ).all()

    return [
        ChatSessionSummary(session_id=session_id, title=_session_title(db, ctx, session_id), last_message_at=last_at)
        for session_id, last_at in rows
    ]


def delete_session(db: Session, ctx: ClinicContext, session_id: uuid.UUID) -> bool:
    """Deletes a thread outright — its rows also drop out of every future
    cross-session history load, so a deleted chat is genuinely forgotten, not just
    hidden from the sidebar. Returns False if the session didn't exist (or belonged to
    someone else), so the caller can 404 rather than silently no-op.

    Also wipes the patient's cross-session memory digest (see
    app.services.memory_summary.reset_patient_summary) — otherwise a fact the digest
    already folded in from the deleted session (e.g. a name mentioned there) would
    keep surfacing in future new sessions even after the transcript it came from is
    gone, breaking the "genuinely forgotten" promise above. It's regenerated fresh
    from whatever sessions remain the next time a new session starts."""
    result = db.execute(
        delete(ConversationMemory).where(
            ConversationMemory.clinic_id == ctx.clinic_id,
            ConversationMemory.user_id == ctx.user_id,
            ConversationMemory.session_id == session_id,
        )
    )
    deleted = result.rowcount > 0
    if deleted:
        reset_patient_summary(db, ctx.clinic_id, ctx.user_id)
    db.commit()
    return deleted
