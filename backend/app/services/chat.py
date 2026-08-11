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
from app.services.orchestrator.agents.appointment_agent import run_appointment_agent
from app.services.orchestrator.agents.general_info_agent import run_general_info_agent
from app.services.orchestrator.agents.symptom_agent import run_symptom_agent
from app.services.orchestrator.router import APPOINTMENT, GENERAL_INFO, SYMPTOM_GENERAL, classify_agent_intent
from app.services.red_flag import detect_red_flag, detect_severity_escalation, red_flag_message

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


def _red_flag_guard(message: str, history: list[ConversationMemory]) -> GuardOutcome | None:
    """Emergency detection runs before anything else — a real emergency
    short-circuits triage, booking, and KB retrieval entirely and returns the
    urgent routing message immediately. A prompt instruction alone is not a
    guarantee, so this is a plain server-side regex gate, not something the LLM
    is asked to decide."""
    if detect_red_flag(message):
        return GuardOutcome(reply=red_flag_message(), red_flag=True)
    return None


def _severity_escalation_guard(message: str, history: list[ConversationMemory]) -> GuardOutcome | None:
    """Backstop for PATH 2's "a severe answer -> PATH 1 immediately" rule (see
    detect_severity_escalation's own docstring for the reported failure this
    covers) — the model did not reliably comply with that prompt instruction on
    its own, even after it was strengthened once already. `history` is most-
    recent-LAST (see _load_recent_history), so the assistant's own immediately-
    preceding turn — the question this message is answering, if any — is
    history[-1] whenever history is non-empty and didn't just end on a user turn
    (a user turn there means this isn't actually answering an assistant
    question at all, so the guard doesn't fire)."""
    last_assistant_message = history[-1].content if history and history[-1].role == "assistant" else None
    if detect_severity_escalation(message, last_assistant_message):
        return GuardOutcome(reply=red_flag_message(), red_flag=True)
    return None


# Pre-agent guards: each takes the raw incoming message plus the session's prior
# history and can fully resolve the turn (return a GuardOutcome) before routing/
# the specialist agent ever runs. First non-None result wins; None means "not
# this guard's concern, try the next one" — same first-match-wins shape as
# router._RULE_CASCADE.
_PRE_AGENT_GUARDS: tuple = (_red_flag_guard, _severity_escalation_guard)


def _run_pre_agent_guards(message: str, history: list[ConversationMemory]) -> GuardOutcome | None:
    for guard in _PRE_AGENT_GUARDS:
        outcome = guard(message, history)
        if outcome is not None:
            return outcome
    return None


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

    # Loaded before the current turn is saved, so the prompt's history never
    # double-includes the message being answered right now. Also loaded before
    # the pre-agent guards run (moved up from after them) — _severity_escalation_
    # guard needs to see the assistant's own immediately-preceding turn to know
    # whether this message is actually answering a severity-screening question.
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

    pre_guard_outcome = _run_pre_agent_guards(message, history)
    if pre_guard_outcome is not None:
        _save_message(db, ctx, session_id, "user", message)
        _save_message(db, ctx, session_id, "assistant", pre_guard_outcome.reply, red_flag=pre_guard_outcome.red_flag)
        db.commit()
        return ChatTurnResult(session_id=session_id, reply=pre_guard_outcome.reply, red_flag=pre_guard_outcome.red_flag)

    # The orchestrator's intent layer replaces classify_message_intent() + the old
    # single agent entirely: exactly one specialist handles this turn, decided by
    # app.services.orchestrator.router.classify_agent_intent (free heuristic first,
    # one GROQ_HELPER_MODEL call only on genuine ambiguity — see its own docstring).
    # general_info_agent covers what get_chat_reply used to (small talk,
    # personal-recall, and real hospital_info KB questions all handled by its own
    # prompt's CONVERSATIONAL EXCEPTION + retrieval, same as before).
    intent = classify_agent_intent(message, history)
    if intent == SYMPTOM_GENERAL:
        reply = run_symptom_agent(db, ctx, message, language, _trim_history(history, SYMPTOM_AGENT_HISTORY_LIMIT))
    elif intent == APPOINTMENT:
        reply = run_appointment_agent(
            db, ctx, message, language, _trim_history(history, APPOINTMENT_AGENT_HISTORY_LIMIT)
        )
    else:
        assert intent == GENERAL_INFO
        reply = run_general_info_agent(
            db, ctx, message, language, _trim_history(history, GENERAL_INFO_AGENT_HISTORY_LIMIT)
        )

    reply = _run_post_agent_guards(reply, language)

    _save_message(db, ctx, session_id, "user", message)
    _save_message(db, ctx, session_id, "assistant", reply)
    db.commit()

    return ChatTurnResult(session_id=session_id, reply=reply, red_flag=False)


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
