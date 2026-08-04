"""Orchestrates one chat turn: retrieve -> ground -> reply -> persist.

Memory is stored against the patient account from the very first message (not from
first booking, not after a threshold) — every turn, including the very first one in a
brand-new session, is written to conversation_memory before this function returns.

session_id IS a memory boundary for the full verbatim transcript: a continuing session
(one that already has at least one message) replays its OWN messages only, most-recent-
first, capped at HISTORY_MESSAGE_LIMIT — never another session's. Starting a genuinely
new session instead gets an empty transcript PLUS a short cross-session digest (see
app.services.memory_summary.refresh_patient_summary_for_new_session) — symptoms the
patient has previously described and general personal info they've volunteered, not the
old conversation itself. This keeps a "New Chat" feeling like a real fresh start while
still letting the assistant recall the kind of thing a receiving clinician would want to
know without replaying everything that was ever said.
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
from app.services.memory_summary import (
    get_patient_summary,
    refresh_patient_summary_for_new_session,
    reset_patient_summary,
)
from app.services.orchestrator.agents.appointment_agent import run_appointment_agent
from app.services.orchestrator.agents.general_info_agent import run_general_info_agent
from app.services.orchestrator.agents.symptom_agent import run_symptom_agent
from app.services.orchestrator.router import APPOINTMENT, GENERAL_INFO, SYMPTOM_GENERAL, classify_agent_intent
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

# general_info_agent: KB question-answering doesn't depend on conversation history at
# all (query_rewrite only rescues a raw retrieval miss, and direct personal-recall
# questions are now answered deterministically from patient_memory — see
# message_classifier.is_personal_recall_message — not by scanning history), so this
# only needs enough for query_rewrite's own immediate-context rescue and general
# conversational continuity.
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


def handle_chat_message(
    db: Session, ctx: ClinicContext, message: str, session_id: uuid.UUID | None
) -> ChatTurnResult:
    session_id = session_id or uuid.uuid4()
    language = detect_language(message)

    # Emergency detection runs before anything else — a real emergency short-circuits
    # triage, booking, and KB retrieval entirely and returns the urgent routing
    # message immediately. A prompt instruction alone is not a guarantee, so this is
    # a plain server-side regex gate, not something the LLM is asked to decide.
    if detect_red_flag(message):
        reply = red_flag_message(language)
        _save_message(db, ctx, session_id, "user", message)
        _save_message(db, ctx, session_id, "assistant", reply, red_flag=True)
        db.commit()
        return ChatTurnResult(session_id=session_id, reply=reply, red_flag=True)

    # Loaded before the current turn is saved, so the prompt's history never
    # double-includes the message being answered right now. A brand new session (no
    # rows yet) starts with an empty transcript plus a freshly re-summarized
    # cross-session memory digest — see module docstring and
    # app.services.memory_summary. A continuing session gets its own full (capped)
    # transcript, plus the ALREADY-computed digest (a cheap read, no re-summarization
    # — there's nothing new to fold in until a later session starts) so a fact from an
    # earlier session (e.g. the patient's name) stays answerable for the whole
    # session, not just its first turn.
    if _session_has_messages(db, ctx, session_id):
        history = _load_recent_history(db, ctx, session_id)
        patient_memory = get_patient_summary(db, ctx.clinic_id, ctx.user_id)
    else:
        history = []
        patient_memory = refresh_patient_summary_for_new_session(db, ctx.clinic_id, ctx.user_id)

    # The orchestrator's intent layer replaces classify_message_intent() + the old
    # single agent entirely: exactly one specialist handles this turn, decided by
    # app.services.orchestrator.router.classify_agent_intent (free heuristic first,
    # one GROQ_HELPER_MODEL call only on genuine ambiguity — see its own docstring).
    # general_info_agent covers what get_chat_reply used to (small talk,
    # personal-recall, and real hospital_info KB questions all handled by its own
    # prompt's CONVERSATIONAL EXCEPTION + retrieval, same as before).
    intent = classify_agent_intent(message, history)
    if intent == SYMPTOM_GENERAL:
        reply = run_symptom_agent(
            db, ctx, message, language, _trim_history(history, SYMPTOM_AGENT_HISTORY_LIMIT), patient_memory
        )
    elif intent == APPOINTMENT:
        reply = run_appointment_agent(
            db, ctx, message, language, _trim_history(history, APPOINTMENT_AGENT_HISTORY_LIMIT), patient_memory
        )
    else:
        assert intent == GENERAL_INFO
        reply = run_general_info_agent(
            db, ctx, message, language, _trim_history(history, GENERAL_INFO_AGENT_HISTORY_LIMIT), patient_memory
        )

    if not _is_marker_reply(reply):
        reply = enforce_no_diagnosis(reply, language)

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
