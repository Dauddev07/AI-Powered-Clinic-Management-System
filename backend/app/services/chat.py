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
from app.rag.retrieval import retrieve
from app.services.chat_markers import BOOKING_MARKER, DEPARTMENT_LIST_MARKER, DOCTOR_OPTIONS_MARKER
from app.services.chat_tools import build_tools
from app.services.department_availability import list_active_department_names
from app.services.diagnosis_guard import enforce_no_diagnosis
from app.services.language import detect_language
from app.services.llm import get_chat_reply, run_chat_agent
from app.services.memory_summary import refresh_patient_summary_for_new_session
from app.services.message_classifier import (
    CONVERSATIONAL,
    PERSONAL_RECALL,
    classify_message_intent,
    is_symptom_message,
)
from app.services.query_rewrite import rewrite_query
from app.services.red_flag import detect_red_flag, red_flag_message

# How many prior turns (user+assistant messages, not pairs) of THIS SAME SESSION are
# replayed to the LLM as conversation context, most-recent-first. Bounded so a
# long-running session doesn't grow the prompt unboundedly — recent context matters
# far more than the full history for a triage chat.
HISTORY_MESSAGE_LIMIT = 30

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
    department-list cards) are composed deterministically from real DB rows in
    app.services.chat_tools, never freehanded by the LLM — running the
    diagnosis-phrasing guard's regex over their raw JSON payload would risk
    corrupting a valid card for no safety benefit, so those are passed straight
    through untouched."""
    return reply.startswith(BOOKING_MARKER) or reply.startswith(DOCTOR_OPTIONS_MARKER) or reply.startswith(
        DEPARTMENT_LIST_MARKER
    )


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
    # rows yet) starts with an empty transcript plus the patient's cross-session
    # memory digest instead — see module docstring and app.services.memory_summary.
    # A continuing session gets its own full (capped) transcript and no digest call,
    # since there's nothing new to fold in until a later session starts.
    if _session_has_messages(db, ctx, session_id):
        history = _load_recent_history(db, ctx, session_id)
        patient_memory = ""
    else:
        history = []
        patient_memory = refresh_patient_summary_for_new_session(db, ctx.clinic_id, ctx.user_id)

    # General small talk ("hi", "thanks", an off-topic statement) and personal-recall
    # questions ("what is my name?") have no KB question to ground against — the
    # latter's real answer is already sitting in `history` above (or, in a brand new
    # session, `patient_memory`), retrieve() would only ever return the fixed
    # fallback for it — so both skip retrieval and tools entirely and get a direct,
    # natural reply.
    if classify_message_intent(message, history) in (CONVERSATIONAL, PERSONAL_RECALL):
        reply = get_chat_reply(
            message=message, language=language, context_chunks=[], history=history, patient_memory=patient_memory
        )
    else:
        # A symptom/complaint message never touches KB retrieval at all — medical_kb
        # doesn't exist (there was never a reliable way to guarantee a reference
        # document covered every symptom/department combination), and hospital_info
        # is the wrong KB for it anyway. Instead the agent gets the clinic's real
        # active department names as context, straight from the structured
        # doctors/departments tables — the same source of truth
        # get_department_availability itself queries — so triage picks a department
        # that's guaranteed to actually exist.
        if is_symptom_message(message):
            department_names = list_active_department_names(db, ctx.clinic_id)
            context_chunks = (
                [f"Active departments at this clinic: {', '.join(department_names)}."]
                if department_names
                else []
            )
        else:
            # Try the raw message first. A clean, standalone question already clears
            # the threshold on its own and must never be touched by rewriting —
            # rewriting unconditionally would let leftover context from earlier in
            # the SAME session contaminate an otherwise-clear question. Rewriting
            # only kicks in as a rescue when the raw message genuinely fails on its
            # own. hospital_info retrieval is otherwise completely unaffected by the
            # symptom-routing change above.
            result = retrieve(db, ctx.clinic_id, message)
            if not result.matched:
                retrieval_query = rewrite_query(message, history)
                if retrieval_query != message:
                    result = retrieve(db, ctx.clinic_id, retrieval_query)
            context_chunks = result.chunks if result.matched else []

        # Everything that isn't pure small talk/personal-recall now goes through the
        # tool-calling agent, not a plain grounded completion: symptom triage,
        # booking/reschedule/cancel, appointment lookup, and department availability
        # all live behind this same path, alongside ordinary hospital_info Q&A. When
        # neither a symptom nor a KB match applies, the agent's own system prompt
        # still refuses to answer a genuine out-of-scope knowledge question from
        # outside knowledge — it isn't a free pass to hallucinate, it's what lets a
        # booking/triage message (which never lives in the KB) through.
        tools = build_tools(db, ctx)
        # The symptom-triage instruction block is large and irrelevant to most turns
        # (booking, availability, general clinic-info questions) — only send it when
        # this message or an earlier one in the conversation actually looked
        # symptom-related, so a mid-triage follow-up reply ("severe, started
        # yesterday") that itself contains no symptom keyword still keeps the rules
        # it needs rather than losing them partway through a triage flow.
        include_triage = is_symptom_message(message) or any(
            row.role == "user" and is_symptom_message(row.content) for row in history
        )
        reply = run_chat_agent(
            message=message,
            language=language,
            context_chunks=context_chunks,
            history=history,
            tools=tools,
            include_triage=include_triage,
            patient_memory=patient_memory,
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
    someone else), so the caller can 404 rather than silently no-op."""
    result = db.execute(
        delete(ConversationMemory).where(
            ConversationMemory.clinic_id == ctx.clinic_id,
            ConversationMemory.user_id == ctx.user_id,
            ConversationMemory.session_id == session_id,
        )
    )
    db.commit()
    return result.rowcount > 0
