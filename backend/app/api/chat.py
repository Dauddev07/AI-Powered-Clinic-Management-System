import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_clinic_context, require_role
from app.core.db import get_db
from app.core.tenancy import ClinicContext
from app.models.conversation_memory import ConversationMemory
from app.models.user import User
from app.schemas.appointment_feedback import FeedbackSubmitIn, FeedbackSubmitOut, PendingFeedbackOut
from app.schemas.chat import ChatHistoryOut, ChatRequest, ChatResponse, ChatSessionOut
from app.services.chat import delete_session, handle_chat_message, list_sessions
from app.services.feedback import build_feedback_prompt, get_pending_feedback_appointments, submit_feedback

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def send_chat_message(
    payload: ChatRequest,
    _current_user: User = Depends(require_role("patient")),
    ctx: ClinicContext = Depends(get_clinic_context),
    db: Session = Depends(get_db),
) -> ChatResponse:
    result = handle_chat_message(db, ctx, payload.message.strip(), payload.session_id)
    return ChatResponse(session_id=result.session_id, reply=result.reply, red_flag=result.red_flag)


@router.get("/pending-feedback", response_model=PendingFeedbackOut)
def get_pending_feedback(
    _current_user: User = Depends(require_role("patient")),
    ctx: ClinicContext = Depends(get_clinic_context),
    db: Session = Depends(get_db),
) -> PendingFeedbackOut:
    appointments = get_pending_feedback_appointments(db, ctx)
    return PendingFeedbackOut(appointments=appointments, prompt=build_feedback_prompt(appointments))


@router.post("/feedback", response_model=FeedbackSubmitOut)
def post_feedback(
    payload: FeedbackSubmitIn,
    _current_user: User = Depends(require_role("patient")),
    ctx: ClinicContext = Depends(get_clinic_context),
    db: Session = Depends(get_db),
) -> FeedbackSubmitOut:
    message = submit_feedback(db, ctx, payload.appointment_ids, payload.rating, payload.reason)
    return FeedbackSubmitOut(message=message)


@router.get("/sessions", response_model=list[ChatSessionOut])
def get_chat_sessions(
    _current_user: User = Depends(require_role("patient")),
    ctx: ClinicContext = Depends(get_clinic_context),
    db: Session = Depends(get_db),
) -> list[ChatSessionOut]:
    return list_sessions(db, ctx)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat_session(
    session_id: uuid.UUID,
    _current_user: User = Depends(require_role("patient")),
    ctx: ClinicContext = Depends(get_clinic_context),
    db: Session = Depends(get_db),
) -> None:
    if not delete_session(db, ctx, session_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")


@router.get("/history", response_model=ChatHistoryOut)
def get_chat_history(
    session_id: uuid.UUID | None = Query(default=None),
    _current_user: User = Depends(require_role("patient")),
    ctx: ClinicContext = Depends(get_clinic_context),
    db: Session = Depends(get_db),
) -> ChatHistoryOut:
    resolved_session_id = session_id
    if resolved_session_id is None:
        # No session_id supplied (e.g. first load in a fresh browser session): fall
        # back to the patient's most recently active session, if any, so history
        # still restores without the frontend having to persist an id from before
        # it ever had one.
        latest = db.execute(
            select(ConversationMemory.session_id)
            .where(ConversationMemory.clinic_id == ctx.clinic_id, ConversationMemory.user_id == ctx.user_id)
            .order_by(ConversationMemory.created_at.desc(), ConversationMemory.seq.desc())
            .limit(1)
        ).scalar()
        resolved_session_id = latest

    if resolved_session_id is None:
        return ChatHistoryOut(session_id=None, messages=[])

    rows = db.execute(
        select(ConversationMemory)
        .where(
            ConversationMemory.clinic_id == ctx.clinic_id,
            ConversationMemory.user_id == ctx.user_id,
            ConversationMemory.session_id == resolved_session_id,
        )
        # created_at alone isn't a strict order: a single chat turn saves its user +
        # assistant rows in the same transaction, and Postgres now() returns the same
        # value for every statement within one transaction, so those two rows tie.
        # `seq` (a real auto-incrementing identity column) breaks the tie
        # deterministically, in true insertion order.
        .order_by(ConversationMemory.created_at.asc(), ConversationMemory.seq.asc())
    ).scalars().all()

    return ChatHistoryOut(session_id=resolved_session_id, messages=list(rows))
