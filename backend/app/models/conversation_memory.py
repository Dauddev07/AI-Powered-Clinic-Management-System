import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, ForeignKey, Identity, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class ConversationMemory(Base):
    """Chatbot session/turn history for context continuity.

    user_id is NOT NULL: login is required before using the chatbot (confirmed in the
    SOW) — there is no anonymous/pre-login triage, so the user is always known.
    """

    __tablename__ = "conversation_memory"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="ck_conversation_memory_role"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False
    )
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # True only for the deterministic same-message emergency redirect (see
    # app.services.red_flag) — lets the frontend re-apply its distinct red-bordered
    # emergency bubble style after a page refresh/history reload, not just on the
    # live response. Without this, GET /chat/history had no way to tell that
    # assistant row apart from an ordinary reply, so the emergency styling silently
    # reverted to a normal chat bubble the moment the page reloaded.
    red_flag: Mapped[bool] = mapped_column(Boolean, server_default="false", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    # Deterministic insertion-order tiebreaker for created_at ties: `now()` returns
    # the same value for every statement within one Postgres transaction, and a
    # single chat turn saves its user + assistant rows in the same transaction, so
    # created_at alone cannot order them. `id` can't fill in either — it's a random
    # UUID, not sortable by insertion order. This is a real GENERATED ALWAYS AS
    # IDENTITY column, which advances per row even within one transaction.
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True), nullable=False, unique=True)
