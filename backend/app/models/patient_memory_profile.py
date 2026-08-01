import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class PatientMemoryProfile(Base):
    """A short, LLM-maintained digest of what's worth carrying into a BRAND NEW chat
    session for this patient — symptoms they've previously described and general
    personal info they've volunteered (name, allergies, preferences, etc.) — as
    opposed to conversation_memory, which holds the full verbatim transcript of every
    turn but is only replayed within the session it belongs to (see app.services.chat:
    a new session starts with empty conversation history, not the full cross-session
    log). One row per (clinic_id, user_id).
    """

    __tablename__ = "patient_memory_profiles"
    __table_args__ = (
        UniqueConstraint("clinic_id", "user_id", name="uq_patient_memory_profiles_clinic_id_user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    # The highest conversation_memory.seq already folded into summary_text — lets a
    # refresh only summarize messages that arrived since the last update instead of
    # re-summarizing the patient's entire history every time a new session starts.
    last_summarized_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
