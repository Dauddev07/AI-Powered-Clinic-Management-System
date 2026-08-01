import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class IngestionLog(Base):
    """Audit trail for ingestion runs (RAG document ingestion and doctor CSV ingestion)."""

    __tablename__ = "ingestion_logs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('success', 'failed', 'preview_failed')", name="ck_ingestion_logs_status"
        ),
        CheckConstraint(
            "source_type IN ('kb_document', 'doctor_csv')", name="ck_ingestion_logs_source_type"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False
    )
    kb_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("kb_documents.id"), nullable=True
    )
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, server_default="kb_document")
    uploaded_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    rows_accepted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rows_rejected: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
