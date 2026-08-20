import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class FeedbackInsightDigest(Base):
    """A short, LLM-generated synthesis of this clinic's recurring low-rating (1-2
    star) feedback themes — see app.services.feedback_insights. Same cache-and-refresh
    shape as app.models.admin_insight_digest: one row per clinic_id, refreshed at most
    once per app.services.feedback_insights.REFRESH_INTERVAL rather than regenerated
    on every admin feedback-page visit.
    """

    __tablename__ = "feedback_insight_digests"
    __table_args__ = (UniqueConstraint("clinic_id", name="uq_feedback_insight_digests_clinic_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False)
    digest_text: Mapped[str] = mapped_column(Text, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
