import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class AdminInsightDigest(Base):
    """A short, LLM-generated plain-English summary of this clinic's own admin
    dashboard numbers (see app.api.admin_dashboard for the underlying queries) —
    refreshed at most once per app.services.admin_insights.REFRESH_INTERVAL rather
    than regenerated on every dashboard visit. One row per clinic_id, same
    cache-and-refresh shape as app.models.patient_memory_profile.
    """

    __tablename__ = "admin_insight_digests"
    __table_args__ = (UniqueConstraint("clinic_id", name="uq_admin_insight_digests_clinic_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False)
    digest_text: Mapped[str] = mapped_column(Text, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
