import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class Slot(Base):
    """Materialized bookable time slots, generated server-side from shifts minus leave dates.

    NOTE: same cross-clinic caveat as DoctorShift — the service layer must verify
    doctor.clinic_id == clinic_id before every insert/update.
    """

    __tablename__ = "slots"
    __table_args__ = (
        UniqueConstraint("clinic_id", "doctor_id", "start_utc", name="uq_slots_clinic_doctor_start"),
        CheckConstraint("end_utc > start_utc", name="ck_slots_end_after_start"),
        CheckConstraint("status IN ('open', 'booked', 'blocked')", name="ck_slots_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False
    )
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False
    )
    start_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="open")
    # Set by slot regeneration's booking-preservation diff: a shift shrank (or a leave/
    # block date was added) underneath a slot that still has a confirmed appointment.
    # The slot and its appointment are left completely untouched — this is only a flag
    # for admin review, never an automatic cancellation.
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
