import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class Appointment(Base):
    """Status values: 'confirmed' is the one active/"booked" state — it's what the
    partial unique index keys off and what patient-overlap/cancellation rules check
    against. 'completed' is set automatically (app.services.appointments.
    auto_complete_past_appointments) the moment a 'confirmed' appointment's slot end_utc
    passes — no manual admin marking step. 'cancelled' is the patient-initiated final
    state. 'no_show'/'expired' remain valid per the CHECK constraint for any historical
    rows but nothing in the system sets them anymore.
    """

    __tablename__ = "appointments"
    __table_args__ = (
        CheckConstraint(
            "status IN ('confirmed', 'cancelled', 'completed', 'no_show', 'expired')",
            name="ck_appointments_status",
        ),
        CheckConstraint("booked_via IN ('chatbot', 'manual')", name="ck_appointments_booked_via"),
        # Guarantees at most one *active* (confirmed) appointment per slot, while still
        # allowing historical cancelled/completed rows for the same slot to coexist.
        Index(
            "uq_appointments_slot_id_active",
            "slot_id",
            unique=True,
            postgresql_where=text("status = 'confirmed'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False
    )
    slot_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("slots.id"), nullable=False)
    patient_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    # Denormalized for query convenience.
    doctor_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="confirmed")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    booked_via: Mapped[str] = mapped_column(String(16), nullable=False, server_default="chatbot")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
