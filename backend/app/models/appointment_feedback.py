import uuid
from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Identity, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class AppointmentFeedback(Base):
    """One row per completed appointment the patient has rated (1-5 stars), collected
    the first time they open the chatbot after the appointment completes. `reason` is
    only ever populated for a low rating (1-2 stars) — the frontend only shows the
    reason field in that case, nothing server-side forces this, so it stays nullable.
    An appointment with no row here simply hasn't been rated yet — that absence is
    what app.services.feedback.get_pending_feedback_appointments queries for, rather
    than tracking a separate "asked" flag.
    """

    __tablename__ = "appointment_feedback"
    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 5", name="ck_appointment_feedback_rating"),
        UniqueConstraint("appointment_id", name="uq_appointment_feedback_appointment_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    appointment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("appointments.id"), nullable=False
    )
    # Denormalized for admin listing convenience, same rationale as Appointment.doctor_id.
    doctor_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    # Same insertion-order tiebreaker as notifications.seq / conversation_memory.seq —
    # submit_feedback can write more than one row in a single transaction (a patient
    # rating several pending appointments in one combined chat prompt), and Postgres
    # now() ties for every statement within one transaction.
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True), nullable=False, unique=True)
