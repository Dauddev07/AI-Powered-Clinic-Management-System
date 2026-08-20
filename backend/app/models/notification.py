import uuid
from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Identity, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class Notification(Base):
    """In-app notifications for a patient's own booking activity: appointment
    booked/rescheduled/cancelled/auto-completed. Always created server-side via
    app.services.notifications.create_notification — never client-writable.
    """

    __tablename__ = "notifications"
    __table_args__ = (
        CheckConstraint(
            "type IN ('appointment_booked', 'appointment_rescheduled', 'appointment_cancelled', "
            "'appointment_auto_completed', 'appointment_reminder_60m')",
            name="ck_notifications_type",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    related_appointment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("appointments.id"), nullable=True
    )
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    # Same insertion-order tiebreaker as conversation_memory.seq — a booking action
    # can generate more than one notification-worthy event in the same transaction
    # (e.g. a scheduler tick auto-completing several appointments at once), and
    # now() ties within one transaction while `id` is a random UUID that isn't
    # sortable by insertion order.
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True), nullable=False, unique=True)
