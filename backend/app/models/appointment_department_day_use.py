import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class AppointmentDepartmentDayUse(Base):
    """One permanent row per (patient, department, clinic-local calendar day) booking
    attempt — written once when a slot in that department/day is first attached to a
    patient (initial booking or reschedule-into) and never deleted or decremented.
    Backs the daily per-department booking cap: cancelling or rescheduling an
    appointment away from that department/day does not remove its row here, so the
    day's count can only ever grow, never shrink.
    """

    __tablename__ = "appointment_department_day_uses"
    __table_args__ = (
        Index(
            "ix_appointment_department_day_uses_lookup",
            "patient_id",
            "department_id",
            "local_date",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    department_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("departments.id"), nullable=False
    )
    appointment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("appointments.id"), nullable=False
    )
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
