import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class AppointmentDepartmentDayRescheduleUse(Base):
    """One permanent row per (patient, department, clinic-local calendar day) reschedule
    of an appointment that ORIGINALLY sat in that department/day — written once per
    successful reschedule_appointment call, keyed by the OLD slot's department/day (the
    appointment being moved away FROM), never deleted or decremented. Backs the daily
    per-department reschedule cap: mirrors AppointmentDepartmentDayUse's own shape
    exactly, kept as a separate table since a reschedule and a fresh booking are
    different actions with their own independent daily allowance.
    """

    __tablename__ = "appointment_department_day_reschedule_uses"
    __table_args__ = (
        Index(
            "ix_appointment_department_day_reschedule_uses_lookup",
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
