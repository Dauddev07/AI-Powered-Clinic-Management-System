import uuid
from datetime import datetime, time

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, SmallInteger, Time, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class DoctorShift(Base):
    """Recurring weekly availability template.

    NOTE: clinic_id and doctor_id are both present, but Postgres cannot express
    "doctor.clinic_id == clinic_id" as a plain FK/CHECK (no cross-table CHECK
    constraints). The service layer MUST verify this before every insert/update —
    see docs/architecture-data-dictionary.md §3.5.
    """

    __tablename__ = "doctor_shifts"
    __table_args__ = (
        UniqueConstraint("clinic_id", "doctor_id", "weekday", name="uq_doctor_shifts_clinic_doctor_weekday"),
        CheckConstraint("end_time_utc > start_time_utc", name="ck_doctor_shifts_end_after_start"),
        CheckConstraint("weekday >= 0 AND weekday <= 6", name="ck_doctor_shifts_weekday_range"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False
    )
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False
    )
    weekday: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    start_time_utc: Mapped[time] = mapped_column(Time, nullable=False)
    end_time_utc: Mapped[time] = mapped_column(Time, nullable=False)
    slot_minutes: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
