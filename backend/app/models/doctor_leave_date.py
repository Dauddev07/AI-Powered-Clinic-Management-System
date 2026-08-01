import uuid
from datetime import date, datetime

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class DoctorLeaveDate(Base):
    """Exceptions that override shifts (holidays, leave).

    NOTE: same cross-clinic caveat as DoctorShift — the service layer must verify
    doctor.clinic_id == clinic_id before every insert/update.

    `source` distinguishes rows the doctor CSV re-upload is allowed to replace ('csv')
    from rows an admin added directly via the block-a-date action ('admin_block') —
    CSV confirm only ever deletes-and-reinserts its own 'csv' rows for a doctor, so an
    admin's manual block always survives a re-upload.
    """

    __tablename__ = "doctor_leave_dates"
    __table_args__ = (
        UniqueConstraint(
            "clinic_id", "doctor_id", "leave_date_utc", name="uq_doctor_leave_dates_clinic_doctor_date"
        ),
        CheckConstraint("source IN ('csv', 'admin_block')", name="ck_doctor_leave_dates_source"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False
    )
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False
    )
    leave_date_utc: Mapped[date] = mapped_column(Date, nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False, server_default="csv")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
