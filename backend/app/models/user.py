import uuid
from datetime import date, datetime

from sqlalchemy import Boolean, CheckConstraint, Date, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class User(Base):
    """Admins and patients. No doctor login — doctors have no User row."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("clinic_id", "email", name="uq_users_clinic_id_email"),
        CheckConstraint("role IN ('admin', 'patient')", name="ck_users_role"),
        CheckConstraint("gender IS NULL OR gender IN ('male', 'female', 'other')", name="ck_users_gender"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Set true by the Superadmin CLI when creating an admin with a temporary password,
    # forcing a change on first login (Day 3, Day 9).
    must_change_password: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    # NULL until the first password change. Any JWT whose `iat` predates this timestamp
    # is rejected in get_current_user, so changing a password immediately invalidates
    # every token issued before the change (see app/api/deps.py).
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
