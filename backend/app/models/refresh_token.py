import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class RefreshToken(Base):
    """One row per issued refresh token (see app/services/refresh_tokens.py). Only the
    sha256 hash of the token is stored — same principle as the password/OTP tables —
    so a database leak alone can't be replayed as a live session. Unlike the OTP
    tables' bcrypt hash, this uses a plain, fast hash: the token itself is a
    32-byte-random, high-entropy secret (not a low-entropy 6-digit code or a
    human password), so it isn't brute-forceable regardless of hash speed, and a
    fast hash is what makes an indexed exact-match lookup by hash practical here.

    Rotated, not just re-verified, on every /auth/refresh call: revoked_at is set on
    the old row and a brand new row is inserted, so a stolen-and-replayed refresh
    token is immediately detectable (its row is already revoked) rather than staying
    silently valid for its full lifetime.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
