"""add password_reset_otps table

Revision ID: b3f7a1c9e2d4
Revises: 9d2c7f4b1a3e
Create Date: 2026-08-13 00:00:00.000000

Backs the forgot-password flow (see app/services/password_reset.py) — a 6-digit code
emailed to the patient, stored here only as a hash with an expiry, a used flag, and a
failed-attempt counter so a leaked/guessed row can't be replayed or brute-forced.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b3f7a1c9e2d4"
down_revision: Union[str, Sequence[str], None] = "9d2c7f4b1a3e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "password_reset_otps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("otp_hash", sa.String(length=255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_password_reset_otps_user_id", "password_reset_otps", ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_password_reset_otps_user_id", table_name="password_reset_otps")
    op.drop_table("password_reset_otps")
