"""add email verification

Revision ID: c8e2f5a9d1b6
Revises: b3f7a1c9e2d4
Create Date: 2026-08-13 00:00:00.000000

Adds users.email_verified (default true — grandfathers in every existing account,
see User.email_verified's own docstring) and the email_verification_otps table
backing the new self-registration OTP flow (see app/services/email_verification.py).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c8e2f5a9d1b6"
down_revision: Union[str, Sequence[str], None] = "b3f7a1c9e2d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("email_verified", sa.Boolean(), nullable=False, server_default="true"),
    )

    op.create_table(
        "email_verification_otps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("otp_hash", sa.String(length=255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_email_verification_otps_user_id", "email_verification_otps", ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_email_verification_otps_user_id", table_name="email_verification_otps")
    op.drop_table("email_verification_otps")
    op.drop_column("users", "email_verified")
