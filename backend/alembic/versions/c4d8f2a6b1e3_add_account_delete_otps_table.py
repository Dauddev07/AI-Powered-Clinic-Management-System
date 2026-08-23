"""add account_delete_otps table

Revision ID: c4d8f2a6b1e3
Revises: b2d6f9a1c3e7
Create Date: 2026-08-23 00:00:00.000000

Backs the patient self-service account-deletion flow (see
app/services/account_deletion.py) — a 6-digit code emailed to the patient before the
account and all its data are permanently deleted, stored here only as a hash with an
expiry, a used flag, and a failed-attempt counter, same shape as password_reset_otps.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c4d8f2a6b1e3"
down_revision: Union[str, Sequence[str], None] = "b2d6f9a1c3e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "account_delete_otps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("otp_hash", sa.String(length=255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_account_delete_otps_user_id", "account_delete_otps", ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_account_delete_otps_user_id", table_name="account_delete_otps")
    op.drop_table("account_delete_otps")
