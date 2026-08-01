"""add notifications table

Revision ID: b3f6e2a0d4c1
Revises: 1cf074a2e9e6
Create Date: 2026-07-29 00:00:00.000000

New in-app notification system for patients: appointment booked/rescheduled/
cancelled/auto-completed. Includes the same seq Identity-column insertion-order
tiebreaker as conversation_memory (see 1cf074a2e9e6) for the same reason — a single
scheduler tick can auto-complete several appointments in one transaction, and
created_at's server-side now() ties for every statement within it.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b3f6e2a0d4c1"
down_revision: Union[str, Sequence[str], None] = "1cf074a2e9e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "notifications",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("clinic_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("clinics.id"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column(
            "related_appointment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("appointments.id"),
            nullable=True,
        ),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("seq", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.CheckConstraint(
            "type IN ('appointment_booked', 'appointment_rescheduled', 'appointment_cancelled', "
            "'appointment_auto_completed')",
            name="ck_notifications_type",
        ),
    )
    op.create_index("ix_notifications_seq", "notifications", ["seq"], unique=True)
    op.create_index("ix_notifications_clinic_id", "notifications", ["clinic_id"])
    op.create_index("ix_notifications_user_id", "notifications", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_notifications_user_id", table_name="notifications")
    op.drop_index("ix_notifications_clinic_id", table_name="notifications")
    op.drop_index("ix_notifications_seq", table_name="notifications")
    op.drop_table("notifications")
