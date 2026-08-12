"""add push_subscriptions table

Revision ID: 0fb8eb1b200f
Revises: 9d2c7f4b1a3e
Create Date: 2026-08-12 00:00:00.000000

Web Push subscriptions — one row per browser/device a patient has enabled
notifications on (see app.services.push_notifications and app.models.
push_subscription's own docstring). Unlike the notifications table, rows here are
only ever created via the patient's own explicit action, never server-side.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0fb8eb1b200f"
down_revision: Union[str, Sequence[str], None] = "9d2c7f4b1a3e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "push_subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("clinic_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("clinics.id"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column("p256dh_key", sa.String(length=255), nullable=False),
        sa.Column("auth_key", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_push_subscriptions_clinic_id", "push_subscriptions", ["clinic_id"])
    op.create_index("ix_push_subscriptions_user_id", "push_subscriptions", ["user_id"])
    op.create_unique_constraint("uq_push_subscriptions_endpoint", "push_subscriptions", ["endpoint"])


def downgrade() -> None:
    op.drop_constraint("uq_push_subscriptions_endpoint", "push_subscriptions", type_="unique")
    op.drop_index("ix_push_subscriptions_user_id", table_name="push_subscriptions")
    op.drop_index("ix_push_subscriptions_clinic_id", table_name="push_subscriptions")
    op.drop_table("push_subscriptions")
