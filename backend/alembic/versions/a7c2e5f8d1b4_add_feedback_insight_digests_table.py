"""add feedback_insight_digests table

Revision ID: a7c2e5f8d1b4
Revises: f4b7d1e9a3c6
Create Date: 2026-08-20 00:00:00.000000

Backs the admin feedback page's recurring-complaint-theme digest (see
app/services/feedback_insights.py) — one row per clinic, holding the most recently
generated synthesis of low-rating feedback and when it was generated, so a page visit
doesn't cost an LLM call every time. Same shape as admin_insight_digests (f4b7d1e9a3c6).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a7c2e5f8d1b4"
down_revision: Union[str, Sequence[str], None] = "f4b7d1e9a3c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "feedback_insight_digests",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("clinic_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("digest_text", sa.Text(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["clinic_id"], ["clinics.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("clinic_id", name="uq_feedback_insight_digests_clinic_id"),
    )


def downgrade() -> None:
    op.drop_table("feedback_insight_digests")
