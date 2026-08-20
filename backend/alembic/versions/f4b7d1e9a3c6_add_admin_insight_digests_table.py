"""add admin_insight_digests table

Revision ID: f4b7d1e9a3c6
Revises: e1a9c4f6b2d7
Create Date: 2026-08-20 00:00:00.000000

Backs the admin dashboard's weekly plain-English digest (see
app/services/admin_insights.py) — one row per clinic, holding the most recently
generated summary text and when it was generated, so a dashboard visit doesn't cost an
LLM call every time. Same shape as patient_memory_profiles (af620748f1c6).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f4b7d1e9a3c6"
down_revision: Union[str, Sequence[str], None] = "e1a9c4f6b2d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "admin_insight_digests",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("clinic_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("digest_text", sa.Text(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["clinic_id"], ["clinics.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("clinic_id", name="uq_admin_insight_digests_clinic_id"),
    )


def downgrade() -> None:
    op.drop_table("admin_insight_digests")
