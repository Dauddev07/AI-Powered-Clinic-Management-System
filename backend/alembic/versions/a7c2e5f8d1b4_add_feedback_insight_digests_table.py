"""add feedback_insight_digests table

Revision ID: a7c2e5f8d1b4
Revises: f4b7d1e9a3c6
Create Date: 2026-08-20 00:00:00.000000

Backs the admin feedback page's recurring-complaint-theme digest — one row per
clinic, holding the most recently generated synthesis of low-rating feedback and when
it was generated. Same shape as admin_insight_digests (f4b7d1e9a3c6).

The feature this backed was later removed (see b2d6f9a1c3e7, which drops this table
again) — this file is kept in place rather than deleted because production already
ran it: alembic tracks the current revision by id in `alembic_version`, and deleting a
migration file that a real database has already applied leaves that id unresolvable,
breaking `alembic upgrade head` there. Always add a new migration to undo a change
that shipped, never delete the one that made it.
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
