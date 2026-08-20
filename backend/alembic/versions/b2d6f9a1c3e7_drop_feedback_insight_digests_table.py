"""drop feedback_insight_digests table

Revision ID: b2d6f9a1c3e7
Revises: a7c2e5f8d1b4
Create Date: 2026-08-20 00:00:00.000001

Reverses a7c2e5f8d1b4 — the recurring-complaint-theme digest feature it backed was
removed shortly after shipping (product decision). A forward migration rather than
running `alembic downgrade`, since the deploy pipeline only ever calls `upgrade head`.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2d6f9a1c3e7"
down_revision: Union[str, Sequence[str], None] = "a7c2e5f8d1b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("feedback_insight_digests")


def downgrade() -> None:
    import sqlalchemy as sa
    from sqlalchemy.dialects import postgresql

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
