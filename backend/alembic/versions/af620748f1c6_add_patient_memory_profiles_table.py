"""add patient_memory_profiles table

Revision ID: af620748f1c6
Revises: d4e8f1a6c3b7
Create Date: 2026-08-01 00:00:00.000000

Backs the per-patient cross-session memory digest (see
app.models.patient_memory_profile / app.services.memory_summary): a short,
LLM-maintained summary of symptoms previously described and general personal info
volunteered, carried into a brand new chat session in place of the full transcript.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "af620748f1c6"
down_revision: Union[str, Sequence[str], None] = "d4e8f1a6c3b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "patient_memory_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("clinic_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("clinics.id"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("last_summarized_seq", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.UniqueConstraint("clinic_id", "user_id", name="uq_patient_memory_profiles_clinic_id_user_id"),
    )
    op.create_index("ix_patient_memory_profiles_user_id", "patient_memory_profiles", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_patient_memory_profiles_user_id", table_name="patient_memory_profiles")
    op.drop_table("patient_memory_profiles")
