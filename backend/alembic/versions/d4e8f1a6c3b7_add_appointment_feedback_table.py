"""add appointment_feedback table

Revision ID: d4e8f1a6c3b7
Revises: c7a1e9d5f2b8
Create Date: 2026-07-30 00:00:00.000000

Post-appointment star-rating feedback. One row per completed appointment the
patient has rated — no separate "asked" flag; an appointment with no matching row
here just hasn't been rated yet (see app.services.feedback).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "d4e8f1a6c3b7"
down_revision: Union[str, Sequence[str], None] = "c7a1e9d5f2b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "appointment_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("clinic_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("clinics.id"), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            "appointment_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("appointments.id"), nullable=False
        ),
        sa.Column("doctor_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("doctors.id"), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("seq", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.CheckConstraint("rating BETWEEN 1 AND 5", name="ck_appointment_feedback_rating"),
        sa.UniqueConstraint("appointment_id", name="uq_appointment_feedback_appointment_id"),
    )
    op.create_index("ix_appointment_feedback_seq", "appointment_feedback", ["seq"], unique=True)
    op.create_index("ix_appointment_feedback_clinic_id", "appointment_feedback", ["clinic_id"])
    op.create_index("ix_appointment_feedback_patient_id", "appointment_feedback", ["patient_id"])


def downgrade() -> None:
    op.drop_index("ix_appointment_feedback_patient_id", table_name="appointment_feedback")
    op.drop_index("ix_appointment_feedback_clinic_id", table_name="appointment_feedback")
    op.drop_index("ix_appointment_feedback_seq", table_name="appointment_feedback")
    op.drop_table("appointment_feedback")
