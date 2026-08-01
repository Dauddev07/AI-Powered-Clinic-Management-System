"""slot regeneration support: leave date source, slot needs_review, expired appointment status

Revision ID: d6db5f297254
Revises: f3f60d2323da
Create Date: 2026-07-21 13:52:16.032952

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd6db5f297254'
down_revision: Union[str, Sequence[str], None] = 'f3f60d2323da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "doctor_leave_dates",
        sa.Column("source", sa.String(length=16), server_default="csv", nullable=False),
    )
    op.create_check_constraint(
        "ck_doctor_leave_dates_source", "doctor_leave_dates", "source IN ('csv', 'admin_block')"
    )

    op.add_column(
        "slots",
        sa.Column("needs_review", sa.Boolean(), server_default="false", nullable=False),
    )

    op.drop_constraint("ck_appointments_status", "appointments", type_="check")
    op.create_check_constraint(
        "ck_appointments_status",
        "appointments",
        "status IN ('confirmed', 'cancelled', 'completed', 'no_show', 'expired')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_appointments_status", "appointments", type_="check")
    op.create_check_constraint(
        "ck_appointments_status",
        "appointments",
        "status IN ('confirmed', 'cancelled', 'completed', 'no_show')",
    )

    op.drop_column("slots", "needs_review")

    op.drop_constraint("ck_doctor_leave_dates_source", "doctor_leave_dates", type_="check")
    op.drop_column("doctor_leave_dates", "source")
