"""add appointment reminder notification types

Revision ID: 9d2c7f4b1a3e
Revises: af620748f1c6
Create Date: 2026-08-05 00:00:00.000000

Four new notification types backing the scheduled appointment-reminder job (see
app/services/appointment_reminders.py and the scheduler tick in
app/services/scheduler.py): a reminder sent exactly once at each of 60 minutes, 30
minutes, and 5 minutes before an appointment's start, plus one sent when the
appointment's own start time arrives. Postgres has no ALTER CHECK CONSTRAINT, so the
existing constraint is dropped and recreated with the widened list.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9d2c7f4b1a3e"
down_revision: Union[str, Sequence[str], None] = "af620748f1c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLD_TYPES = (
    "appointment_booked", "appointment_rescheduled", "appointment_cancelled", "appointment_auto_completed",
)
_NEW_TYPES = _OLD_TYPES + (
    "appointment_reminder_60m", "appointment_reminder_30m", "appointment_reminder_5m", "appointment_starting",
)


def upgrade() -> None:
    op.drop_constraint("ck_notifications_type", "notifications", type_="check")
    op.create_check_constraint(
        "ck_notifications_type",
        "notifications",
        "type IN (" + ", ".join(f"'{t}'" for t in _NEW_TYPES) + ")",
    )


def downgrade() -> None:
    op.drop_constraint("ck_notifications_type", "notifications", type_="check")
    op.create_check_constraint(
        "ck_notifications_type",
        "notifications",
        "type IN (" + ", ".join(f"'{t}'" for t in _OLD_TYPES) + ")",
    )
