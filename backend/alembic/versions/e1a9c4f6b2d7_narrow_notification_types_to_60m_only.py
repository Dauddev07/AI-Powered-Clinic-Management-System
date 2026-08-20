"""narrow notification types to 60m reminder only

Revision ID: e1a9c4f6b2d7
Revises: b3f7e6a1c9d4
Create Date: 2026-08-20 00:00:00.000000

9d2c7f4b1a3e widened ck_notifications_type to four reminder-cadence types
(appointment_reminder_60m/30m/5m, appointment_starting) anticipating a multi-stage
reminder job. Only the 60-minute tier was ever implemented (see
app/services/appointment_reminders.py) — the other three were never sent by any code
path. Product decision: keep the single 60-minute reminder rather than build out the
rest, so the unused types are dropped from the constraint. Postgres has no ALTER CHECK
CONSTRAINT, so the constraint is dropped and recreated with the narrower list, same
technique as 9d2c7f4b1a3e.

A handful of leftover rows (5 total, seen in dev) use the 30m/5m/starting types from
whatever manual testing happened while that wider reminder job was being tried out —
no code path has ever sent these in production, so they're deleted here rather than
kept around as permanently-orphaned rows the narrowed constraint would otherwise
reject outright.
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "e1a9c4f6b2d7"
down_revision: Union[str, Sequence[str], None] = "b3f7e6a1c9d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_WIDE_TYPES = (
    "appointment_booked", "appointment_rescheduled", "appointment_cancelled", "appointment_auto_completed",
    "appointment_reminder_60m", "appointment_reminder_30m", "appointment_reminder_5m", "appointment_starting",
)
_NARROW_TYPES = (
    "appointment_booked", "appointment_rescheduled", "appointment_cancelled", "appointment_auto_completed",
    "appointment_reminder_60m",
)


def upgrade() -> None:
    op.execute(
        text(
            "DELETE FROM notifications WHERE type IN "
            "('appointment_reminder_30m', 'appointment_reminder_5m', 'appointment_starting')"
        )
    )
    op.drop_constraint("ck_notifications_type", "notifications", type_="check")
    op.create_check_constraint(
        "ck_notifications_type",
        "notifications",
        "type IN (" + ", ".join(f"'{t}'" for t in _NARROW_TYPES) + ")",
    )


def downgrade() -> None:
    op.drop_constraint("ck_notifications_type", "notifications", type_="check")
    op.create_check_constraint(
        "ck_notifications_type",
        "notifications",
        "type IN (" + ", ".join(f"'{t}'" for t in _WIDE_TYPES) + ")",
    )
