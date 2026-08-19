"""add appointment department day reschedule uses table

Revision ID: b3f7e6a1c9d4
Revises: 9d7b61597e99
Create Date: 2026-08-19 00:00:00.000000

Backs the daily per-department reschedule cap (booking_engine.
DAILY_DEPARTMENT_RESCHEDULE_CAP) — an independent counter from
appointment_department_day_uses (the booking cap), mirroring its exact shape,
keyed by the OLD appointment's department/day being rescheduled away from.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3f7e6a1c9d4'
down_revision: Union[str, Sequence[str], None] = '9d7b61597e99'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('appointment_department_day_reschedule_uses',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('clinic_id', sa.UUID(), nullable=False),
    sa.Column('patient_id', sa.UUID(), nullable=False),
    sa.Column('department_id', sa.UUID(), nullable=False),
    sa.Column('appointment_id', sa.UUID(), nullable=False),
    sa.Column('local_date', sa.Date(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['appointment_id'], ['appointments.id'], ),
    sa.ForeignKeyConstraint(['clinic_id'], ['clinics.id'], ),
    sa.ForeignKeyConstraint(['department_id'], ['departments.id'], ),
    sa.ForeignKeyConstraint(['patient_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        'ix_appointment_department_day_reschedule_uses_lookup',
        'appointment_department_day_reschedule_uses',
        ['patient_id', 'department_id', 'local_date'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_appointment_department_day_reschedule_uses_lookup', table_name='appointment_department_day_reschedule_uses')
    op.drop_table('appointment_department_day_reschedule_uses')
