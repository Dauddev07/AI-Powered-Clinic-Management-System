"""add red_flag to conversation_memory

Revision ID: c7a1e9d5f2b8
Revises: b3f6e2a0d4c1
Create Date: 2026-07-30 00:00:00.000000

Marks the deterministic same-message emergency-redirect assistant row (see
app.services.red_flag) so the frontend can re-apply its red-bordered emergency
bubble style after a history reload — previously that styling only lived on the
live in-memory response and silently reverted to a normal bubble on page refresh,
since GET /chat/history had no way to distinguish that row from an ordinary reply.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7a1e9d5f2b8"
down_revision: Union[str, Sequence[str], None] = "b3f6e2a0d4c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversation_memory",
        sa.Column("red_flag", sa.Boolean(), server_default="false", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("conversation_memory", "red_flag")
