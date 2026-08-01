"""add seq tiebreaker to conversation_memory

Revision ID: 1cf074a2e9e6
Revises: a02cdace2194
Create Date: 2026-07-28 00:00:00.000000

conversation_memory.created_at uses server_default now() — inside a single Postgres
transaction, now() returns the SAME value for every statement in that transaction, and
handle_chat_message() saves the user and assistant rows of one turn in the same
transaction. So the two rows of a single turn (and often two adjacent turns saved in
quick succession) can tie exactly on created_at, and ORDER BY created_at alone then
falls back to undefined physical row order — which is what produced the reported
out-of-order history after a page refresh. `id` can't serve as a tiebreaker since it's
a random UUID, not sortable by insertion order. This adds a real auto-incrementing
column for that purpose: unlike now(), a GENERATED ALWAYS AS IDENTITY column advances
per row inserted, even within the same transaction, so it reflects true insertion
order regardless of timestamp collisions.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1cf074a2e9e6"
down_revision: Union[str, Sequence[str], None] = "a02cdace2194"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversation_memory",
        sa.Column("seq", sa.BigInteger(), sa.Identity(always=True), nullable=False),
    )
    op.create_index(
        "ix_conversation_memory_seq", "conversation_memory", ["seq"], unique=True
    )


def downgrade() -> None:
    op.drop_index("ix_conversation_memory_seq", table_name="conversation_memory")
    op.drop_column("conversation_memory", "seq")
