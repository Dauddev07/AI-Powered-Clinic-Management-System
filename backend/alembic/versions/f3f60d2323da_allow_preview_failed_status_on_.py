"""allow preview_failed status on ingestion_logs

Revision ID: f3f60d2323da
Revises: 1f983a79f8ce
Create Date: 2026-07-20 16:56:55.184345

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3f60d2323da'
down_revision: Union[str, Sequence[str], None] = '1f983a79f8ce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint("ck_ingestion_logs_status", "ingestion_logs", type_="check")
    op.create_check_constraint(
        "ck_ingestion_logs_status", "ingestion_logs", "status IN ('success', 'failed', 'preview_failed')"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_ingestion_logs_status", "ingestion_logs", type_="check")
    op.create_check_constraint(
        "ck_ingestion_logs_status", "ingestion_logs", "status IN ('success', 'failed')"
    )
