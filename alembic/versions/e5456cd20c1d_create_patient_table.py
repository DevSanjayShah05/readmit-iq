"""create patient table

Revision ID: e5456cd20c1d
Revises: 
Create Date: 2026-05-29 15:22:28.705418

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5456cd20c1d'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE patient (
            id SERIAL PRIMARY KEY,
            mrn VARCHAR(64) UNIQUE NOT NULL,
            age INTEGER NOT NULL,
            sex VARCHAR(1) NOT NULL,
            admission_date DATE NOT NULL,
            discharge_date DATE NOT NULL,
            primary_diagnosis VARCHAR(10),
            readmitted_30d BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    op.execute("CREATE INDEX idx_patient_admission_date ON patient (admission_date);")
    op.execute("CREATE INDEX idx_patient_readmitted ON patient (readmitted_30d);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS patient;")
