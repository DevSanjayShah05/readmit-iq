"""
Plain data types representing rows in our database tables.

These are the canonical Python shapes our application reasons about. The
DAO layer converts between SQL rows and these objects in both directions.

We use frozen dataclasses (immutable) so that once a Patient is constructed,
its fields cannot accidentally be mutated by code elsewhere. If you need a
modified copy, use dataclasses.replace() — it returns a new object rather
than mutating the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class Patient:
    """One row of the `patient` table."""

    mrn: str
    age: int
    sex: str
    admission_date: date
    discharge_date: date
    primary_diagnosis: str | None
    readmitted_30d: bool
    # These two are server-assigned, so they may be None on a new object
    # that hasn't been inserted yet. Once inserted, the DAO fills them in.
    id: int | None = None
    created_at: datetime | None = None
