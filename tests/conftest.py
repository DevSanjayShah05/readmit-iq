"""
Shared pytest fixtures.

The `clean_patient_table` fixture truncates the patient table before each
test that uses it, so tests start from a known empty state. Truncate is
fast and resets the SERIAL id counter — handy for predictable test ids.
"""
from __future__ import annotations

import pytest
import psycopg

from readmit_iq.utils.db import get_connection


@pytest.fixture
def clean_patient_table():
    """Truncate the patient table before each test."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE patient RESTART IDENTITY;")
                conn.commit()
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    yield  # test runs here
    # No cleanup needed after — the next test's setup will truncate again.
