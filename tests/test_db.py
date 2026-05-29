"""
Tests for readmit_iq.utils.db.

These tests require a running Postgres at the URL in DATABASE_URL. If the
database is unreachable, the test is skipped rather than failed — that's
the right semantics for integration tests that depend on external services.
"""
from __future__ import annotations

import pytest
import psycopg

from readmit_iq.utils.db import _redact, get_connection


def test_redact_hides_password() -> None:
    """Connection URLs with passwords should be redacted before logging."""
    url = "postgresql://user:supersecret@host:5432/db"
    redacted = _redact(url)
    assert "supersecret" not in redacted
    assert "***" in redacted
    assert "user" in redacted


def test_redact_handles_urls_without_password() -> None:
    """URLs without a password should pass through unchanged."""
    assert _redact("postgresql://localhost/db") == "postgresql://localhost/db"


def test_connection_round_trip() -> None:
    """A simple SELECT should return the expected result."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 + 1 AS result")
                row = cur.fetchone()
                assert row is not None
                assert row[0] == 2
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
