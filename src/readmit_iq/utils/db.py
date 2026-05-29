"""
Database connection helpers.

A single place to get a Postgres connection, so we don't repeat
connection-string parsing or context-manager boilerplate everywhere.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from loguru import logger

from readmit_iq.config import get_settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """
    Yield a Postgres connection that auto-closes on exit.

    Used like:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                print(cur.fetchone())
    """
    settings = get_settings()
    logger.debug(f"Connecting to Postgres at {_redact(settings.database_url)}")
    conn = psycopg.connect(settings.database_url)
    try:
        yield conn
    finally:
        conn.close()
        logger.debug("Postgres connection closed")


def _redact(url: str) -> str:
    """Hide the password in a connection URL before logging it."""
    # Postgres URLs look like: postgresql://user:password@host:port/db
    # We replace the password segment with '***'.
    if "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host_part = rest.split("@", 1)
    if ":" in creds:
        user, _ = creds.split(":", 1)
        return f"{scheme}://{user}:***@{host_part}"
    return url
