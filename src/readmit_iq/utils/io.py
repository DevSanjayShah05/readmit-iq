"""
File I/O helpers.

This module wraps the most common file operations with logging and
useful error messages, so the rest of the project doesn't repeat the
same boilerplate everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def read_csv(path: str | Path) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Args:
        path: location of the CSV. Accepts both a string and a Path.

    Returns:
        The file's contents as a DataFrame.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")

    logger.info(f"Reading CSV: {path} ({path.stat().st_size:,} bytes)")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df
