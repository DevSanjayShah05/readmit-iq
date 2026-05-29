"""
Tests for readmit_iq.utils.io
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from readmit_iq.utils.io import read_csv


def test_read_csv_returns_dataframe(tmp_path: Path) -> None:
    """A valid CSV should be loaded into a DataFrame."""
    # Arrange: write a tiny CSV to a temporary file
    csv_path = tmp_path / "demo.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n")

    # Act
    df = read_csv(csv_path)

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
    assert df.iloc[0]["a"] == 1


def test_read_csv_missing_file_raises() -> None:
    """A missing file should raise FileNotFoundError with a useful message."""
    with pytest.raises(FileNotFoundError, match="not found"):
        read_csv("nowhere/does-not-exist.csv")
