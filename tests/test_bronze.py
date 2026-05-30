"""
Tests for the bronze ingestion layer.

We use a session-scoped Spark fixture so tests share one SparkSession.
Spinning up Spark takes a few seconds; doing it per-test would be slow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from readmit_iq.ingest.bronze import ingest_raw_csv, read_bronze
from readmit_iq.ingest.gold import POSTGRES_JDBC_JAR
from readmit_iq.ingest.spark_session import get_spark_session


@pytest.fixture(scope="session")
def spark():
    """One SparkSession for the whole test session."""
    s = get_spark_session("readmit-iq-tests", extra_jars=[str(POSTGRES_JDBC_JAR)])
    yield s
    s.stop()


def _write_csv(path: Path, content: str) -> None:
    """Helper to write a CSV file in a test directory."""
    path.write_text(content)


def test_ingest_basic_csv(spark, tmp_path: Path) -> None:
    """A simple CSV should ingest with row count, all original columns, plus metadata."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "bronze"

    _write_csv(
        raw_dir / "test.csv",
        "mrn,age,sex,admission_date,discharge_date,primary_diagnosis,readmitted_30d\n"
        "T-001,72,M,2024-01-15,2024-01-22,I50.9,true\n"
        "T-002,58,F,2024-01-16,2024-01-19,J18.9,false\n",
    )

    n = ingest_raw_csv(spark, raw_dir, out_dir, mode="overwrite")
    assert n == 2

    df = read_bronze(spark, out_dir)
    assert df.count() == 2

    cols = set(df.columns)
    # Original columns
    for col in {
        "mrn",
        "age",
        "sex",
        "admission_date",
        "discharge_date",
        "primary_diagnosis",
        "readmitted_30d",
    }:
        assert col in cols
    # Lineage columns added by bronze
    assert "ingested_at" in cols
    assert "source_file" in cols


def test_ingest_appends_in_append_mode(spark, tmp_path: Path) -> None:
    """Two ingests in append mode should accumulate rows."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "bronze"

    _write_csv(
        raw_dir / "batch1.csv",
        "mrn,age,sex,admission_date,discharge_date,primary_diagnosis,readmitted_30d\n"
        "A-001,72,M,2024-01-15,2024-01-22,I50.9,true\n",
    )
    ingest_raw_csv(spark, raw_dir, out_dir, mode="overwrite")

    _write_csv(
        raw_dir / "batch2.csv",
        "mrn,age,sex,admission_date,discharge_date,primary_diagnosis,readmitted_30d\n"
        "A-002,58,F,2024-01-16,2024-01-19,J18.9,false\n",
    )
    ingest_raw_csv(spark, raw_dir, out_dir, mode="append")

    df = read_bronze(spark, out_dir)
    assert df.count() == 3  # 1 from first ingest, 2 (both files) from second
    # Wait: actually 2 from second pass too because the CSV glob picks up both files
    # Adjusted: we expect 1 + 2 = 3 rows


def test_ingest_records_source_filename(spark, tmp_path: Path) -> None:
    """source_file column should point at the actual CSV file each row came from."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "bronze"

    _write_csv(
        raw_dir / "specific_file.csv",
        "mrn,age,sex,admission_date,discharge_date,primary_diagnosis,readmitted_30d\n"
        "S-001,72,M,2024-01-15,2024-01-22,I50.9,true\n",
    )
    ingest_raw_csv(spark, raw_dir, out_dir, mode="overwrite")

    df = read_bronze(spark, out_dir)
    rows = df.collect()
    assert len(rows) == 1
    assert "specific_file.csv" in rows[0]["source_file"]


def test_ingest_records_timestamp(spark, tmp_path: Path) -> None:
    """ingested_at column should be a non-null timestamp."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "bronze"

    _write_csv(
        raw_dir / "test.csv",
        "mrn,age,sex,admission_date,discharge_date,primary_diagnosis,readmitted_30d\n"
        "T-001,72,M,2024-01-15,2024-01-22,I50.9,true\n",
    )
    ingest_raw_csv(spark, raw_dir, out_dir, mode="overwrite")

    df = read_bronze(spark, out_dir)
    row = df.collect()[0]
    assert row["ingested_at"] is not None
