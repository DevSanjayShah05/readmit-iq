"""
Tests for the silver layer.

We construct small synthetic DataFrames covering deduplication, validation,
and filtering cases, then assert the silver transformations produce the
right output.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from readmit_iq.ingest.silver import (
    read_silver,
    run_silver,
    transform_bronze_to_silver,
)
from readmit_iq.ingest.spark_session import get_spark_session


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """One SparkSession for the whole test session."""
    s = get_spark_session("readmit-iq-silver-tests")
    yield s
    s.stop()


def _make_bronze_df(spark: SparkSession, rows: list[dict]):
    """Helper: build a bronze-shaped DataFrame from a list of dicts."""
    return spark.createDataFrame(rows)


# ---------- Deduplication tests ----------


def test_dedup_keeps_latest_ingested_at(spark: SparkSession) -> None:
    """When the same MRN appears twice, the row with the latest ingested_at wins."""
    rows = [
        {
            "mrn": "DUP-001",
            "age": 60,
            "sex": "M",
            "admission_date": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 20),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "old.csv",
        },
        {
            "mrn": "DUP-001",
            "age": 60,
            "sex": "M",
            "admission_date": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 20),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": True,
            "ingested_at": datetime(2024, 6, 2, 10, 0, 0),
            "source_file": "corrected.csv",
        },
    ]
    bronze = _make_bronze_df(spark, rows)
    silver = transform_bronze_to_silver(bronze).collect()

    assert len(silver) == 1
    assert silver[0]["readmitted_30d"] is True
    assert silver[0]["source_file"] == "corrected.csv"


def test_dedup_preserves_distinct_mrns(spark: SparkSession) -> None:
    """Different MRNs shouldn't be deduplicated together."""
    rows = [
        {
            "mrn": "PATIENT-A",
            "age": 60,
            "sex": "M",
            "admission_date": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 20),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "a.csv",
        },
        {
            "mrn": "PATIENT-B",
            "age": 45,
            "sex": "F",
            "admission_date": date(2024, 2, 1),
            "discharge_date": date(2024, 2, 3),
            "primary_diagnosis": "J18.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "a.csv",
        },
    ]
    bronze = _make_bronze_df(spark, rows)
    silver = transform_bronze_to_silver(bronze).collect()
    assert len(silver) == 2


# ---------- Age validation tests ----------


def test_age_in_range_is_kept(spark: SparkSession) -> None:
    """Valid ages should pass through unchanged."""
    rows = [
        {
            "mrn": "AGE-001",
            "age": 65,
            "sex": "M",
            "admission_date": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 20),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "a.csv",
        },
    ]
    silver = transform_bronze_to_silver(_make_bronze_df(spark, rows)).collect()
    assert silver[0]["age"] == 65


def test_age_out_of_range_becomes_null(spark: SparkSession) -> None:
    """Negative or absurdly high ages should be nulled, not dropped."""
    rows = [
        {
            "mrn": "BAD-AGE-001",
            "age": -5,
            "sex": "M",
            "admission_date": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 20),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "a.csv",
        },
        {
            "mrn": "BAD-AGE-002",
            "age": 500,
            "sex": "F",
            "admission_date": date(2024, 2, 1),
            "discharge_date": date(2024, 2, 3),
            "primary_diagnosis": "J18.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "a.csv",
        },
    ]
    silver = transform_bronze_to_silver(_make_bronze_df(spark, rows)).collect()
    assert len(silver) == 2
    for row in silver:
        assert row["age"] is None


# ---------- Sex normalization tests ----------


def test_sex_lowercase_is_uppercased(spark: SparkSession) -> None:
    """Sex 'm' should become 'M'."""
    rows = [
        {
            "mrn": "SEX-001",
            "age": 60,
            "sex": "m",
            "admission_date": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 20),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "a.csv",
        },
    ]
    silver = transform_bronze_to_silver(_make_bronze_df(spark, rows)).collect()
    assert silver[0]["sex"] == "M"


def test_sex_invalid_value_becomes_null(spark: SparkSession) -> None:
    """Unrecognized sex values should be nulled."""
    rows = [
        {
            "mrn": "SEX-002",
            "age": 60,
            "sex": "unknown",
            "admission_date": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 20),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "a.csv",
        },
    ]
    silver = transform_bronze_to_silver(_make_bronze_df(spark, rows)).collect()
    assert silver[0]["sex"] is None


# ---------- MRN filtering tests ----------


def test_null_mrn_row_is_dropped(spark: SparkSession) -> None:
    """Rows without an MRN are fundamentally unusable; drop them."""
    rows = [
        {
            "mrn": None,
            "age": 60,
            "sex": "M",
            "admission_date": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 20),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "a.csv",
        },
        {
            "mrn": "VALID-001",
            "age": 45,
            "sex": "F",
            "admission_date": date(2024, 2, 1),
            "discharge_date": date(2024, 2, 3),
            "primary_diagnosis": "J18.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "a.csv",
        },
    ]
    silver = transform_bronze_to_silver(_make_bronze_df(spark, rows)).collect()
    assert len(silver) == 1
    assert silver[0]["mrn"] == "VALID-001"


# ---------- End-to-end test ----------


def test_run_silver_end_to_end(spark: SparkSession, tmp_path: Path) -> None:
    """Full silver pipeline: bronze on disk -> silver on disk."""
    rows = [
        {
            "mrn": "E2E-001",
            "age": 70,
            "sex": "M",
            "admission_date": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 20),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": True,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "test.csv",
        },
        {
            "mrn": "E2E-002",
            "age": 200,
            "sex": "f",
            "admission_date": date(2024, 2, 1),
            "discharge_date": date(2024, 2, 3),
            "primary_diagnosis": "J18.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1, 10, 0, 0),
            "source_file": "test.csv",
        },
    ]
    bronze_path = tmp_path / "bronze"
    silver_path = tmp_path / "silver"
    _make_bronze_df(spark, rows).write.format("delta").save(str(bronze_path))

    n = run_silver(spark, bronze_path, silver_path, mode="overwrite")
    assert n == 2

    silver = read_silver(spark, silver_path).collect()
    e2e_2 = next(r for r in silver if r["mrn"] == "E2E-002")
    assert e2e_2["age"] is None
    assert e2e_2["sex"] == "F"
