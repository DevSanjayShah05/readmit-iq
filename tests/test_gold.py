"""
Tests for the gold layer.

Gold writes to a real Postgres table, so the integration tests require
Docker Compose Postgres to be running. We use the clean_patient_table
fixture from conftest.py to ensure each test starts with an empty table.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from readmit_iq.dao.patient_dao import PatientDAO
from readmit_iq.ingest.gold import (
    POSTGRES_JDBC_JAR,
    _jdbc_url_for_spark,
    _parse_credentials,
    _project_to_patient_schema,
    run_gold,
)
from readmit_iq.ingest.spark_session import get_spark_session


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """One SparkSession (with the Postgres JDBC driver) for the gold tests."""
    s = get_spark_session(
        app_name="readmit-iq-gold-tests",
        extra_jars=[str(POSTGRES_JDBC_JAR)],
    )
    yield s
    s.stop()


# ---------- Pure-function unit tests (no Postgres or Spark IO) ----------


def test_jdbc_url_strips_credentials() -> None:
    url = "postgresql://user:pw@localhost:5432/mydb"
    assert _jdbc_url_for_spark(url) == "jdbc:postgresql://localhost:5432/mydb"


def test_jdbc_url_handles_psycopg_dialect() -> None:
    url = "postgresql+psycopg://u:p@h:5432/d"
    assert _jdbc_url_for_spark(url) == "jdbc:postgresql://h:5432/d"


def test_jdbc_url_rejects_other_dialects() -> None:
    with pytest.raises(ValueError):
        _jdbc_url_for_spark("mysql://u:p@h:3306/d")


def test_parse_credentials_returns_user_and_password() -> None:
    user, pw = _parse_credentials("postgresql://alice:secret@host:5432/db")
    assert user == "alice"
    assert pw == "secret"


def test_parse_credentials_raises_when_missing() -> None:
    with pytest.raises(ValueError):
        _parse_credentials("postgresql://host:5432/db")


# ---------- Schema projection test (Spark, no Postgres) ----------


def test_project_to_patient_drops_lineage_and_orders_columns(
    spark: SparkSession,
) -> None:
    """Gold projection must drop lineage columns AND match patient column order."""
    rows = [
        {
            "mrn": "P-001",
            "age": 60,
            "sex": "M",
            "admission_date": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 20),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1),
            "source_file": "a.csv",
        },
    ]
    df = spark.createDataFrame(rows)
    projected = _project_to_patient_schema(df)
    assert "ingested_at" not in projected.columns
    assert "source_file" not in projected.columns
    # Column order matches the Postgres patient table (per the migration)
    assert projected.columns == [
        "mrn",
        "age",
        "sex",
        "admission_date",
        "discharge_date",
        "primary_diagnosis",
        "readmitted_30d",
    ]


# ---------- End-to-end integration tests (Spark + Postgres) ----------


def test_run_gold_loads_silver_into_postgres(
    spark: SparkSession, tmp_path: Path, clean_patient_table
) -> None:
    """Full gold pipeline: silver Delta -> Postgres patient table."""
    rows = [
        {
            "mrn": "GOLD-001",
            "age": 72,
            "sex": "M",
            "admission_date": date(2024, 1, 15),
            "discharge_date": date(2024, 1, 22),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": True,
            "ingested_at": datetime(2024, 6, 1),
            "source_file": "test.csv",
        },
        {
            "mrn": "GOLD-002",
            "age": 58,
            "sex": "F",
            "admission_date": date(2024, 1, 16),
            "discharge_date": date(2024, 1, 19),
            "primary_diagnosis": "J18.9",
            "readmitted_30d": False,
            "ingested_at": datetime(2024, 6, 1),
            "source_file": "test.csv",
        },
    ]
    silver_path = tmp_path / "silver"
    spark.createDataFrame(rows).write.format("delta").save(str(silver_path))

    n = run_gold(spark, silver_path, mode="overwrite")
    assert n == 2

    dao = PatientDAO()
    assert dao.count() == 2

    patient = dao.find_by_mrn("GOLD-001")
    assert patient is not None
    assert patient.age == 72
    assert patient.primary_diagnosis == "I50.9"
    assert patient.readmitted_30d is True


def test_run_gold_overwrite_replaces_prior_data(
    spark: SparkSession, tmp_path: Path, clean_patient_table
) -> None:
    """After overwrite, only the new silver rows should be in the table."""
    from readmit_iq.models import Patient

    dao = PatientDAO()
    dao.insert(
        Patient(
            mrn="PRIOR-001",
            age=80,
            sex="F",
            admission_date=date(2023, 1, 1),
            discharge_date=date(2023, 1, 5),
            primary_diagnosis="K92.2",
            readmitted_30d=False,
        )
    )
    assert dao.count() == 1

    rows = [
        {
            "mrn": "GOLD-NEW-001",
            "age": 65,
            "sex": "M",
            "admission_date": date(2024, 5, 1),
            "discharge_date": date(2024, 5, 4),
            "primary_diagnosis": "I50.9",
            "readmitted_30d": True,
            "ingested_at": datetime(2024, 6, 1),
            "source_file": "test.csv",
        },
    ]
    silver_path = tmp_path / "silver"
    spark.createDataFrame(rows).write.format("delta").save(str(silver_path))

    run_gold(spark, silver_path, mode="overwrite")

    assert dao.count() == 1
    assert dao.find_by_mrn("PRIOR-001") is None
    assert dao.find_by_mrn("GOLD-NEW-001") is not None
