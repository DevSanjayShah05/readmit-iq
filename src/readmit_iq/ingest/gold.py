"""
Gold layer: loads silver-cleaned data into Postgres.

Reads from the silver Delta table at data/silver/patient_silver and writes
into the Postgres `patient` table -- the same table our DAO/services/ML/API
all read from. After gold runs, the application stack runs on data that
flowed through the Spark medallion.

Write semantics: 'overwrite' for now.
- Silver is our source of truth going forward.
- Overwrite means: truncate `patient`, then bulk insert silver rows.
- The Faker-based seed_data.py path stays for development/testing convenience,
  but production data comes from this pipeline.

In a real shop, you'd use an upsert pattern instead (stage to temp table,
then INSERT ... ON CONFLICT DO UPDATE). We can refactor to that later.

Usage:
    python -m readmit_iq.ingest.gold \\
        --silver data/silver/patient_silver
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger
from pyspark.sql import DataFrame, SparkSession

from readmit_iq.config import get_settings
from readmit_iq.ingest.spark_session import get_spark_session

# Path to the Postgres JDBC driver, downloaded to .venv/jars/ during setup.
# Using an explicit path (rather than Maven coordinates) avoids issues where
# Spark ignores `spark.jars.packages` if a session already exists.
POSTGRES_JDBC_JAR = Path(".venv/jars/postgresql-42.7.3.jar").resolve()

# Columns that exist in silver but should NOT be written to Postgres.
# `ingested_at` and `source_file` are lineage metadata; Postgres doesn't
# have columns for them. (A more sophisticated design would carry them
# into a separate `patient_lineage` table.)
SILVER_ONLY_COLUMNS = {"ingested_at", "source_file"}

# The Postgres `patient` table's column order (from the Alembic migration),
# excluding auto-generated `id` and `created_at`. Spark JDBC writes are
# positional, so we project silver into this exact order before writing.
PATIENT_COLUMNS = [
    "mrn",
    "age",
    "sex",
    "admission_date",
    "discharge_date",
    "primary_diagnosis",
    "readmitted_30d",
]


def _project_to_patient_schema(silver_df: DataFrame) -> DataFrame:
    """
    Drop silver-only columns and reorder to match the Postgres `patient` table.

    Postgres-managed columns (`id`, `created_at`) are not written by gold;
    Postgres generates them via SERIAL and DEFAULT NOW().
    """
    return silver_df.select(*PATIENT_COLUMNS)


def _jdbc_url_for_spark(database_url: str) -> str:
    """
    Convert a SQLAlchemy/psycopg-style URL to a Spark JDBC URL.

    SQLAlchemy:  postgresql://user:pass@host:port/db
    Spark JDBC:  jdbc:postgresql://host:port/db (auth goes in properties)
    """
    if database_url.startswith("postgresql+psycopg://"):
        database_url = database_url.replace("postgresql+psycopg://", "postgresql://")
    if not database_url.startswith("postgresql://"):
        raise ValueError(f"Expected a postgresql:// URL, got: {database_url}")

    without_scheme = database_url[len("postgresql://") :]
    if "@" in without_scheme:
        _, host_part = without_scheme.split("@", 1)
    else:
        host_part = without_scheme
    return f"jdbc:postgresql://{host_part}"


def _parse_credentials(database_url: str) -> tuple[str, str]:
    """Extract (user, password) from the DATABASE_URL."""
    if database_url.startswith("postgresql+psycopg://"):
        database_url = database_url.replace("postgresql+psycopg://", "postgresql://")
    without_scheme = database_url[len("postgresql://") :]
    if "@" not in without_scheme:
        raise ValueError("DATABASE_URL missing credentials (no @ found)")
    creds, _ = without_scheme.split("@", 1)
    if ":" not in creds:
        raise ValueError("DATABASE_URL missing password (no : in credentials)")
    user, password = creds.split(":", 1)
    return user, password


def write_to_postgres(
    df: DataFrame,
    table: str,
    database_url: str,
    mode: str = "overwrite",
) -> None:
    """
    Bulk-write a Spark DataFrame to a Postgres table via JDBC.

    Note on 'overwrite': Spark's default would DROP and RECREATE the table,
    which loses our Alembic-managed schema. We set `truncate=true` to make
    it TRUNCATE-and-insert, preserving the schema and indexes.
    """
    jdbc_url = _jdbc_url_for_spark(database_url)
    user, password = _parse_credentials(database_url)

    logger.info(f"Writing {df.count():,} rows to {table} ({jdbc_url}, mode={mode})")

    writer = (
        df.write.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table)
        .option("user", user)
        .option("password", password)
        .option("driver", "org.postgresql.Driver")
        .option("truncate", "true")
        .mode(mode)
    )
    writer.save()
    logger.success(f"Postgres write complete: {table}")


def run_gold(
    spark: SparkSession,
    silver_path: str | Path,
    database_url: str | None = None,
    table: str = "patient",
    mode: str = "overwrite",
) -> int:
    """
    Read silver, project to the Postgres schema, write to the `patient` table.
    Returns the number of rows written.
    """
    silver_path = str(silver_path)
    if database_url is None:
        database_url = get_settings().database_url

    logger.info(f"Reading silver from {silver_path}")
    silver_df = spark.read.format("delta").load(silver_path)
    n = silver_df.count()
    logger.info(f"Silver rows: {n:,}")

    gold_df = _project_to_patient_schema(silver_df)
    logger.info(f"Projected to Postgres schema. Columns: {gold_df.columns}")

    write_to_postgres(gold_df, table=table, database_url=database_url, mode=mode)
    logger.success(f"Gold complete: {n:,} rows written to Postgres.{table}")
    return n


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Load silver data into Postgres patient table."
    )
    parser.add_argument(
        "--silver", type=Path, default=Path("data/silver/patient_silver")
    )
    parser.add_argument("--table", default="patient")
    parser.add_argument("--mode", choices=["append", "overwrite"], default="overwrite")
    args = parser.parse_args()

    spark = get_spark_session(
        app_name="readmit-iq-gold",
        extra_jars=[str(POSTGRES_JDBC_JAR)],
    )
    try:
        run_gold(spark, args.silver, table=args.table, mode=args.mode)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
