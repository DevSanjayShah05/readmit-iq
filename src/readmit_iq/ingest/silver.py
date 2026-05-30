"""
Silver layer: clean, validated, deduplicated patient data.

Reads from the bronze Delta table at data/bronze/patient_raw and writes
to a silver Delta table at data/silver/patient_silver. Each silver row
represents one unique patient (by MRN), with validated values.

Transformations:
- Deduplicate by MRN, keeping the row with the latest ingested_at.
  This implements 'corrections win' semantics: if a source system re-emits
  a patient record (because the data was corrected upstream), the latest
  version wins.
- Validate age is between 0 and 120; null it if not (don't drop the row).
- Normalize sex to uppercase and validate it's in {F, M, O}; null if not.
- Drop rows where MRN is null (useless without a patient identifier).
- Carry forward ingested_at and source_file for downstream lineage.

Usage:
    python -m readmit_iq.ingest.silver \\
        --bronze data/bronze/patient_raw \\
        --silver data/silver/patient_silver
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import col, row_number, upper, when

from readmit_iq.ingest.spark_session import get_spark_session


def _validate_age(df: DataFrame) -> DataFrame:
    """Null out age values outside 0-120 (invalid)."""
    return df.withColumn(
        "age",
        when((col("age") >= 0) & (col("age") <= 120), col("age")).otherwise(None),
    )


def _normalize_sex(df: DataFrame) -> DataFrame:
    """Uppercase sex and null out anything not in {F, M, O}."""
    valid = upper(col("sex")).isin("F", "M", "O")
    return df.withColumn(
        "sex",
        when(valid, upper(col("sex"))).otherwise(None),
    )


def _drop_rows_without_mrn(df: DataFrame) -> DataFrame:
    """Drop rows where MRN is null or empty — they can't be patient records."""
    return df.filter(col("mrn").isNotNull() & (col("mrn") != ""))


def _deduplicate_by_mrn(df: DataFrame) -> DataFrame:
    """
    Keep one row per MRN: the one with the latest ingested_at.

    Window functions are SQL's way of saying 'rank rows within a group.'
    Here we partition by MRN (group rows with the same patient ID) and
    sort within each group by ingested_at descending (newest first).
    row_number() assigns 1 to the newest row in each group, 2 to the
    next, and so on. We keep only row_num=1.
    """
    window = Window.partitionBy("mrn").orderBy(col("ingested_at").desc())
    return (
        df.withColumn("_row_num", row_number().over(window))
        .filter(col("_row_num") == 1)
        .drop("_row_num")
    )


def transform_bronze_to_silver(bronze_df: DataFrame) -> DataFrame:
    """
    Apply all silver-layer transformations to a bronze DataFrame.

    Pulled out as a function so tests can pass in any bronze-shaped
    DataFrame and verify the transformations.
    """
    return (
        bronze_df.transform(_drop_rows_without_mrn)
        .transform(_validate_age)
        .transform(_normalize_sex)
        .transform(_deduplicate_by_mrn)
    )


def run_silver(
    spark: SparkSession,
    bronze_path: str | Path,
    silver_path: str | Path,
    mode: str = "overwrite",
) -> int:
    """
    Read bronze, transform, write silver. Returns silver row count.
    """
    bronze_path = str(bronze_path)
    silver_path = str(silver_path)

    logger.info(f"Reading bronze from {bronze_path}")
    bronze_df = spark.read.format("delta").load(bronze_path)
    bronze_count = bronze_df.count()
    logger.info(f"Bronze rows: {bronze_count:,}")

    silver_df = transform_bronze_to_silver(bronze_df)
    silver_count = silver_df.count()
    logger.info(
        f"Silver rows: {silver_count:,} "
        f"(dropped/deduplicated: {bronze_count - silver_count:,})"
    )

    silver_df.write.format("delta").mode(mode).save(silver_path)
    logger.success(f"Silver write complete: {silver_count:,} rows -> {silver_path}")
    return silver_count


def read_silver(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read the silver Delta table as a Spark DataFrame."""
    return spark.read.format("delta").load(str(path))


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Clean bronze patient data into silver."
    )
    parser.add_argument("--bronze", type=Path, default=Path("data/bronze/patient_raw"))
    parser.add_argument(
        "--silver", type=Path, default=Path("data/silver/patient_silver")
    )
    parser.add_argument("--mode", choices=["append", "overwrite"], default="overwrite")
    args = parser.parse_args()

    spark = get_spark_session("readmit-iq-silver")
    try:
        run_silver(spark, args.bronze, args.silver, mode=args.mode)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
