"""
Bronze layer: raw ingestion of incoming patient data.

The bronze layer's job is to durably capture inbound data with minimal
transformation. We add two metadata columns (ingested_at, source_file)
so downstream layers can trace the lineage of any record, and we write
to a Delta table for ACID guarantees + time travel.

Usage:
    python -m readmit_iq.ingest.bronze --input data/raw --output data/bronze/patient_raw
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name, lit

from readmit_iq.ingest.spark_session import get_spark_session


def ingest_raw_csv(
    spark: SparkSession,
    input_path: str | Path,
    output_path: str | Path,
    mode: str = "append",
) -> int:
    """
    Read raw CSVs from input_path and write them to a Delta table at output_path.

    Args:
        spark: SparkSession to use.
        input_path: directory containing patient CSVs (any matching *.csv).
        output_path: directory where the Delta table is written.
        mode: 'append' (default), 'overwrite', or 'ignore'.
            - 'append' adds new rows; safe for incremental loads.
            - 'overwrite' replaces the table entirely; useful for full reloads.

    Returns:
        Number of rows ingested.
    """
    input_path = str(input_path)
    output_path = str(output_path)

    logger.info(f"Reading CSVs from {input_path}")
    # spark.read.csv with header=True and inferSchema=True is convenient but
    # slow (it scans the file twice). In production we'd declare an explicit
    # schema. For bronze, where we deliberately accept loose typing, inferring
    # is fine.
    raw_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(f"{input_path}/*.csv")
    )

    # Attach lineage metadata to every row. input_file_name() returns the
    # full path of the file each row came from; current_timestamp() is the
    # ingest time. These two columns are the bronze layer's contribution.
    enriched_df = raw_df.withColumn("ingested_at", current_timestamp()).withColumn(
        "source_file", input_file_name()
    )

    n_rows = enriched_df.count()
    logger.info(f"Read {n_rows:,} rows; writing to Delta at {output_path}")

    (enriched_df.write.format("delta").mode(mode).save(output_path))

    logger.success(f"Bronze ingest complete: {n_rows:,} rows -> {output_path}")
    return n_rows


def read_bronze(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read the bronze Delta table as a Spark DataFrame."""
    return spark.read.format("delta").load(str(path))


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Ingest raw patient CSVs to bronze Delta table."
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/bronze/patient_raw"))
    parser.add_argument(
        "--mode", choices=["append", "overwrite", "ignore"], default="append"
    )
    args = parser.parse_args()

    spark = get_spark_session("readmit-iq-bronze")
    try:
        ingest_raw_csv(spark, args.input, args.output, mode=args.mode)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
