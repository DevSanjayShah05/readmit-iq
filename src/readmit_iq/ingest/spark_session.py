"""
Spark session factory for the ReadmitIQ data pipeline.

Centralizes Spark config so bronze/silver/gold layers don't duplicate it
and so tests can reuse the same settings as production. Local development
runs against the laptop's cores; on a real cluster the master URL changes
and the rest of the code is identical.
"""

from __future__ import annotations

from delta import configure_spark_with_delta_pip
from loguru import logger
from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "readmit-iq") -> SparkSession:
    """
    Build or fetch a SparkSession configured for Delta Lake.

    Args:
        app_name: shows up in the Spark UI and logs. Useful when debugging
            multiple concurrent jobs.

    Returns:
        A ready-to-use SparkSession. Idempotent — calling twice returns
        the same session.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        # These two lines wire Delta Lake into the SQL engine so that
        # `spark.read.format("delta")` and `df.write.format("delta")` work.
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    # delta-spark provides this helper that auto-downloads the right Delta
    # JAR for the running Spark version. Saves us from pinning JARs by hand.
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"Spark session ready (version {spark.version}, app={app_name})")
    return spark
