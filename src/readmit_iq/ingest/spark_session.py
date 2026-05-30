"""
Spark session factory for the ReadmitIQ data pipeline.

Centralizes Spark config so bronze/silver/gold layers don't duplicate it
and so tests can reuse the same settings as production.

Two ways to add extra JVM libraries:
- extra_packages: Maven coordinates, resolved on the fly. Convenient but
  flaky if a Spark session already exists (config gets ignored).
- extra_jars: explicit filesystem paths. More reliable.
"""

from __future__ import annotations

from delta import configure_spark_with_delta_pip
from loguru import logger
from pyspark.sql import SparkSession


def get_spark_session(
    app_name: str = "readmit-iq",
    extra_packages: list[str] | None = None,
    extra_jars: list[str] | None = None,
) -> SparkSession:
    """
    Build or fetch a SparkSession configured for Delta Lake.

    Args:
        app_name: shows up in the Spark UI and logs.
        extra_packages: optional Maven coordinates for extra JARs to fetch.
        extra_jars: optional list of local JAR file paths to add to the
            classpath. More reliable than extra_packages when a Spark
            session might already exist (its config gets ignored on
            re-use; jars passed by path are always honored).

    Returns:
        A ready-to-use SparkSession. Idempotent: calling twice returns
        the same session.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    if extra_packages:
        builder = builder.config("spark.jars.packages", ",".join(extra_packages))
    if extra_jars:
        builder = builder.config("spark.jars", ",".join(extra_jars))

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"Spark session ready (version {spark.version}, app={app_name})")
    return spark
