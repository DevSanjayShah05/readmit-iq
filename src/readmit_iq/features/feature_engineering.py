"""
Feature engineering: convert Patient objects to model-ready DataFrames.

The functions here are pure — they take inputs, return outputs, and have
no side effects. This makes them trivially testable and trivially safe to
call in parallel (which matters when we eventually do this in Spark).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from loguru import logger

from readmit_iq.features.feature_spec import (
    AGE_BUCKETS,
    DIAGNOSIS_VALUES,
    LABEL_COLUMN,
    SEX_VALUES,
    feature_names,
)
from readmit_iq.models import Patient


def _length_of_stay_days(p: Patient) -> int:
    """Compute length of stay from admission/discharge dates."""
    return (p.discharge_date - p.admission_date).days


def _age_bucket(age: int) -> str:
    """Bin age into a small set of clinically meaningful buckets."""
    if age < 50:
        return "under_50"
    elif age < 70:
        return "age_50_69"
    else:
        return "age_70_plus"


def _patient_to_dict(p: Patient) -> dict[str, int | float]:
    """
    Convert one Patient into a dict of feature columns. The keys exactly
    match feature_names() from feature_spec.
    """
    row: dict[str, int | float] = {}

    # Numeric features
    row["age"] = p.age
    row["length_of_stay_days"] = _length_of_stay_days(p)

    # One-hot encode sex: one column per value, 1 where the value matches.
    for sex in SEX_VALUES:
        row[f"sex_{sex}"] = 1 if p.sex == sex else 0

    # One-hot encode primary diagnosis. Missing diagnoses go to "UNKNOWN".
    actual_dx = p.primary_diagnosis if p.primary_diagnosis else "UNKNOWN"
    if actual_dx not in DIAGNOSIS_VALUES:
        # An unrecognized diagnosis code also goes into the UNKNOWN bucket.
        # Logged because in real systems this often signals a data issue.
        logger.debug(f"Unknown diagnosis code '{actual_dx}' bucketed to UNKNOWN")
        actual_dx = "UNKNOWN"
    for dx in DIAGNOSIS_VALUES:
        row[f"dx_{dx}"] = 1 if actual_dx == dx else 0

    # One-hot encode age bucket
    bucket = _age_bucket(p.age)
    for b in AGE_BUCKETS:
        row[f"age_bucket_{b}"] = 1 if bucket == b else 0

    return row


def patients_to_features(
    patients: Sequence[Patient],
    include_label: bool = True,
) -> pd.DataFrame:
    """
    Convert a sequence of Patients to a model-ready feature DataFrame.

    Args:
        patients: the input cohort.
        include_label: if True, includes the readmitted_30d column. Set to
                       False at prediction time, when the label is unknown.

    Returns:
        DataFrame with rows = patients and columns = feature_names() (and
        optionally the label). Column order matches feature_spec exactly.
    """
    if not patients:
        # Empty cohort: return an empty DataFrame with the right columns
        # so downstream code never has to special-case empty input.
        cols = feature_names()
        if include_label:
            cols.append(LABEL_COLUMN)
        return pd.DataFrame(columns=cols)

    rows = [_patient_to_dict(p) for p in patients]
    df = pd.DataFrame(rows, columns=feature_names())

    if include_label:
        df[LABEL_COLUMN] = [p.readmitted_30d for p in patients]

    logger.info(
        f"Engineered features for {len(df):,} patients "
        f"({len(df.columns)} columns, label={'yes' if include_label else 'no'})"
    )
    return df
