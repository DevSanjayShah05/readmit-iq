"""
Tests for feature engineering.

We test that each Patient field correctly maps to the expected feature
columns, that one-hot encoding is mutually exclusive, that empty cohorts
are handled, and that column order matches the spec.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from readmit_iq.features.feature_engineering import (
    _age_bucket,
    _length_of_stay_days,
    _patient_to_dict,
    patients_to_features,
)
from readmit_iq.features.feature_spec import (
    LABEL_COLUMN,
    feature_names,
)
from readmit_iq.models import Patient


def _make(
    mrn: str = "TEST",
    age: int = 65,
    sex: str = "M",
    diagnosis: str | None = "I50.9",
    admission: date = date(2024, 6, 1),
    discharge: date = date(2024, 6, 8),
    readmitted: bool = False,
) -> Patient:
    """Build a Patient with sensible defaults for tests."""
    return Patient(
        mrn=mrn,
        age=age,
        sex=sex,
        admission_date=admission,
        discharge_date=discharge,
        primary_diagnosis=diagnosis,
        readmitted_30d=readmitted,
    )


def test_length_of_stay_simple_case() -> None:
    """LOS is discharge minus admission."""
    p = _make(admission=date(2024, 6, 1), discharge=date(2024, 6, 8))
    assert _length_of_stay_days(p) == 7


def test_length_of_stay_same_day() -> None:
    """Same-day discharge -> LOS of 0."""
    p = _make(admission=date(2024, 6, 1), discharge=date(2024, 6, 1))
    assert _length_of_stay_days(p) == 0


@pytest.mark.parametrize(
    "age,expected",
    [
        (18, "under_50"),
        (49, "under_50"),
        (50, "age_50_69"),
        (69, "age_50_69"),
        (70, "age_70_plus"),
        (95, "age_70_plus"),
    ],
)
def test_age_bucket_boundaries(age: int, expected: str) -> None:
    """Age bucket boundaries: 50 starts 50_69, 70 starts 70_plus."""
    assert _age_bucket(age) == expected


def test_patient_to_dict_has_all_expected_keys() -> None:
    """The dict should contain exactly the columns from feature_names()."""
    p = _make()
    row = _patient_to_dict(p)
    expected_keys = set(feature_names())
    actual_keys = set(row.keys())
    assert actual_keys == expected_keys


def test_one_hot_encoding_is_mutually_exclusive() -> None:
    """Exactly one sex_* column should be 1; the rest should be 0."""
    row = _patient_to_dict(_make(sex="F"))
    sex_columns = {k: v for k, v in row.items() if k.startswith("sex_")}
    assert sum(sex_columns.values()) == 1
    assert sex_columns["sex_F"] == 1


def test_unknown_diagnosis_buckets_to_unknown() -> None:
    """A diagnosis not in DIAGNOSIS_VALUES should be one-hot as UNKNOWN."""
    row = _patient_to_dict(_make(diagnosis="X99.9"))  # not in known list
    assert row["dx_UNKNOWN"] == 1
    assert row["dx_I50.9"] == 0


def test_null_diagnosis_buckets_to_unknown() -> None:
    """A None diagnosis should be one-hot as UNKNOWN."""
    row = _patient_to_dict(_make(diagnosis=None))
    assert row["dx_UNKNOWN"] == 1


def test_patients_to_features_preserves_column_order() -> None:
    """DataFrame columns must exactly match feature_names() order."""
    patients = [_make()]
    df = patients_to_features(patients, include_label=False)
    assert list(df.columns) == feature_names()


def test_patients_to_features_includes_label_when_asked() -> None:
    """include_label=True should add the readmitted_30d column."""
    patients = [_make(readmitted=True)]
    df = patients_to_features(patients, include_label=True)
    assert LABEL_COLUMN in df.columns
    assert df[LABEL_COLUMN].iloc[0] is True or df[LABEL_COLUMN].iloc[0] == 1


def test_patients_to_features_omits_label_when_asked() -> None:
    """include_label=False should not include the label column."""
    df = patients_to_features([_make()], include_label=False)
    assert LABEL_COLUMN not in df.columns


def test_empty_cohort_returns_dataframe_with_correct_columns() -> None:
    """Empty input -> empty DataFrame with right columns, not a crash."""
    df = patients_to_features([], include_label=True)
    assert len(df) == 0
    assert LABEL_COLUMN in df.columns
    assert all(col in df.columns for col in feature_names())


def test_dataframe_shape_matches_input() -> None:
    """3 patients -> 3 rows. Feature column count = len(feature_names())."""
    patients = [_make(mrn=f"P{i}") for i in range(3)]
    df = patients_to_features(patients, include_label=False)
    assert df.shape == (3, len(feature_names()))
