"""
Feature specification — the single source of truth for what features the
model consumes.

This file is the *contract* between feature engineering and the rest of
the ML pipeline. The training code reads FEATURE_NAMES; the prediction
code reads FEATURE_NAMES; the feature engineering code produces a
DataFrame with exactly these columns in this order. Mismatch = silent
prediction bugs, so we centralize the definition here.

When you add a new feature:
  1. Add it to FEATURE_NAMES below.
  2. Implement it in feature_engineering.py.
  3. Add a test for it.
  4. Retrain the model (old models won't expect the new column).
"""

from __future__ import annotations

# Numeric features (computed from raw fields)
NUMERIC_FEATURES: tuple[str, ...] = (
    "age",
    "length_of_stay_days",
)

# One-hot encoded categorical features. The model sees binary 0/1 columns;
# we list the *base* categorical fields here, and the engineering code
# expands each into one column per value.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "sex",  # expands to sex_F, sex_M, sex_O
    "primary_diagnosis",  # expands to one column per diagnosis code
)

# Bucketized numeric features (age binned into ranges)
BUCKETED_FEATURES: tuple[str, ...] = (
    "age_bucket",  # under_50 / age_50_69 / age_70_plus
)

# Known categorical values, used for one-hot encoding. Fixing these here
# (rather than learning them from training data) means prediction works
# even if a particular batch happens to have no patients of a given sex.
SEX_VALUES: tuple[str, ...] = ("F", "M", "O")
DIAGNOSIS_VALUES: tuple[str, ...] = (
    "I50.9",  # Heart failure
    "J44.9",  # COPD
    "J18.9",  # Pneumonia
    "E11.9",  # Diabetes
    "N17.9",  # Acute kidney injury
    "I63.9",  # Stroke
    "I21.9",  # Acute MI
    "K92.2",  # GI bleed
    "A41.9",  # Sepsis
    "Z51.5",  # Palliative
    "UNKNOWN",
)
AGE_BUCKETS: tuple[str, ...] = ("under_50", "age_50_69", "age_70_plus")


def feature_names() -> list[str]:
    """
    The full ordered list of feature column names the model expects.
    This is what the model is trained on; predictions must match exactly.
    """
    names = list(NUMERIC_FEATURES)
    for sex in SEX_VALUES:
        names.append(f"sex_{sex}")
    for dx in DIAGNOSIS_VALUES:
        names.append(f"dx_{dx}")
    for bucket in AGE_BUCKETS:
        names.append(f"age_bucket_{bucket}")
    return names


# The label column the model predicts. Kept separate from features
# because at prediction time we have features but no label.
LABEL_COLUMN: str = "readmitted_30d"
