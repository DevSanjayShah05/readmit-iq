"""
SHAP-based explanations for the readmission classifier.

For every prediction the model makes, we can decompose the predicted
probability into per-feature contributions:

    P(readmit) = baseline + sum(contribution of each feature)

Positive contribution -> this feature pushed risk UP for this patient.
Negative contribution -> this feature pulled risk DOWN.

These explanations are what we'll show clinicians alongside each prediction.

We use TreeSHAP (shap.TreeExplainer), an exact and fast algorithm for
tree-based models. For Random Forest specifically, it computes the
contribution of each feature in milliseconds per patient.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import shap
from loguru import logger

from readmit_iq.features.feature_engineering import patients_to_features
from readmit_iq.features.feature_spec import feature_names
from readmit_iq.models import Patient


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's contribution to one patient's prediction."""

    feature_name: str
    feature_value: float
    shap_value: float


@dataclass(frozen=True)
class PatientExplanation:
    """SHAP explanation for one patient's predicted readmission probability."""

    mrn: str
    predicted_probability: float
    baseline_probability: float
    contributions: tuple[FeatureContribution, ...]

    def top_features(self, n: int = 5) -> tuple[FeatureContribution, ...]:
        """The N features with the largest absolute contribution, sorted descending."""
        ranked = sorted(
            self.contributions, key=lambda c: abs(c.shap_value), reverse=True
        )
        return tuple(ranked[:n])

    def positive_features(self) -> tuple[FeatureContribution, ...]:
        """Features that pushed the prediction UP (risk increases)."""
        return tuple(c for c in self.contributions if c.shap_value > 0)

    def negative_features(self) -> tuple[FeatureContribution, ...]:
        """Features that pulled the prediction DOWN (risk decreases)."""
        return tuple(c for c in self.contributions if c.shap_value < 0)


class ShapExplainer:
    """Compute SHAP explanations for trained Random Forest predictions."""

    def __init__(
        self, model_path: Path | str = Path("models/readmit_rf.joblib")
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Train one first with: python -m readmit_iq.ml.train"
            )
        self.model = joblib.load(self.model_path)
        # TreeExplainer is the exact, efficient SHAP variant for tree models.
        # The explainer needs the model itself, not the training data, because
        # tree-based SHAP works purely from the tree structure.
        self.explainer = shap.TreeExplainer(self.model)
        logger.info(f"Loaded model and built SHAP explainer from {self.model_path}")

    def explain(self, patients: Sequence[Patient]) -> tuple[PatientExplanation, ...]:
        """Produce a SHAP explanation for each patient in the cohort."""
        if not patients:
            return ()

        X = patients_to_features(patients, include_label=False)
        if list(X.columns) != feature_names():
            raise ValueError(
                "Feature columns don't match feature_spec; retrain the model."
            )

        # shap_values for a binary classifier returns shape (n_patients, n_features, 2).
        # The last axis is the two classes; we want index 1 (the positive class:
        # readmitted). Some shap versions return a plain ndarray, others return
        # an Explanation object; we normalize to ndarray.
        raw = self.explainer.shap_values(X)
        if isinstance(raw, list):
            # Older shap: list of arrays, one per class.
            shap_arr = raw[1]
        elif raw.ndim == 3:
            # Newer shap: single 3D array.
            shap_arr = raw[..., 1]
        else:
            shap_arr = raw

        # Baseline probability — what the model would predict if we knew nothing
        # about the patient. For a RandomForestClassifier this is roughly the
        # base rate of the positive class in training.
        expected_value = self.explainer.expected_value
        if (
            isinstance(expected_value, (list, np.ndarray))
            and np.ndim(expected_value) > 0
        ):
            baseline = float(expected_value[1])
        else:
            baseline = float(expected_value)

        # Current predicted probabilities for sanity-check / display
        proba = self.model.predict_proba(X)[:, 1]

        cols = list(X.columns)
        explanations: list[PatientExplanation] = []
        for i, patient in enumerate(patients):
            contribs = tuple(
                FeatureContribution(
                    feature_name=cols[j],
                    feature_value=float(X.iloc[i, j]),
                    shap_value=float(shap_arr[i, j]),
                )
                for j in range(len(cols))
            )
            explanations.append(
                PatientExplanation(
                    mrn=patient.mrn,
                    predicted_probability=float(proba[i]),
                    baseline_probability=baseline,
                    contributions=contribs,
                )
            )

        logger.info(f"Generated SHAP explanations for {len(explanations)} patients")
        return tuple(explanations)
