"""
Prediction interface for the readmission classifier.

Loads a trained model from disk and produces readmission probabilities
for new patients. The predict function is what an API endpoint or
batch-scoring job would call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from loguru import logger

from readmit_iq.features.feature_engineering import patients_to_features
from readmit_iq.features.feature_spec import feature_names
from readmit_iq.models import Patient


class ReadmissionPredictor:
    """Loaded model + the prediction interface around it."""

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
        logger.info(f"Loaded model from {self.model_path}")

    def predict_proba(self, patients: Sequence[Patient]) -> np.ndarray:
        """
        Predict readmission probabilities for a cohort of patients.

        Returns a numpy array of probabilities (one per patient), each in [0, 1].
        """
        if not patients:
            return np.array([])

        X = patients_to_features(patients, include_label=False)
        # Defensive: confirm the column order matches what the model was trained on.
        # If anyone changes feature_spec without retraining, this catches it loud.
        expected_cols = feature_names()
        if list(X.columns) != expected_cols:
            raise ValueError(
                "Feature columns don't match the trained model. "
                "Retrain the model after changing feature_spec.py."
            )
        return self.model.predict_proba(X)[:, 1]

    def predict_label(
        self, patients: Sequence[Patient], threshold: float = 0.5
    ) -> np.ndarray:
        """Predict binary readmission labels using a probability threshold."""
        probs = self.predict_proba(patients)
        return (probs >= threshold).astype(int)
