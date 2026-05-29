"""
Tests for the model training and prediction pipelines.

We train on a small synthetic cohort and verify the resulting model
produces sensible outputs and roundtrips through joblib correctly.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from readmit_iq.dao.patient_dao import PatientDAO
from readmit_iq.ml.predict import ReadmissionPredictor
from readmit_iq.ml.train import TrainingResult, train_model
from readmit_iq.models import Patient
from readmit_iq.scripts.seed_data import seed_patients


def test_train_returns_metrics_and_saves_model(
    clean_patient_table, tmp_path: Path
) -> None:
    """End-to-end: seed data, train, verify metrics and model file."""
    seed_patients(count=300, seed=42, truncate=False)
    patients = PatientDAO().find_admissions_between(
        date(1900, 1, 1), date(2099, 12, 31)
    )

    output = tmp_path / "test_model.joblib"
    result = train_model(patients=patients, output_path=output, n_estimators=20)

    assert isinstance(result, TrainingResult)
    assert result.n_train + result.n_test == 300
    assert 0.0 <= result.auc <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert output.exists()


def test_train_raises_on_too_few_patients(clean_patient_table, tmp_path: Path) -> None:
    """Fewer than 30 patients should refuse to train, not silently produce garbage."""
    with pytest.raises(ValueError, match="at least 30"):
        train_model(patients=[], output_path=tmp_path / "model.joblib")


def test_predictor_loads_trained_model(clean_patient_table, tmp_path: Path) -> None:
    """A model written by train_model should be loadable by ReadmissionPredictor."""
    seed_patients(count=200, seed=7, truncate=False)
    patients = PatientDAO().find_admissions_between(
        date(1900, 1, 1), date(2099, 12, 31)
    )

    model_path = tmp_path / "loadable.joblib"
    train_model(patients=patients, output_path=model_path, n_estimators=20)

    predictor = ReadmissionPredictor(model_path=model_path)
    probs = predictor.predict_proba(patients[:5])
    assert len(probs) == 5
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_predictor_handles_empty_input(clean_patient_table, tmp_path: Path) -> None:
    """Predicting on an empty cohort should return an empty array, not crash."""
    seed_patients(count=200, seed=7, truncate=False)
    patients = PatientDAO().find_admissions_between(
        date(1900, 1, 1), date(2099, 12, 31)
    )
    model_path = tmp_path / "model.joblib"
    train_model(patients=patients, output_path=model_path, n_estimators=20)

    predictor = ReadmissionPredictor(model_path=model_path)
    probs = predictor.predict_proba([])
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 0


def test_predictor_raises_on_missing_model_file() -> None:
    """A nonexistent model path should raise FileNotFoundError, not hang."""
    with pytest.raises(FileNotFoundError, match="Model not found"):
        ReadmissionPredictor(model_path="/nonexistent/path/model.joblib")


def test_predictor_higher_risk_for_clinical_signal(
    clean_patient_table, tmp_path: Path
) -> None:
    """An elderly heart-failure patient should score higher than a young palliative one."""
    seed_patients(count=500, seed=42, truncate=False)
    patients = PatientDAO().find_admissions_between(
        date(1900, 1, 1), date(2099, 12, 31)
    )
    model_path = tmp_path / "model.joblib"
    train_model(patients=patients, output_path=model_path, n_estimators=50)

    high_risk = Patient(
        mrn="HIGH",
        age=85,
        sex="M",
        admission_date=date(2024, 6, 1),
        discharge_date=date(2024, 6, 15),
        primary_diagnosis="I50.9",
        readmitted_30d=False,
    )
    low_risk = Patient(
        mrn="LOW",
        age=35,
        sex="F",
        admission_date=date(2024, 7, 1),
        discharge_date=date(2024, 7, 2),
        primary_diagnosis="Z51.5",
        readmitted_30d=False,
    )

    predictor = ReadmissionPredictor(model_path=model_path)
    probs = predictor.predict_proba([high_risk, low_risk])
    assert (
        probs[0] > probs[1]
    ), f"Expected high>low, got {probs[0]:.3f} vs {probs[1]:.3f}"
