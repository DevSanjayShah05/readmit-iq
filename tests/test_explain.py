"""
Tests for SHAP-based explanations.

We verify the shape and consistency of explanations, plus a few clinical
sanity properties — e.g., a high-risk patient's explanation should weight
clinically-sensible features highly.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from readmit_iq.dao.patient_dao import PatientDAO
from readmit_iq.ml.explain import (
    FeatureContribution,
    PatientExplanation,
    ShapExplainer,
)
from readmit_iq.ml.train import train_model
from readmit_iq.models import Patient
from readmit_iq.scripts.seed_data import seed_patients


@pytest.fixture
def trained_model_path(clean_patient_table, tmp_path: Path) -> Path:
    """Seed data and train a small model. Returns the path to the .joblib."""
    seed_patients(count=400, seed=42, truncate=False)
    patients = PatientDAO().find_admissions_between(
        date(1900, 1, 1), date(2099, 12, 31)
    )
    path = tmp_path / "explain_test_model.joblib"
    train_model(patients=patients, output_path=path, n_estimators=30)
    return path


def _high_risk_patient() -> Patient:
    """Elderly heart-failure patient with a long stay — should score high."""
    return Patient(
        mrn="HIGH",
        age=88,
        sex="M",
        admission_date=date(2024, 6, 1),
        discharge_date=date(2024, 6, 18),
        primary_diagnosis="I50.9",
        readmitted_30d=False,
    )


def _low_risk_patient() -> Patient:
    """Young patient with a palliative encounter and short stay — should score low."""
    return Patient(
        mrn="LOW",
        age=32,
        sex="F",
        admission_date=date(2024, 7, 1),
        discharge_date=date(2024, 7, 2),
        primary_diagnosis="Z51.5",
        readmitted_30d=False,
    )


def test_explain_returns_one_explanation_per_patient(trained_model_path: Path) -> None:
    """N patients in -> N PatientExplanations out, with matching mrns."""
    explainer = ShapExplainer(model_path=trained_model_path)
    patients = [_high_risk_patient(), _low_risk_patient()]
    explanations = explainer.explain(patients)
    assert len(explanations) == 2
    assert [e.mrn for e in explanations] == ["HIGH", "LOW"]


def test_explain_handles_empty_input(trained_model_path: Path) -> None:
    """Explaining zero patients should return an empty tuple, not crash."""
    explainer = ShapExplainer(model_path=trained_model_path)
    explanations = explainer.explain([])
    assert explanations == ()


def test_shap_values_sum_to_prediction_minus_baseline(trained_model_path: Path) -> None:
    """
    SHAP's mathematical guarantee: sum(shap_values) == prediction - baseline.
    If this ever breaks, the explanation is no longer trustworthy.
    """
    explainer = ShapExplainer(model_path=trained_model_path)
    e = explainer.explain([_high_risk_patient()])[0]
    contribution_sum = sum(c.shap_value for c in e.contributions)
    expected = e.predicted_probability - e.baseline_probability
    assert contribution_sum == pytest.approx(expected, abs=1e-3), (
        f"SHAP additivity violated: sum={contribution_sum:.4f}, "
        f"prediction-baseline={expected:.4f}"
    )


def test_explanation_has_one_contribution_per_feature(trained_model_path: Path) -> None:
    """A PatientExplanation should have exactly len(feature_names()) contributions."""
    from readmit_iq.features.feature_spec import feature_names

    explainer = ShapExplainer(model_path=trained_model_path)
    e = explainer.explain([_high_risk_patient()])[0]
    assert len(e.contributions) == len(feature_names())


def test_top_features_returns_largest_absolute_contributions(
    trained_model_path: Path,
) -> None:
    """top_features should rank by |shap_value|, descending."""
    explainer = ShapExplainer(model_path=trained_model_path)
    e = explainer.explain([_high_risk_patient()])[0]
    top5 = e.top_features(5)
    assert len(top5) == 5
    # Each is at least as large in magnitude as the next
    magnitudes = [abs(c.shap_value) for c in top5]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_high_risk_patient_has_clinically_sensible_top_features(
    trained_model_path: Path,
) -> None:
    """
    For an elderly HF patient with a long stay, the top contributors should
    include some combination of: dx_I50.9, age, length_of_stay_days, age_bucket_70_plus.
    This is a 'clinical sanity' check — if the model attributes risk to weird
    features instead, we want to know.
    """
    explainer = ShapExplainer(model_path=trained_model_path)
    e = explainer.explain([_high_risk_patient()])[0]
    top_names = {c.feature_name for c in e.top_features(5)}
    clinical_features = {
        "dx_I50.9",
        "age",
        "length_of_stay_days",
        "age_bucket_70_plus",
    }
    assert (
        top_names & clinical_features
    ), f"Top features {top_names} did not intersect clinical signal set {clinical_features}"


def test_high_risk_has_higher_prediction_than_low_risk(
    trained_model_path: Path,
) -> None:
    """Sanity: the high-risk patient's predicted probability is higher than low-risk."""
    explainer = ShapExplainer(model_path=trained_model_path)
    explanations = explainer.explain([_high_risk_patient(), _low_risk_patient()])
    assert explanations[0].predicted_probability > explanations[1].predicted_probability


def test_explainer_raises_on_missing_model() -> None:
    """A nonexistent model file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ShapExplainer(model_path="/nonexistent/path/model.joblib")
