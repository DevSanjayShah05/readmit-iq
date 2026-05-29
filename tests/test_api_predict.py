"""
Tests for the /predict and /explain endpoints.

We train a tiny model in a temp directory, configure the app to use it,
and hit the endpoints via async httpx + asgi-lifespan. End-to-end coverage
of the HTTP -> FastAPI -> ML pipeline.

Important: LifespanManager is required so FastAPI's startup event runs
(which is where we load the model into app.state). Without it the
predictor is None and every prediction endpoint returns 503.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from readmit_iq.api.app import create_app
from readmit_iq.dao.patient_dao import PatientDAO
from readmit_iq.ml.train import train_model
from readmit_iq.scripts.seed_data import seed_patients


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory) -> Path:
    """Seed data once per module and train a small model. Reused across tests."""
    seed_patients(count=400, seed=42, truncate=True)
    patients = PatientDAO().find_admissions_between(
        date(1900, 1, 1), date(2099, 12, 31)
    )
    path = tmp_path_factory.mktemp("api_test") / "model.joblib"
    train_model(patients=patients, output_path=path, n_estimators=30)
    return path


@pytest.fixture
def app(trained_model_path: Path) -> FastAPI:
    """A fresh FastAPI app pointed at the tiny test model."""
    return create_app(model_path=trained_model_path)


@pytest.fixture
async def client(app: FastAPI):
    """
    Async httpx client wired to the in-memory app.

    LifespanManager triggers FastAPI's startup events (loading the model
    into app.state) when entering its async context, and shutdown events
    when exiting. Without it, lifespan code never runs in tests.
    """
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _sample_patient_body(mrn: str = "TEST-001", age: int = 78, dx: str = "I50.9") -> dict:
    """Build a request body with one patient."""
    return {
        "patients": [
            {
                "mrn": mrn,
                "age": age,
                "sex": "M",
                "admission_date": "2024-06-15",
                "discharge_date": "2024-06-22",
                "primary_diagnosis": dx,
            }
        ]
    }


@pytest.mark.anyio
async def test_health_reports_model_loaded(client: httpx.AsyncClient) -> None:
    """With a valid model path, /health should report model_loaded=True."""
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True


@pytest.mark.anyio
async def test_predict_returns_probability_and_band(client: httpx.AsyncClient) -> None:
    """A valid /predict request returns a probability and risk band."""
    response = await client.post("/predict", json=_sample_patient_body())
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 1
    pred = body["predictions"][0]
    assert pred["mrn"] == "TEST-001"
    assert 0.0 <= pred["readmission_probability"] <= 1.0
    assert pred["risk_band"] in {"low", "medium", "high"}


@pytest.mark.anyio
async def test_predict_high_risk_patient_scores_higher_than_low(
    client: httpx.AsyncClient,
) -> None:
    """An elderly HF patient should score higher than a young palliative one."""
    body = {
        "patients": [
            {
                "mrn": "HIGH", "age": 85, "sex": "M",
                "admission_date": "2024-06-01",
                "discharge_date": "2024-06-15",
                "primary_diagnosis": "I50.9",
            },
            {
                "mrn": "LOW", "age": 35, "sex": "F",
                "admission_date": "2024-07-01",
                "discharge_date": "2024-07-02",
                "primary_diagnosis": "Z51.5",
            },
        ]
    }
    response = await client.post("/predict", json=body)
    assert response.status_code == 200
    preds = response.json()["predictions"]
    high = next(p for p in preds if p["mrn"] == "HIGH")
    low = next(p for p in preds if p["mrn"] == "LOW")
    assert high["readmission_probability"] > low["readmission_probability"]


@pytest.mark.anyio
async def test_predict_rejects_invalid_age(client: httpx.AsyncClient) -> None:
    """Age out of [0, 120] should fail validation with 422."""
    body = _sample_patient_body()
    body["patients"][0]["age"] = 500
    response = await client.post("/predict", json=body)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_rejects_invalid_sex(client: httpx.AsyncClient) -> None:
    """Sex must be F, M, or O."""
    body = _sample_patient_body()
    body["patients"][0]["sex"] = "X"
    response = await client.post("/predict", json=body)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_rejects_empty_patient_list(client: httpx.AsyncClient) -> None:
    """An empty patients list should fail validation."""
    response = await client.post("/predict", json={"patients": []})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_explain_returns_contributions(client: httpx.AsyncClient) -> None:
    """An /explain request returns per-feature SHAP contributions."""
    response = await client.post("/explain", json=_sample_patient_body())
    assert response.status_code == 200
    body = response.json()
    assert len(body["explanations"]) == 1
    e = body["explanations"][0]
    assert e["mrn"] == "TEST-001"
    assert "predicted_probability" in e
    assert "baseline_probability" in e
    assert len(e["contributions"]) == 19


@pytest.mark.anyio
async def test_explain_shap_values_sum_to_prediction_minus_baseline(
    client: httpx.AsyncClient,
) -> None:
    """SHAP additivity should hold through the API just like it does in-process."""
    response = await client.post("/explain", json=_sample_patient_body())
    body = response.json()
    e = body["explanations"][0]
    contribution_sum = sum(c["shap_value"] for c in e["contributions"])
    expected = e["predicted_probability"] - e["baseline_probability"]
    assert abs(contribution_sum - expected) < 1e-3
