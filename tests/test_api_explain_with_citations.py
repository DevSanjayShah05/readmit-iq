"""
Tests for the /explain-with-citations endpoint.

We reuse the same tiny-model setup as test_api_predict, but we mock the
retriever on app.state so tests don't depend on Qdrant being up. This
verifies the route's logic (composing queries, assembling responses,
graceful degradation) independent of the vector store.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from readmit_iq.api.app import create_app
from readmit_iq.dao.patient_dao import PatientDAO
from readmit_iq.ml.train import train_model
from readmit_iq.rag.retriever import RetrievedDocument
from readmit_iq.scripts.seed_data import seed_patients


# ---------- Fixtures (mirror test_api_predict structure) ----------


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory) -> Path:
    """Seed data once per module and train a small model. Reused across tests."""
    seed_patients(count=400, seed=42, truncate=True)
    patients = PatientDAO().find_admissions_between(
        date(1900, 1, 1), date(2099, 12, 31)
    )
    path = tmp_path_factory.mktemp("citations_test") / "model.joblib"
    train_model(patients=patients, output_path=path, n_estimators=30)
    return path


@pytest.fixture
def app(trained_model_path: Path) -> FastAPI:
    return create_app(model_path=trained_model_path)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _sample_docs() -> list[RetrievedDocument]:
    """Canned retrieval results the mock returns."""
    return [
        RetrievedDocument(
            pmid="40903382",
            title="30-Day Unplanned Readmissions After Heart Failure",
            abstract="Body of the heart-failure readmission paper.",
            journal="Heart, lung & circulation",
            year="2025",
            authors=["Smith J", "Doe A"],
            score=0.76,
        ),
        RetrievedDocument(
            pmid="41891134",
            title="Heart failure readmissions in urban and rural hospitals",
            abstract="Body of the urban/rural paper.",
            journal="The Journal of rural health",
            year="2026",
            authors=["Jones B"],
            score=0.69,
        ),
    ]


@pytest.fixture
async def client_with_mock_retriever(app: FastAPI):
    """
    Async httpx client with a mocked retriever installed on app.state.

    LifespanManager runs the real startup (which tries to load the actual
    Retriever and may fail in CI where Qdrant isn't running). We replace
    app.state.retriever immediately after with our mock — the route reads
    from app.state at request time, so this works regardless of whether
    the real retriever loaded.
    """
    async with LifespanManager(app):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _sample_docs()
        app.state.retriever = mock_retriever

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, mock_retriever


@pytest.fixture
async def client_without_retriever(app: FastAPI):
    """Same as above but with retriever=None to test graceful degradation."""
    async with LifespanManager(app):
        app.state.retriever = None

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


def _sample_patient_body(
    mrn: str = "TEST-001", age: int = 78, dx: str = "I50.9"
) -> dict:
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


# ---------- Tests ----------


@pytest.mark.anyio
async def test_explain_with_citations_returns_full_shape(
    client_with_mock_retriever,
) -> None:
    """The endpoint returns prediction, SHAP, query, and citations."""
    client, _ = client_with_mock_retriever
    response = await client.post(
        "/explain-with-citations", json=_sample_patient_body()
    )
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert len(body["results"]) == 1

    result = body["results"][0]
    assert result["mrn"] == "TEST-001"
    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert result["risk_band"] in ("low", "medium", "high")
    assert len(result["contributions"]) == 19
    assert "query" in result
    assert len(result["citations"]) == 2


@pytest.mark.anyio
async def test_explain_with_citations_composes_clinical_query(
    client_with_mock_retriever,
) -> None:
    """The composed query should read like a clinician's question."""
    client, mock_retriever = client_with_mock_retriever
    response = await client.post(
        "/explain-with-citations", json=_sample_patient_body()
    )
    body = response.json()
    query = body["results"][0]["query"]

    # The query should reflect the patient's actual situation
    assert "older adult" in query  # age=78
    assert "heart failure" in query  # dx=I50.9
    assert "30-day" in query

    # The retriever should have been called with that exact query
    mock_retriever.retrieve.assert_called_once()
    call_args = mock_retriever.retrieve.call_args
    assert call_args.args[0] == query or call_args.kwargs.get("query") == query


@pytest.mark.anyio
async def test_explain_with_citations_payload_shape(
    client_with_mock_retriever,
) -> None:
    """Each citation should have all required fields, including pubmed_url."""
    client, _ = client_with_mock_retriever
    response = await client.post(
        "/explain-with-citations", json=_sample_patient_body()
    )
    body = response.json()
    citation = body["results"][0]["citations"][0]

    assert citation["pmid"] == "40903382"
    assert "Heart Failure" in citation["title"]
    assert citation["journal"] == "Heart, lung & circulation"
    assert citation["year"] == "2025"
    assert citation["authors"] == ["Smith J", "Doe A"]
    assert citation["score"] == 0.76
    assert citation["pubmed_url"] == "https://pubmed.ncbi.nlm.nih.gov/40903382/"


@pytest.mark.anyio
async def test_explain_with_citations_handles_missing_retriever(
    client_without_retriever,
) -> None:
    """If the retriever is None, predictions still work; citations are empty."""
    response = await client_without_retriever.post(
        "/explain-with-citations", json=_sample_patient_body()
    )
    assert response.status_code == 200
    body = response.json()
    result = body["results"][0]

    # Prediction and SHAP still work
    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert len(result["contributions"]) == 19
    # Query is still composed (it doesn't depend on the retriever)
    assert "30-day" in result["query"]
    # But citations are empty
    assert result["citations"] == []


@pytest.mark.anyio
async def test_explain_with_citations_batch(
    client_with_mock_retriever,
) -> None:
    """Two patients should each get their own composed query and citations."""
    client, mock_retriever = client_with_mock_retriever
    body = {
        "patients": [
            {
                "mrn": "PATIENT-A",
                "age": 85,
                "sex": "F",
                "admission_date": "2024-06-15",
                "discharge_date": "2024-07-02",
                "primary_diagnosis": "A41.9",  # sepsis, 17-day stay
            },
            {
                "mrn": "PATIENT-B",
                "age": 45,
                "sex": "M",
                "admission_date": "2024-06-15",
                "discharge_date": "2024-06-16",
                "primary_diagnosis": "J44.9",  # COPD, 1-day stay
            },
        ]
    }
    response = await client.post("/explain-with-citations", json=body)
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["mrn"] == "PATIENT-A"
    assert "very elderly" in results[0]["query"]
    assert "sepsis" in results[0]["query"]
    assert results[1]["mrn"] == "PATIENT-B"
    assert "middle-aged" in results[1]["query"]
    assert "COPD" in results[1]["query"]
    # Each patient triggers a separate retrieval call
    assert mock_retriever.retrieve.call_count == 2
