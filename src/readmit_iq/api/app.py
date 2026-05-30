"""
FastAPI application for ReadmitIQ.

Hosts the prediction, explanation, and meta endpoints. The actual ML
logic lives in readmit_iq.ml; this module is just the HTTP boundary —
it parses requests, calls the right service, serializes responses.

The model and explainer are loaded once at startup (via the lifespan
context manager) and reused across requests. Loading takes ~1 second;
doing it per-request would make every API call unacceptably slow.

Run locally:
    uvicorn readmit_iq.api.app:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from readmit_iq.api.schemas import (
    BatchExplanationResponse,
    BatchExplanationWithCitationsResponse,
    BatchPredictionResponse,
    BatchPredictRequest,
    ExplanationResponse,
    ExplanationWithCitationsResponse,
    FeatureContributionResponse,
    PredictionResponse,
    RetrievedCitationResponse,
    probability_to_risk_band,
)
from readmit_iq.config import get_settings
from readmit_iq.ml.explain import ShapExplainer
from readmit_iq.ml.predict import ReadmissionPredictor
from readmit_iq.rag.query_composer import compose_query
from readmit_iq.rag.retriever import Retriever

# Default model location. Can be overridden by the MODEL_PATH env var if needed
# (for example, when running tests with a temp-trained model).
_DEFAULT_MODEL_PATH = Path("models/readmit_rf.joblib")


class HealthResponse(BaseModel):
    """Shape returned by GET /health."""

    status: str = Field(..., description="'ok' if the service is healthy")
    app_env: str = Field(
        ..., description="Which environment this service is running in"
    )
    timestamp: datetime = Field(
        ..., description="Server time when the response was generated"
    )
    model_loaded: bool = Field(
        ..., description="Whether the ML model loaded successfully at startup"
    )


def _build_lifespan(model_path: Path):
    """
    Build a lifespan context manager that loads the model at startup.

    Wrapping this in a builder lets create_app() inject a custom model
    path — useful for tests that train a tiny model in tmp_path and want
    the API to use it.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # --- Startup ---
        logger.info(f"Loading model from {model_path}")
        try:
            app.state.predictor = ReadmissionPredictor(model_path=model_path)
            app.state.explainer = ShapExplainer(model_path=model_path)
            app.state.model_loaded = True
            logger.success("Model and SHAP explainer ready")
        except FileNotFoundError as exc:
            logger.error(f"Model loading failed: {exc}")
            app.state.predictor = None
            app.state.explainer = None
            app.state.model_loaded = False

        # Retriever uses Qdrant; failing gracefully so /predict and /explain
        # still work even if the vector store is down.
        try:
            app.state.retriever = Retriever()
            app.state.retriever_loaded = True
            logger.success("RAG retriever ready")
        except Exception as exc:
            logger.error(f"Retriever loading failed: {exc}")
            app.state.retriever = None
            app.state.retriever_loaded = False

        yield  # the app runs here

        # --- Shutdown ---
        logger.info("API shutting down")

    return lifespan


def create_app(model_path: Path | None = None) -> FastAPI:
    """
    Build and return a FastAPI application.

    Args:
        model_path: optional override for the model location. Defaults to
            models/readmit_rf.joblib. Tests can pass in a temp-trained model.
    """
    settings = get_settings()
    effective_path = model_path or _DEFAULT_MODEL_PATH

    app = FastAPI(
        title="ReadmitIQ API",
        description="30-day readmission risk prediction with explanations.",
        version="0.1.0",
        lifespan=_build_lifespan(effective_path),
    )

    # ---------- Meta endpoints ----------

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(request: Request) -> HealthResponse:
        """Liveness check. Reports whether the model loaded successfully."""
        return HealthResponse(
            status="ok",
            app_env=settings.app_env,
            timestamp=datetime.now(timezone.utc),
            model_loaded=bool(getattr(request.app.state, "model_loaded", False)),
        )

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        """Landing page — points users at the docs."""
        return {
            "service": "ReadmitIQ",
            "docs": "/docs",
            "health": "/health",
        }

    # ---------- Prediction endpoints ----------

    def _require_predictor(request: Request) -> ReadmissionPredictor:
        """Look up the predictor, or 503 if startup failed to load it."""
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded; check server startup logs.",
            )
        return predictor

    def _require_explainer(request: Request) -> ShapExplainer:
        """Look up the explainer, or 503 if startup failed to load it."""
        explainer = getattr(request.app.state, "explainer", None)
        if explainer is None:
            raise HTTPException(
                status_code=503,
                detail="Explainer not loaded; check server startup logs.",
            )
        return explainer

    @app.post(
        "/predict",
        response_model=BatchPredictionResponse,
        tags=["predictions"],
    )
    def predict(request: Request, body: BatchPredictRequest) -> BatchPredictionResponse:
        """
        Score one or more patients for 30-day readmission risk.

        Accepts up to 1000 patients per call. Returns a probability and a
        coarse risk band ('low'|'medium'|'high') per patient.
        """
        predictor = _require_predictor(request)
        domain_patients = [p.to_domain() for p in body.patients]
        probabilities = predictor.predict_proba(domain_patients)

        predictions = [
            PredictionResponse(
                mrn=p.mrn,
                readmission_probability=float(prob),
                risk_band=probability_to_risk_band(float(prob)),
            )
            for p, prob in zip(body.patients, probabilities)
        ]
        logger.info(
            f"Scored {len(predictions)} patients "
            f"(mean prob: {sum(p.readmission_probability for p in predictions) / len(predictions):.3f})"
        )
        return BatchPredictionResponse(predictions=predictions)

    @app.post(
        "/explain",
        response_model=BatchExplanationResponse,
        tags=["predictions"],
    )
    def explain(
        request: Request, body: BatchPredictRequest
    ) -> BatchExplanationResponse:
        """
        Return SHAP explanations for one or more patients.

        Each explanation includes the predicted probability, the baseline,
        and per-feature contributions. The contributions sum (within
        floating-point tolerance) to predicted_probability - baseline.
        """
        explainer = _require_explainer(request)
        domain_patients = [p.to_domain() for p in body.patients]
        results = explainer.explain(domain_patients)

        explanations = [
            ExplanationResponse(
                mrn=e.mrn,
                predicted_probability=e.predicted_probability,
                baseline_probability=e.baseline_probability,
                contributions=[
                    FeatureContributionResponse(
                        feature_name=c.feature_name,
                        feature_value=c.feature_value,
                        shap_value=c.shap_value,
                    )
                    for c in e.contributions
                ],
            )
            for e in results
        ]
        logger.info(f"Explained {len(explanations)} predictions")
        return BatchExplanationResponse(explanations=explanations)

    @app.post(
        "/explain-with-citations",
        response_model=BatchExplanationWithCitationsResponse,
        tags=["predictions"],
    )
    def explain_with_citations(
        request: Request, body: BatchPredictRequest
    ) -> BatchExplanationWithCitationsResponse:
        """
        Predict, explain, and retrieve relevant biomedical literature
        for each patient in one call.

        For each patient:
          1. Score with the trained model (probability + risk band)
          2. Compute SHAP feature contributions
          3. Compose a clinical query from age, diagnosis, length of stay
          4. Retrieve top-3 relevant PubMed abstracts from the vector store

        If the retriever is unavailable, citations are returned as empty
        lists but predictions and explanations still work.
        """
        explainer = _require_explainer(request)
        retriever = getattr(request.app.state, "retriever", None)

        domain_patients = [p.to_domain() for p in body.patients]
        explanations = explainer.explain(domain_patients)

        results: list[ExplanationWithCitationsResponse] = []
        for patient_req, explanation in zip(body.patients, explanations):
            query = compose_query(
                age=patient_req.age,
                sex=patient_req.sex,
                admission_date=patient_req.admission_date.isoformat(),
                discharge_date=patient_req.discharge_date.isoformat(),
                primary_diagnosis=patient_req.primary_diagnosis,
            )
            if retriever is not None:
                retrieved = retriever.retrieve(query, top_k=3)
                citations = [
                    RetrievedCitationResponse(
                        pmid=doc.pmid,
                        title=doc.title,
                        journal=doc.journal,
                        year=doc.year,
                        authors=doc.authors,
                        score=doc.score,
                        pubmed_url=doc.pubmed_url,
                    )
                    for doc in retrieved
                ]
            else:
                citations = []

            results.append(
                ExplanationWithCitationsResponse(
                    mrn=explanation.mrn,
                    predicted_probability=explanation.predicted_probability,
                    baseline_probability=explanation.baseline_probability,
                    risk_band=probability_to_risk_band(
                        explanation.predicted_probability
                    ),
                    contributions=[
                        FeatureContributionResponse(
                            feature_name=c.feature_name,
                            feature_value=c.feature_value,
                            shap_value=c.shap_value,
                        )
                        for c in explanation.contributions
                    ],
                    query=query,
                    citations=citations,
                )
            )

        logger.info(
            f"Explained-with-citations: {len(results)} patients, "
            f"retriever={'up' if retriever else 'down'}"
        )
        return BatchExplanationWithCitationsResponse(results=results)

    logger.info(f"FastAPI app initialized (env={settings.app_env})")
    return app


# Module-level `app` for uvicorn to pick up.
app = create_app()