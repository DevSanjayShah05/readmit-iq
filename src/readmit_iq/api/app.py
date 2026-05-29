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
    BatchPredictionResponse,
    BatchPredictRequest,
    ExplanationResponse,
    FeatureContributionResponse,
    PredictionResponse,
    probability_to_risk_band,
)
from readmit_iq.config import get_settings
from readmit_iq.ml.explain import ShapExplainer
from readmit_iq.ml.predict import ReadmissionPredictor

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
            # Don't crash the whole server — let /health report the failure
            # so ops sees it, but other endpoints can still serve.
            logger.error(f"Model loading failed: {exc}")
            app.state.predictor = None
            app.state.explainer = None
            app.state.model_loaded = False

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

    logger.info(f"FastAPI app initialized (env={settings.app_env})")
    return app


# Module-level `app` for uvicorn to pick up.
app = create_app()
