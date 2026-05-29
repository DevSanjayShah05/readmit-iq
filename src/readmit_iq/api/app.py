"""
FastAPI application for ReadmitIQ.

Hosts the prediction and explanation endpoints. The actual ML logic lives
in readmit_iq.ml; this module is just the HTTP boundary — it parses
requests, calls the right service, serializes responses.

Run locally:
    uvicorn readmit_iq.api.app:app --reload --port 8000

Interactive docs available at:
    http://localhost:8000/docs
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field

from readmit_iq.config import get_settings


class HealthResponse(BaseModel):
    """Shape returned by GET /health."""

    status: str = Field(..., description="'ok' if the service is healthy")
    app_env: str = Field(
        ..., description="Which environment this service is running in"
    )
    timestamp: datetime = Field(
        ..., description="Server time when the response was generated"
    )


def create_app() -> FastAPI:
    """
    Build and return a FastAPI application.

    Using a factory function (rather than a module-level `app = FastAPI()`)
    makes testing easier — tests can construct a fresh app, possibly with
    test-specific configuration, without import-time side effects.
    """
    settings = get_settings()
    app = FastAPI(
        title="ReadmitIQ API",
        description="30-day readmission risk prediction with explanations.",
        version="0.1.0",
    )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Liveness check used by load balancers and uptime monitors."""
        return HealthResponse(
            status="ok",
            app_env=settings.app_env,
            timestamp=datetime.now(timezone.utc),
        )

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        """Landing page — points users at the docs."""
        return {
            "service": "ReadmitIQ",
            "docs": "/docs",
            "health": "/health",
        }

    logger.info(f"FastAPI app initialized (env={settings.app_env})")
    return app


# Module-level `app` for uvicorn to pick up.
# `uvicorn readmit_iq.api.app:app` looks for this name.
app = create_app()
