"""
Tests for the ReadmitIQ API health and root endpoints.

Uses async httpx + asgi-lifespan so FastAPI's startup hooks run during tests.
"""
from __future__ import annotations

from datetime import datetime

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from readmit_iq.api.app import create_app


@pytest.fixture
def app() -> FastAPI:
    """A fresh FastAPI app for each test."""
    return create_app()


@pytest.fixture
async def client(app: FastAPI):
    """Async httpx client with lifespan support."""
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_health_returns_ok(client: httpx.AsyncClient) -> None:
    """GET /health should return 200 OK with the expected fields."""
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "app_env" in body
    assert "timestamp" in body
    datetime.fromisoformat(body["timestamp"])


@pytest.mark.anyio
async def test_root_endpoint_returns_landing(client: httpx.AsyncClient) -> None:
    """GET / should return a small JSON pointing at the docs."""
    response = await client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "ReadmitIQ"
    assert body["docs"] == "/docs"


@pytest.mark.anyio
async def test_docs_endpoint_renders(client: httpx.AsyncClient) -> None:
    """GET /docs should serve the Swagger UI HTML (200, content-type html)."""
    response = await client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.anyio
async def test_openapi_schema_is_valid_json(client: httpx.AsyncClient) -> None:
    """GET /openapi.json should return the spec describing this API."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "ReadmitIQ API"
    assert "/health" in schema["paths"]


@pytest.mark.anyio
async def test_unknown_route_returns_404(client: httpx.AsyncClient) -> None:
    """A nonexistent route should return 404."""
    response = await client.get("/this-does-not-exist")
    assert response.status_code == 404
