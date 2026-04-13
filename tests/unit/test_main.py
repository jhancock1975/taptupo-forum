"""Unit tests for the FastAPI app wired in app.main."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.middleware import CORRELATION_HEADER


@pytest.mark.unit
def test_health_returns_ok() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.unit
def test_health_carries_correlation_id_header() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert CORRELATION_HEADER in resp.headers
