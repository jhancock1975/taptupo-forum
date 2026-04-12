"""Unit tests for app.middleware.CorrelationIdMiddleware."""

from __future__ import annotations

import re

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import CORRELATION_HEADER, CorrelationIdMiddleware

_UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/ping")
    def ping() -> dict[str, str | None]:
        ctx = structlog.contextvars.get_contextvars()
        return {"correlation_id": ctx.get("correlation_id")}

    return app


@pytest.mark.unit
def test_middleware_generates_correlation_id_when_absent() -> None:
    client = TestClient(_app())
    resp = client.get("/ping")
    assert resp.status_code == 200
    cid = resp.headers[CORRELATION_HEADER]
    assert _UUID4.match(cid)
    assert resp.json()["correlation_id"] == cid


@pytest.mark.unit
def test_middleware_reuses_incoming_correlation_id() -> None:
    client = TestClient(_app())
    resp = client.get("/ping", headers={CORRELATION_HEADER: "caller-supplied-id"})
    assert resp.headers[CORRELATION_HEADER] == "caller-supplied-id"
    assert resp.json()["correlation_id"] == "caller-supplied-id"


@pytest.mark.unit
def test_middleware_clears_contextvars_between_requests() -> None:
    client = TestClient(_app())
    r1 = client.get("/ping")
    r2 = client.get("/ping")
    id1 = r1.headers[CORRELATION_HEADER]
    id2 = r2.headers[CORRELATION_HEADER]
    assert id1 != id2
