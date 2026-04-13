"""FastAPI application entrypoint.

Wires the correlation-ID middleware, configures structured logging at
startup, and exposes a minimal ``/health`` route. Routes for auth,
threads, posts, and agents are registered in later phases.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.logging_config import configure_logging
from app.middleware import CorrelationIdMiddleware

configure_logging()

app = FastAPI(title="taptupo-forum")
app.add_middleware(CorrelationIdMiddleware)


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight liveness probe."""
    return {"status": "ok"}
