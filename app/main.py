"""FastAPI application entrypoint.

Wires the correlation-ID middleware, configures structured logging at
startup, and exposes a minimal ``/health`` route. Routes for auth,
threads, posts, and agents are registered in later phases.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.auth.routes import create_auth_router
from app.config import Settings
from app.db.factory import get_repository
from app.logging_config import configure_logging
from app.middleware import CorrelationIdMiddleware

configure_logging()

_settings = Settings()
_repo = get_repository()

app = FastAPI(title="taptupo-forum")
app.add_middleware(CorrelationIdMiddleware)
app.include_router(create_auth_router(repo=_repo, settings=_settings))


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight liveness probe."""
    return {"status": "ok"}
