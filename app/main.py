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
from app.forum.routes import create_forum_router
from app.logging_config import configure_logging
from app.middleware import CorrelationIdMiddleware
from app.realtime.broker import Broker
from app.realtime.websocket import create_websocket_router

configure_logging()

_settings = Settings()
_repo = get_repository()
_broker = Broker()

app = FastAPI(title="taptupo-forum")
app.add_middleware(CorrelationIdMiddleware)
app.include_router(create_auth_router(repo=_repo, settings=_settings))
app.include_router(create_forum_router(repo=_repo, settings=_settings, broker=_broker))
app.include_router(create_websocket_router(broker=_broker))


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight liveness probe."""
    return {"status": "ok"}
