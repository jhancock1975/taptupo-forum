"""FastAPI application entrypoint.

Wires the correlation-ID middleware, configures structured logging at
startup, and exposes a minimal ``/health`` route. Routes for auth,
threads, posts, and agents are registered in later phases.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.auth.routes import create_auth_router
from app.config import Settings
from app.db.factory import get_repository
from app.forum.routes import create_forum_router
from app.logging_config import configure_logging
from app.middleware import CorrelationIdMiddleware
from app.pages.routes import create_pages_router
from app.realtime.broker import Broker
from app.realtime.websocket import create_websocket_router

configure_logging()

_settings = Settings()
_repo = get_repository()
_broker = Broker()

_STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="taptupo-forum")
app.add_middleware(CorrelationIdMiddleware)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
app.include_router(create_auth_router(repo=_repo, settings=_settings))
app.include_router(create_forum_router(repo=_repo, settings=_settings, broker=_broker))
app.include_router(create_pages_router(repo=_repo, settings=_settings))
app.include_router(create_websocket_router(broker=_broker))


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight liveness probe."""
    return {"status": "ok"}
