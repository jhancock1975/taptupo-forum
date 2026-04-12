"""FastAPI middleware for per-request correlation IDs.

The middleware:

* Uses any incoming ``X-Correlation-ID`` header as the correlation ID,
  otherwise generates a fresh UUID4.
* Binds the ID into ``structlog.contextvars`` so every log emitted
  during the request (including from background code awaited within
  the request) carries it.
* Echoes the ID back in the response ``X-Correlation-ID`` header so
  clients can correlate their side of the transaction.
* Clears contextvars at the end of the request so IDs don't leak
  across requests sharing a worker.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

CORRELATION_HEADER = "X-Correlation-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Assign or propagate a correlation ID on every HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        incoming = request.headers.get(CORRELATION_HEADER)
        correlation_id = incoming or str(uuid.uuid4())

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()

        response.headers[CORRELATION_HEADER] = correlation_id
        return response
