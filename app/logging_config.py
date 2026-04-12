"""Structured JSON logging with correlation-ID propagation.

Every log record emitted through ``structlog`` carries whatever keys
are currently bound in :mod:`structlog.contextvars`. The FastAPI
middleware binds ``correlation_id`` at the start of each request so
logs produced deeper in the stack can be tied back to the originating
request or async task.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

import structlog


def configure_logging(
    *, stream: TextIO | None = None, level: int = logging.INFO
) -> None:
    """Configure the stdlib ``logging`` and ``structlog`` stacks to emit JSON.

    Parameters
    ----------
    stream:
        Where to write log records. Defaults to ``sys.stdout``. Tests
        pass an ``io.StringIO`` to capture output.
    level:
        Standard-library log level. Defaults to :data:`logging.INFO`.
    """
    target = stream if stream is not None else sys.stdout

    # Route stdlib records through structlog's processor chain.
    handler = logging.StreamHandler(target)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=target),
        cache_logger_on_first_use=False,
    )


def bind_correlation_id(correlation_id: str) -> None:
    """Bind ``correlation_id`` into the structlog context for this task."""
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def clear_correlation_id() -> None:
    """Clear any previously-bound ``correlation_id``."""
    structlog.contextvars.unbind_contextvars("correlation_id")
