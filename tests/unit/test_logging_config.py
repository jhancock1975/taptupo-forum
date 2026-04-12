"""Unit tests for app.logging_config."""

from __future__ import annotations

import io
import json
import logging

import pytest
import structlog

from app.logging_config import bind_correlation_id, configure_logging


@pytest.mark.unit
def test_configure_logging_emits_json_with_correlation_id() -> None:
    buf = io.StringIO()
    configure_logging(stream=buf, level=logging.INFO)
    bind_correlation_id("corr-123")

    log = structlog.get_logger("test")
    log.info("hello", extra_field="value")

    line = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "hello"
    assert payload["correlation_id"] == "corr-123"
    assert payload["extra_field"] == "value"
    assert payload["level"] == "info"


@pytest.mark.unit
def test_configure_logging_omits_correlation_id_when_unbound() -> None:
    buf = io.StringIO()
    configure_logging(stream=buf, level=logging.INFO)
    # Reset any previously-bound id.
    structlog.contextvars.clear_contextvars()

    structlog.get_logger("test").info("bare")

    payload = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert payload["event"] == "bare"
    assert "correlation_id" not in payload
