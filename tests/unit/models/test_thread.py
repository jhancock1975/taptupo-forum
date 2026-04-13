"""Unit tests for the Thread model."""

from __future__ import annotations

import re
import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models.thread import Thread

pytestmark = pytest.mark.unit

UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def test_thread_constructs_with_defaults() -> None:
    thread = Thread(
        title="Hello world",
        source_type="human",
        created_by=str(uuid.uuid4()),
    )
    assert UUID4_RE.match(thread.thread_id)
    uuid.UUID(thread.thread_id, version=4)
    assert thread.title == "Hello world"
    assert thread.source_url is None
    assert thread.summary == ""
    assert thread.categories == []
    assert isinstance(thread.created_at, datetime)
    assert thread.created_at.tzinfo is not None
    assert isinstance(thread.last_activity_at, datetime)
    assert thread.last_activity_at.tzinfo is not None


def test_thread_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Thread(
            title="T",
            source_type="human",
            created_by="u",
            extra="no",  # type: ignore[call-arg]
        )


def test_thread_rejects_empty_title() -> None:
    with pytest.raises(ValidationError):
        Thread(title="", source_type="human", created_by="u")


def test_thread_rejects_overlong_title() -> None:
    with pytest.raises(ValidationError):
        Thread(title="x" * 301, source_type="human", created_by="u")


def test_thread_rejects_invalid_source_type() -> None:
    with pytest.raises(ValidationError):
        Thread(title="T", source_type="facebook", created_by="u")  # type: ignore[arg-type]
