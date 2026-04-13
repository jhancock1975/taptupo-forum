"""Unit tests for the Post model."""

from __future__ import annotations

import re
import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models.post import Post

pytestmark = pytest.mark.unit

UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def test_post_constructs_with_defaults() -> None:
    post = Post(thread_id="t1", author_id="u1", content="hello")
    assert UUID4_RE.match(post.post_id)
    uuid.UUID(post.post_id, version=4)
    assert post.thread_id == "t1"
    assert post.parent_post_id is None
    assert post.author_id == "u1"
    assert post.content == "hello"
    assert isinstance(post.created_at, datetime)
    assert post.created_at.tzinfo is not None


def test_post_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Post(
            thread_id="t",
            author_id="u",
            content="c",
            surprise=1,  # type: ignore[call-arg]
        )


def test_post_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        Post(thread_id="t", author_id="u", content="")


def test_post_rejects_overlong_content() -> None:
    with pytest.raises(ValidationError):
        Post(thread_id="t", author_id="u", content="x" * 10001)
