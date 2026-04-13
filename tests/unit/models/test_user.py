"""Unit tests for the User model."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.models.user import AgentConfig, User

pytestmark = pytest.mark.unit

UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _agent_config() -> AgentConfig:
    return AgentConfig(
        model_id="openrouter/some-model",
        persona_name="Ada",
        expertise_areas=["math"],
        personality_traits=["curious"],
        response_probability=0.5,
        system_prompt="You are Ada.",
    )


def test_user_constructs_with_defaults() -> None:
    user = User(username="alice_99", password_hash="hashed")
    assert isinstance(user.user_id, str)
    assert UUID4_RE.match(user.user_id)
    # verify parseable as UUID
    uuid.UUID(user.user_id, version=4)
    assert user.username == "alice_99"
    assert user.is_agent is False
    assert user.password_hash == "hashed"  # pragma: allowlist secret
    assert user.agent_config is None
    assert isinstance(user.created_at, datetime)
    assert user.created_at.tzinfo is not None
    assert user.created_at.utcoffset() == UTC.utcoffset(user.created_at)


def test_user_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        User(username="alice", password_hash="h", nope="x")  # type: ignore[call-arg]


def test_user_rejects_invalid_username_pattern() -> None:
    with pytest.raises(ValidationError):
        User(username="no spaces!", password_hash="h")


def test_user_rejects_username_too_short() -> None:
    with pytest.raises(ValidationError):
        User(username="ab", password_hash="h")


def test_user_rejects_username_too_long() -> None:
    with pytest.raises(ValidationError):
        User(username="a" * 33, password_hash="h")


def test_user_agent_requires_agent_config() -> None:
    with pytest.raises(ValidationError):
        User(username="agent_a", is_agent=True, agent_config=None)


def test_user_agent_forbids_password_hash() -> None:
    with pytest.raises(ValidationError):
        User(
            username="agent_a",
            is_agent=True,
            agent_config=_agent_config(),
            password_hash="h",
        )


def test_user_human_requires_password_hash() -> None:
    with pytest.raises(ValidationError):
        User(username="alice", is_agent=False, password_hash=None)


def test_user_human_forbids_agent_config() -> None:
    with pytest.raises(ValidationError):
        User(
            username="alice",
            is_agent=False,
            password_hash="h",
            agent_config=_agent_config(),
        )


def test_agent_config_rejects_bad_response_probability() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(
            model_id="m",
            persona_name="p",
            expertise_areas=[],
            personality_traits=[],
            response_probability=1.5,
            system_prompt="s",
        )


def test_agent_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(
            model_id="m",
            persona_name="p",
            expertise_areas=[],
            personality_traits=[],
            response_probability=0.5,
            system_prompt="s",
            bogus=1,  # type: ignore[call-arg]
        )
