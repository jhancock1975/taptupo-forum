"""Hypothesis property tests for the User model."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.models.user import AgentConfig, User

pytestmark = pytest.mark.hypothesis


usernames = st.from_regex(r"\A[A-Za-z0-9_]{3,32}\Z", fullmatch=True)
non_empty_text = st.text(min_size=1, max_size=50)

agent_configs = st.builds(
    AgentConfig,
    model_id=non_empty_text,
    persona_name=non_empty_text,
    expertise_areas=st.lists(non_empty_text, max_size=4),
    personality_traits=st.lists(non_empty_text, max_size=4),
    response_probability=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    system_prompt=non_empty_text,
)


@st.composite
def users(draw: st.DrawFn) -> User:
    username = draw(usernames)
    is_agent = draw(st.booleans())
    if is_agent:
        return User(
            username=username,
            is_agent=True,
            agent_config=draw(agent_configs),
        )
    return User(
        username=username,
        is_agent=False,
        password_hash=draw(non_empty_text),
    )


@given(users())
def test_user_round_trip(user: User) -> None:
    assert User(**user.model_dump()) == user


@given(agent_configs)
def test_agent_config_round_trip(cfg: AgentConfig) -> None:
    assert AgentConfig(**cfg.model_dump()) == cfg
