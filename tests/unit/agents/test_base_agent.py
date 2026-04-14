"""Unit tests for app.agents.base_agent."""

from __future__ import annotations

import pytest

from app.agents.base_agent import BaseAgent, expertise_matches
from app.models import AgentConfig, User


def _config(*areas: str, prob: float = 0.5) -> AgentConfig:
    return AgentConfig(
        model_id="openrouter/free",
        persona_name="Persona",
        expertise_areas=list(areas),
        personality_traits=["curious"],
        response_probability=prob,
        system_prompt="You are Persona.",
    )


def _agent_user(config: AgentConfig) -> User:
    return User(username="bot_one", is_agent=True, agent_config=config)


@pytest.mark.unit
def test_expertise_matches_is_case_insensitive() -> None:
    assert expertise_matches("Talking about ML today", ["machine learning", "ml"])
    assert expertise_matches("Quantum physics rules", ["QUANTUM"])


@pytest.mark.unit
def test_expertise_matches_false_when_none_appear() -> None:
    assert not expertise_matches("a post about cats", ["quantum", "ml"])


@pytest.mark.unit
def test_expertise_matches_empty_areas_is_false() -> None:
    assert not expertise_matches("anything", [])


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []

    async def ainvoke(self, prompt: str) -> object:
        self.calls.append(prompt)

        class _R:
            content = self.reply

        return _R()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relevance_check_parses_yes_no() -> None:
    agent = BaseAgent(user=_agent_user(_config("ml")), llm=_FakeLLM("YES"))
    assert await agent.llm_says_relevant("about ml") is True
    agent2 = BaseAgent(user=_agent_user(_config("ml")), llm=_FakeLLM("definitely no, not relevant"))
    assert await agent2.llm_says_relevant("something") is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_decide_to_respond_requires_expertise_and_relevance() -> None:
    cfg = _config("ml", prob=1.0)
    agent = BaseAgent(user=_agent_user(cfg), llm=_FakeLLM("YES"))
    assert await agent.decide_to_respond("Deep dive into ML today") is True

    # Expertise miss → skip LLM call, return False.
    llm = _FakeLLM("YES")
    agent2 = BaseAgent(user=_agent_user(cfg), llm=llm)
    assert await agent2.decide_to_respond("Cooking recipes") is False
    assert llm.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_decide_respects_response_probability_zero() -> None:
    cfg = _config("ml", prob=0.0)
    agent = BaseAgent(user=_agent_user(cfg), llm=_FakeLLM("YES"))
    assert await agent.decide_to_respond("ML post") is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_response_returns_stripped_llm_content() -> None:
    agent = BaseAgent(user=_agent_user(_config("ml")), llm=_FakeLLM("  Hi there  "))
    text = await agent.generate_response("Some post about ML")
    assert text == "Hi there"
