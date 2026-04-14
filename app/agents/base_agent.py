"""BaseAgent: persona + expertise + LLM-backed response logic.

Decision pipeline for ``decide_to_respond``:

1. Cheap expertise screen (case-insensitive substring match on keywords).
2. Configured response probability (seeded RNG - skip the rest when the
   dice say no, to keep costs predictable).
3. LLM-based relevance check (we ask the model "YES or NO: is this post
   worth engaging with?").

The LLM is passed in by constructor so tests can inject a fake. Real
deployments build it from OpenRouter via ``app.agents.registry``.
"""

from __future__ import annotations

import random
from typing import Protocol

from app.models import User

_RELEVANCE_PROMPT = (
    "You are {persona}. You post on a forum about {areas}. "
    "A user posted: {post!r}\n"
    "Would you want to reply? Answer with a single word: YES or NO."
)
_RESPONSE_PROMPT = (
    "You are {persona}. Personality: {traits}. "
    "Reply to this forum post in 1-3 sentences.\n\nPost: {post}"
)


class _LLMResponse(Protocol):
    content: str


class _LLM(Protocol):
    async def ainvoke(self, prompt: str, /) -> _LLMResponse: ...


def expertise_matches(text: str, areas: list[str]) -> bool:
    """Return True if any expertise area appears in ``text`` (case-insensitive)."""
    haystack = text.lower()
    for area in areas:
        needle = area.strip().lower()
        if needle and needle in haystack:
            return True
    return False


class BaseAgent:
    """Forum agent backed by an LLM."""

    def __init__(
        self,
        *,
        user: User,
        llm: _LLM,
        rng: random.Random | None = None,
    ) -> None:
        if not user.is_agent or user.agent_config is None:
            raise ValueError("BaseAgent requires a user with is_agent=True and agent_config set")
        self.user = user
        self.config = user.agent_config
        self._llm = llm
        self._rng = rng or random.Random()  # noqa: S311  # nosec B311 — non-crypto use

    async def llm_says_relevant(self, post_text: str) -> bool:
        prompt = _RELEVANCE_PROMPT.format(
            persona=self.config.persona_name,
            areas=", ".join(self.config.expertise_areas),
            post=post_text,
        )
        resp = await self._llm.ainvoke(prompt)
        content = (resp.content or "").strip().upper()
        return content.startswith("YES")

    async def decide_to_respond(self, post_text: str) -> bool:
        if not expertise_matches(post_text, self.config.expertise_areas):
            return False
        if self._rng.random() > self.config.response_probability:
            return False
        return await self.llm_says_relevant(post_text)

    async def generate_response(self, post_text: str) -> str:
        prompt = _RESPONSE_PROMPT.format(
            persona=self.config.persona_name,
            traits=", ".join(self.config.personality_traits),
            post=post_text,
        )
        resp = await self._llm.ainvoke(prompt)
        return (resp.content or "").strip()
