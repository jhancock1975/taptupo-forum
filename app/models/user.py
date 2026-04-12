"""User and AgentConfig Pydantic models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class AgentConfig(BaseModel):
    """Configuration for an LLM-backed agent user."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    model_id: str
    persona_name: str
    expertise_areas: list[str]
    personality_traits: list[str]
    response_probability: float = Field(ge=0.0, le=1.0)
    system_prompt: str


Username = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_]{3,32}$")]


class User(BaseModel):
    """Forum user - either a human or an agent."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: Username
    is_agent: bool = False
    password_hash: str | None = None
    agent_config: AgentConfig | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_agent_vs_human(self) -> User:
        if self.is_agent:
            if self.agent_config is None:
                raise ValueError("agent users require agent_config")
            if self.password_hash is not None:
                raise ValueError("agent users must not have password_hash")
        else:
            if self.password_hash is None:
                raise ValueError("human users require password_hash")
            if self.agent_config is not None:
                raise ValueError("human users must not have agent_config")
        return self
