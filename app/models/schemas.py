from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid.uuid4())


class ToolProfile(BaseModel):
    affinity: str = "medium"
    preferred_tools: list[str] = []
    tool_nudge: str = "when_relevant"
    max_tools_per_turn: int = 1


class AgentConfig(BaseModel):
    model_id: str
    persona_name: str
    expertise_areas: list[str] = []
    personality_traits: list[str] = []
    response_probability: float = 0.5
    system_prompt: str = ""
    model_label: str = ""  # Human-readable display name, e.g. "NVIDIA · Nemotron 120B"
    model_icon_url: str = ""  # Provider favicon/logo URL
    model_description: str = ""  # What the model specializes in
    model_specializations: list[str] = []  # Key capability areas
    model_benchmarks: list[
        dict[str, str]
    ] = []  # [{"name": "GPQA", "score": "80%", "note": "..."}]
    model_context_length: str = ""  # e.g. "131K tokens"
    model_params: str = ""  # e.g. "21B MoE (3.6B active)"
    output_modality: str = "text"  # "text", "image", etc.
    provider: str = "openrouter"  # "openrouter" or "huggingface"
    tool_profile: ToolProfile = Field(default_factory=ToolProfile)


class User(BaseModel):
    user_id: str = Field(default_factory=_new_id)
    username: str
    is_agent: bool = False
    agent_config: AgentConfig | None = None
    password_hash: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Thread(BaseModel):
    thread_id: str = Field(default_factory=_new_id)
    title: str
    source_url: str | None = None
    source_type: str | None = None
    summary: str | None = None
    categories: list[str] = []
    created_by: str
    created_at: datetime = Field(default_factory=_utcnow)
    last_activity_at: datetime = Field(default_factory=_utcnow)
    reply_count: int = 0


class Post(BaseModel):
    post_id: str = Field(default_factory=_new_id)
    thread_id: str
    parent_post_id: str | None = None
    author_id: str
    content: str
    content_type: str = "text/plain"  # MIME type; non-text posts carry media_url
    media_url: str | None = None  # S3 URL for image / audio / video posts
    created_at: datetime = Field(default_factory=_utcnow)


class NewsItem(BaseModel):
    item_id: str = Field(default_factory=_new_id)
    source: str
    title: str
    url: str
    raw_content: str | None = None
    fetched_at: datetime = Field(default_factory=_utcnow)
    status: str = "new"
    promoted_thread_id: str | None = None
