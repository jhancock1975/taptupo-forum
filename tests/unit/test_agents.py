from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents import base_agent, discussion, registry
from app.agents.base_agent import BaseAgent, _tokenize
from app.agents.discussion import DiscussionEngine
from app.agents.news_agent import NewsAgent
from app.models.schemas import AgentConfig, NewsItem, Post, Thread, User


def agent_config(**overrides: Any) -> AgentConfig:
    values: dict[str, Any] = {
        "model_id": "test/model",
        "persona_name": "Bot",
        "expertise_areas": ["python", "systems"],
        "personality_traits": ["direct"],
        "response_probability": 0.5,
        "system_prompt": "Be useful.",
    }
    values.update(overrides)
    return AgentConfig(**values)


def agent_user(**overrides: Any) -> User:
    values: dict[str, Any] = {
        "username": "Bot",
        "is_agent": True,
        "agent_config": agent_config(),
    }
    values.update(overrides)
    return User(**values)


class AgentRepo:
    def __init__(self) -> None:
        self.thread: Thread | None = None
        self.posts: list[Post] = []
        self.users: dict[str, User] = {}
        self.created_posts: list[Post] = []
        self.activity_updates: list[str] = []
        self.all_threads: list[Thread] = []

    async def get_thread(self, thread_id: str) -> Thread | None:
        return self.thread if self.thread and self.thread.thread_id == thread_id else None

    async def get_posts_by_thread(self, thread_id: str) -> list[Post]:
        return [p for p in self.posts if p.thread_id == thread_id]

    async def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    async def create_post(self, post: Post) -> Post:
        self.created_posts.append(post)
        return post

    async def update_thread_activity(self, thread_id: str) -> None:
        self.activity_updates.append(thread_id)

    async def list_threads(self, limit: int = 50) -> list[Thread]:
        return self.all_threads[:limit]


def test_base_agent_requires_config() -> None:
    with pytest.raises(ValueError, match="has no agent_config"):
        BaseAgent(User(username="plain-user"), AgentRepo())  # type: ignore[arg-type]


def test_base_agent_expertise_and_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = BaseAgent(agent_user(), AgentRepo())  # type: ignore[arg-type]

    assert agent._matches_expertise("Python performance tuning")
    assert not agent._matches_expertise("gardening notes")

    monkeypatch.setattr(base_agent.random, "random", lambda: 0.49)
    assert agent._should_respond()
    monkeypatch.setattr(base_agent.random, "random", lambda: 0.51)
    assert not agent._should_respond()


@pytest.mark.anyio
async def test_call_llm_returns_none_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "")
    agent = BaseAgent(agent_user(), AgentRepo())  # type: ignore[arg-type]

    assert await agent._call_llm([{"role": "user", "content": "hello"}]) is None


@pytest.mark.anyio
async def test_call_llm_invokes_langchain_and_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "test-key")
    async def noop_sleep(_: float) -> None:
        pass
    monkeypatch.setattr(base_agent.asyncio, "sleep", noop_sleep)
    agent = BaseAgent(agent_user(), AgentRepo())  # type: ignore[arg-type]
    fake_response = MagicMock()
    fake_response.content = "Synthetic reply"
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=fake_response)
    agent._llm = mock_llm

    result = await agent._call_llm([{"role": "user", "content": "hello"}])

    assert result == "Synthetic reply"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.anyio
async def test_call_llm_converts_messages_to_langchain_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "test-key")
    async def noop_sleep(_: float) -> None:
        pass
    monkeypatch.setattr(base_agent.asyncio, "sleep", noop_sleep)
    agent = BaseAgent(agent_user(), AgentRepo())  # type: ignore[arg-type]
    captured: dict[str, Any] = {}
    fake_response = MagicMock()
    fake_response.content = "ok"

    async def capture_ainvoke(messages: list[Any]) -> MagicMock:
        captured["messages"] = messages
        return fake_response

    mock_llm = MagicMock()
    mock_llm.ainvoke = capture_ainvoke
    agent._llm = mock_llm

    await agent._call_llm([
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
    ])

    assert isinstance(captured["messages"][0], SystemMessage)
    assert captured["messages"][0].content == "be helpful"
    assert isinstance(captured["messages"][1], HumanMessage)
    assert captured["messages"][1].content == "hello"


@pytest.mark.anyio
async def test_call_llm_returns_none_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "test-key")
    agent = BaseAgent(agent_user(), AgentRepo())  # type: ignore[arg-type]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("network down"))
    agent._llm = mock_llm

    assert await agent._call_llm([{"role": "user", "content": "hello"}]) is None


@pytest.mark.anyio
async def test_generate_media_returns_none_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "")
    user = agent_user(agent_config=agent_config(output_modality="image"))
    agent = BaseAgent(user, AgentRepo())  # type: ignore[arg-type]

    assert await agent._generate_media("draw something") is None


@pytest.mark.anyio
async def test_generate_media_returns_none_for_unsupported_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "test-key")
    user = agent_user(agent_config=agent_config(output_modality="video"))
    agent = BaseAgent(user, AgentRepo())  # type: ignore[arg-type]

    assert await agent._generate_media("generate video") is None


@pytest.mark.anyio
async def test_generate_media_decodes_base64_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64 as b64lib

    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(base_agent, "POST_REQUEST_DELAY_SECONDS", 0)

    fake_image = b"PNG\x89"
    encoded = b64lib.b64encode(fake_image).decode()

    class FakeResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"data": [{"b64_json": encoded}]}

    class FakeHttpxClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeHttpxClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def post(self, url: str, **kwargs: Any) -> FakeResp:
            return FakeResp()

    # Create agent BEFORE patching httpx.AsyncClient to avoid breaking
    # the isinstance check inside the openai / langchain initialization.
    user = agent_user(agent_config=agent_config(output_modality="image"))
    agent = BaseAgent(user, AgentRepo())  # type: ignore[arg-type]
    monkeypatch.setattr(base_agent.httpx, "AsyncClient", FakeHttpxClient)

    result = await agent._generate_media("draw a cat")
    assert result is not None
    raw, mime = result
    assert raw == fake_image
    assert mime == "image/png"


@pytest.mark.anyio
async def test_generate_media_returns_none_when_response_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "test-key")

    class EmptyResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"data": []}

    class FakeHttpxClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeHttpxClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def post(self, url: str, **kwargs: Any) -> EmptyResp:
            return EmptyResp()

    user = agent_user(agent_config=agent_config(output_modality="image"))
    agent = BaseAgent(user, AgentRepo())  # type: ignore[arg-type]
    monkeypatch.setattr(base_agent.httpx, "AsyncClient", FakeHttpxClient)

    assert await agent._generate_media("draw something") is None


@pytest.mark.anyio
async def test_generate_media_returns_none_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "test-key")

    class FakeHttpxClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeHttpxClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def post(self, url: str, **kwargs: Any) -> None:
            raise RuntimeError("timeout")

    user = agent_user(agent_config=agent_config(output_modality="image"))
    agent = BaseAgent(user, AgentRepo())  # type: ignore[arg-type]
    monkeypatch.setattr(base_agent.httpx, "AsyncClient", FakeHttpxClient)

    assert await agent._generate_media("draw something") is None


@pytest.mark.anyio
async def test_maybe_respond_image_agent_creates_media_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.storage.s3 as s3_mod

    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "test-key")

    user = agent_user(
        agent_config=agent_config(
            output_modality="image",
            persona_name="Pixel",
        )
    )
    repo = AgentRepo()
    repo.thread = Thread(thread_id="t1", title="Space exploration", created_by="human")
    repo.posts = [Post(thread_id="t1", author_id="human", content="Cool news")]
    agent = BaseAgent(user, repo)  # type: ignore[arg-type]
    monkeypatch.setattr(agent, "_should_respond", lambda: True)
    monkeypatch.setattr(
        agent,
        "_generate_media",
        AsyncMock(return_value=(b"imgdata", "image/png")),
    )
    monkeypatch.setattr(s3_mod, "upload_media", AsyncMock(return_value="http://s3/img.png"))

    parent = Post(thread_id="t1", author_id="human", content="Cool news")
    result = await agent.maybe_respond("t1", parent)

    assert result is not None
    assert result.media_url == "http://s3/img.png"
    assert result.content_type == "image/png"
    assert "[image generated by Pixel]" in result.content
    assert repo.created_posts == [result]


@pytest.mark.anyio
async def test_maybe_respond_image_agent_returns_none_when_media_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "test-key")

    user = agent_user(agent_config=agent_config(output_modality="image"))
    repo = AgentRepo()
    repo.thread = Thread(thread_id="t1", title="Test", created_by="human")
    repo.posts = []
    agent = BaseAgent(user, repo)  # type: ignore[arg-type]
    monkeypatch.setattr(agent, "_should_respond", lambda: True)
    monkeypatch.setattr(agent, "_generate_media", AsyncMock(return_value=None))

    parent = Post(thread_id="t1", author_id="human", content="hi")
    assert await agent.maybe_respond("t1", parent) is None


@pytest.mark.anyio
async def test_maybe_respond_image_agent_returns_none_when_s3_upload_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.storage.s3 as s3_mod

    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "test-key")

    user = agent_user(agent_config=agent_config(output_modality="image"))
    repo = AgentRepo()
    repo.thread = Thread(thread_id="t1", title="Test", created_by="human")
    repo.posts = []
    agent = BaseAgent(user, repo)  # type: ignore[arg-type]
    monkeypatch.setattr(agent, "_should_respond", lambda: True)
    monkeypatch.setattr(
        agent, "_generate_media", AsyncMock(return_value=(b"data", "image/png"))
    )
    monkeypatch.setattr(
        s3_mod, "upload_media", AsyncMock(side_effect=RuntimeError("S3 down"))
    )

    parent = Post(thread_id="t1", author_id="human", content="hi")
    assert await agent.maybe_respond("t1", parent) is None


@pytest.mark.anyio
async def test_maybe_respond_short_circuits_for_own_post_and_missing_thread() -> None:
    user = agent_user()
    repo = AgentRepo()
    agent = BaseAgent(user, repo)  # type: ignore[arg-type]

    own_post = Post(thread_id="thread-1", author_id=user.user_id, content="self")
    assert await agent.maybe_respond("thread-1", own_post) is None

    other_post = Post(thread_id="missing", author_id="other", content="hello")
    assert await agent.maybe_respond("missing", other_post) is None


@pytest.mark.anyio
async def test_maybe_respond_skips_when_conversation_is_outside_expertise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "agent_seed_reply_post_count", 2)
    user = agent_user(agent_config=agent_config(expertise_areas=["databases"]))
    repo = AgentRepo()
    repo.thread = Thread(thread_id="thread-1", title="Cooking", created_by="human")
    repo.posts = [
        Post(thread_id="thread-1", author_id="u1", content="one"),
        Post(thread_id="thread-1", author_id="u2", content="two"),
        Post(thread_id="thread-1", author_id="u3", content="three"),
    ]
    agent = BaseAgent(user, repo)  # type: ignore[arg-type]

    post = Post(thread_id="thread-1", author_id="human", content="recipes")

    assert await agent.maybe_respond("thread-1", post) is None


@pytest.mark.anyio
async def test_maybe_respond_respects_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    user = agent_user()
    repo = AgentRepo()
    repo.thread = Thread(thread_id="thread-1", title="Python", created_by="human")
    repo.posts = [Post(thread_id="thread-1", author_id="human", content="question")]
    agent = BaseAgent(user, repo)  # type: ignore[arg-type]
    monkeypatch.setattr(agent, "_should_respond", lambda: False)

    post = Post(thread_id="thread-1", author_id="human", content="Python help")

    assert await agent.maybe_respond("thread-1", post) is None


@pytest.mark.anyio
async def test_maybe_respond_returns_none_when_llm_has_no_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = agent_user()
    repo = AgentRepo()
    repo.thread = Thread(thread_id="thread-1", title="Python", created_by="human")
    repo.posts = [Post(thread_id="thread-1", author_id="human", content="question")]
    agent = BaseAgent(user, repo)  # type: ignore[arg-type]
    monkeypatch.setattr(agent, "_should_respond", lambda: True)

    async def no_reply(messages: list[dict[str, str]]) -> None:
        return None

    monkeypatch.setattr(agent, "_call_llm", no_reply)

    post = Post(thread_id="thread-1", author_id="human", content="Python help")

    assert await agent.maybe_respond("thread-1", post) is None


@pytest.mark.anyio
async def test_maybe_respond_creates_trimmed_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    user = agent_user()
    human = User(user_id="human", username="Alice")
    repo = AgentRepo()
    repo.thread = Thread(thread_id="thread-1", title="Python", created_by=human.user_id)
    repo.posts = [
        Post(thread_id="thread-1", author_id=human.user_id, content="How do decorators work?"),
        Post(thread_id="thread-1", author_id="missing", content="Unknown author here"),
    ]
    repo.users[human.user_id] = human
    agent = BaseAgent(user, repo)  # type: ignore[arg-type]
    monkeypatch.setattr(agent, "_should_respond", lambda: True)
    captured: dict[str, list[dict[str, str]]] = {}

    async def reply(messages: list[dict[str, str]]) -> str:
        captured["messages"] = messages
        return "  Use a wrapper function.  "

    monkeypatch.setattr(agent, "_call_llm", reply)

    parent = Post(thread_id="thread-1", author_id=human.user_id, content="Python help")
    result = await agent.maybe_respond("thread-1", parent)

    assert result is not None
    assert result.content == "Use a wrapper function."
    assert result.parent_post_id == parent.post_id
    assert repo.created_posts == [result]
    assert repo.activity_updates == ["thread-1"]
    assert captured["messages"][0] == {"role": "system", "content": "Be useful."}
    assert any("Alice:" in message["content"] for message in captured["messages"])
    assert any("Unknown:" in message["content"] for message in captured["messages"])


class RenderTemplate:
    def render(self, **context: object) -> str:
        return f"rendered:{context['post'].post_id}"


class RenderTemplates:
    def get_template(self, name: str) -> RenderTemplate:
        assert name == "fragments/post.html"
        return RenderTemplate()


class RecordingWSManager:
    def __init__(self) -> None:
        self.broadcasts: list[tuple[str, str]] = []

    async def broadcast(self, thread_id: str, html: str) -> None:
        self.broadcasts.append((thread_id, html))


@pytest.mark.anyio
async def test_discussion_engine_schedules_agent_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = agent_user()
    engine = DiscussionEngine(
        repo=AgentRepo(),  # type: ignore[arg-type]
        agents=[user, User(username="human")],
        ws_manager=RecordingWSManager(),  # type: ignore[arg-type]
        templates=RenderTemplates(),
    )
    delays: list[float] = []
    scheduled: list[object] = []

    class Loop:
        def call_later(self, delay: float, callback: object) -> None:
            delays.append(delay)
            callback()  # type: ignore[operator]

    def fake_ensure_future(coro: object) -> None:
        scheduled.append(coro)
        coro.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(discussion.random, "uniform", lambda low, high: 7.5)
    monkeypatch.setattr(discussion.asyncio, "get_event_loop", lambda: Loop())
    monkeypatch.setattr(discussion.asyncio, "ensure_future", fake_ensure_future)

    await engine.on_new_post("thread-1", Post(thread_id="thread-1", author_id="human", content="hi"))

    assert len(engine._agents) == 1
    assert delays == [7.5]
    assert len(scheduled) == 1


@pytest.mark.anyio
async def test_discussion_engine_broadcasts_agent_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = RecordingWSManager()
    engine = DiscussionEngine(
        repo=AgentRepo(),  # type: ignore[arg-type]
        agents=[],
        ws_manager=ws,  # type: ignore[arg-type]
        templates=RenderTemplates(),
    )
    reply = Post(thread_id="thread-1", author_id="agent", content="reply")
    retriggered: list[tuple[str, Post]] = []

    class FakeAgent:
        user = agent_user(user_id="agent", username="Agent")

        async def maybe_respond(self, thread_id: str, post: Post) -> Post:
            return reply

    async def record_new_post(thread_id: str, post: Post) -> None:
        retriggered.append((thread_id, post))

    monkeypatch.setattr(engine, "on_new_post", record_new_post)

    await engine._agent_respond(FakeAgent(), "thread-1", Post(thread_id="thread-1", author_id="human", content="hi"))  # type: ignore[arg-type]

    assert ws.broadcasts == [("thread-1", f'<div hx-swap-oob="beforeend:#posts">rendered:{reply.post_id}</div>')]
    assert retriggered == [("thread-1", reply)]


@pytest.mark.anyio
async def test_discussion_engine_logs_agent_errors() -> None:
    engine = DiscussionEngine(
        repo=AgentRepo(),  # type: ignore[arg-type]
        agents=[],
        ws_manager=RecordingWSManager(),  # type: ignore[arg-type]
        templates=RenderTemplates(),
    )

    class BrokenAgent:
        user = agent_user(username="Broken")

        async def maybe_respond(self, thread_id: str, post: Post) -> Post | None:
            raise RuntimeError("boom")

    await engine._agent_respond(BrokenAgent(), "thread-1", Post(thread_id="thread-1", author_id="human", content="hi"))  # type: ignore[arg-type]


class NewsRepo:
    def __init__(self, items: list[NewsItem]) -> None:
        self.items = items
        self.created_threads: list[Thread] = []
        self.created_posts: list[Post] = []
        self.status_updates: list[tuple[str, str, str | None]] = []

    async def get_news_items_by_status(self, status: str) -> list[NewsItem]:
        return [item for item in self.items if item.status == status]

    async def create_thread(self, thread: Thread) -> Thread:
        self.created_threads.append(thread)
        return thread

    async def create_post(self, post: Post) -> Post:
        self.created_posts.append(post)
        return post

    async def update_news_item_status(
        self,
        item_id: str,
        status: str,
        promoted_thread_id: str | None = None,
    ) -> None:
        self.status_updates.append((item_id, status, promoted_thread_id))


@pytest.mark.anyio
async def test_news_agent_promotes_limited_items_and_skips_remaining() -> None:
    long_content = "x" * 350
    items = [
        NewsItem(item_id="n1", source="hackernews", title="Long", url="https://one", raw_content=long_content),
        NewsItem(item_id="n2", source="hackernews", title="Empty", url="https://two", raw_content=None),
        NewsItem(item_id="n3", source="hackernews", title="Skipped", url="https://three"),
    ]
    repo = NewsRepo(items)
    user = agent_user(username="Nova")
    agent = NewsAgent(user, repo)  # type: ignore[arg-type]

    threads = await agent.promote_news(max_items=2)

    assert threads == repo.created_threads
    assert [thread.title for thread in threads] == ["Long", "Empty"]
    assert repo.created_threads[0].summary == long_content[:500]
    assert repo.created_threads[1].summary is None
    assert repo.created_posts[0].content.endswith("What are your thoughts on this?")
    assert f"{long_content[:300]}..." in repo.created_posts[0].content
    assert "No summary available." in repo.created_posts[1].content
    assert repo.status_updates == [
        ("n1", "promoted", threads[0].thread_id),
        ("n2", "promoted", threads[1].thread_id),
        ("n3", "skipped", None),
    ]


@pytest.mark.anyio
async def test_register_agents_reuses_existing_and_creates_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = User(username="Nova", is_agent=True, agent_config=agent_config(persona_name="Nova"))
    created: list[User] = []

    class Repo:
        async def get_user_by_username(self, username: str) -> User | None:
            return existing if username == "Nova" else None

        async def create_user(self, user: User) -> User:
            created.append(user)
            return user

    monkeypatch.setattr(registry, "hash_password", lambda password: f"hashed:{password}")

    users = await registry.register_agents(Repo())  # type: ignore[arg-type]

    assert users[0] is existing
    assert len(users) == len(registry.PERSONA_PRESETS)
    assert [user.username for user in created] == ["Sage", "Pixel", "Ember", "Atlas", "Zara"]
    assert all(
        user.password_hash == f"hashed:{user.username}_agent_secret" for user in created
    )
    assert all(user.is_agent for user in created)


@pytest.mark.anyio
async def test_discussion_engine_reload_agents_replaces_agent_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_user = agent_user(username="Nova")
    engine = DiscussionEngine(
        repo=AgentRepo(),  # type: ignore[arg-type]
        agents=[original_user],
        ws_manager=RecordingWSManager(),  # type: ignore[arg-type]
        templates=RenderTemplates(),
    )
    assert len(engine._agents) == 1
    assert engine._agents[0].user.username == "Nova"

    new_user = agent_user(username="Sage", user_id="sage-id")
    engine.reload_agents([new_user])

    assert len(engine._agents) == 1
    assert engine._agents[0].user.username == "Sage"


# ── Cross-thread memory ────────────────────────────────────────────────────────


def test_tokenize_returns_meaningful_words() -> None:
    result = _tokenize("Python async patterns are really powerful")
    assert "python" in result
    assert "async" in result
    assert "patterns" in result
    # "are" and "really" are stop-words or too short
    assert "are" not in result
    assert "really" not in result


def test_tokenize_filters_short_words() -> None:
    result = _tokenize("the big cat sat")
    # "big", "cat", "sat" are < 4 chars; "the" is a stop-word
    assert result == set()


def test_tokenize_is_case_insensitive() -> None:
    assert _tokenize("PYTHON Python python") == {"python"}


@pytest.mark.anyio
async def test_find_related_threads_returns_overlapping_threads() -> None:
    repo = AgentRepo()
    current = Thread(thread_id="current", title="Python async programming", created_by="u")
    related = Thread(thread_id="rel1", title="Python performance tips", created_by="u")
    unrelated = Thread(thread_id="other", title="Cooking recipes collection", created_by="u")
    repo.thread = current
    repo.all_threads = [current, related, unrelated]
    repo.posts = [
        Post(thread_id="current", author_id="u", content="hi"),
        Post(thread_id="rel1", author_id="u", content="faster loops"),
    ]

    agent = BaseAgent(agent_user(), repo)  # type: ignore[arg-type]
    keywords = _tokenize("Python async programming")
    result = await agent._find_related_threads("current", keywords)

    assert len(result) == 1
    assert result[0][0].thread_id == "rel1"
    assert len(result[0][1]) == 1  # one post


@pytest.mark.anyio
async def test_find_related_threads_excludes_current_thread() -> None:
    repo = AgentRepo()
    current = Thread(thread_id="current", title="Python async patterns", created_by="u")
    repo.thread = current
    repo.all_threads = [current]
    repo.posts = [Post(thread_id="current", author_id="u", content="hi")]

    agent = BaseAgent(agent_user(), repo)  # type: ignore[arg-type]
    result = await agent._find_related_threads("current", {"python", "async"})

    assert result == []


@pytest.mark.anyio
async def test_find_related_threads_returns_empty_for_empty_keywords() -> None:
    repo = AgentRepo()
    repo.all_threads = [Thread(thread_id="t1", title="Python tips", created_by="u")]
    repo.posts = [Post(thread_id="t1", author_id="u", content="hi")]

    agent = BaseAgent(agent_user(), repo)  # type: ignore[arg-type]
    result = await agent._find_related_threads("other", set())

    assert result == []


@pytest.mark.anyio
async def test_find_related_threads_caps_at_max() -> None:
    repo = AgentRepo()
    # Create 5 threads all matching "python"
    repo.all_threads = [
        Thread(thread_id=f"t{i}", title=f"Python thread {i}", created_by="u")
        for i in range(5)
    ]
    repo.posts = [
        Post(thread_id=f"t{i}", author_id="u", content="some post")
        for i in range(5)
    ]

    agent = BaseAgent(agent_user(), repo)  # type: ignore[arg-type]
    result = await agent._find_related_threads("other-thread", {"python"})

    assert len(result) <= base_agent._MAX_RELATED_THREADS


@pytest.mark.anyio
async def test_maybe_respond_includes_related_thread_in_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(base_agent.settings, "agent_seed_reply_post_count", 0)

    current = Thread(thread_id="t1", title="Python async programming", created_by="human")
    related = Thread(thread_id="t2", title="Python performance patterns", created_by="human")

    repo = AgentRepo()
    repo.thread = current
    repo.all_threads = [current, related]
    repo.posts = [
        Post(thread_id="t1", author_id="human", content="Tell me about async"),
        Post(thread_id="t2", author_id="human", content="Generators are fast"),
    ]
    repo.users = {"human": User(username="alice", user_id="human")}

    captured_messages: list[Any] = []
    fake_response = MagicMock()
    fake_response.content = "Great question!"

    async def fake_ainvoke(messages: list[Any]) -> MagicMock:
        captured_messages.extend(messages)
        return fake_response

    async def noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr(base_agent.asyncio, "sleep", noop_sleep)

    agent = BaseAgent(agent_user(), repo)  # type: ignore[arg-type]
    mock_llm = MagicMock()
    mock_llm.ainvoke = fake_ainvoke
    agent._llm = mock_llm
    monkeypatch.setattr(agent, "_should_respond", lambda: True)

    trigger = Post(thread_id="t1", author_id="human", content="Tell me about async")
    await agent.maybe_respond("t1", trigger)

    assert captured_messages, "LLM was never called"
    prompt_text = " ".join(str(m.content) for m in captured_messages)
    assert "Python performance patterns" in prompt_text
    assert "Generators are fast" in prompt_text


# ── HuggingFace provider routing ──────────────────────────────────────────────


def test_base_agent_uses_hf_base_url_for_hf_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "huggingface_api_key", "hf-test-token")
    user = agent_user(
        agent_config=agent_config(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            provider="huggingface",
        )
    )
    agent = BaseAgent(user, AgentRepo())  # type: ignore[arg-type]
    # The LLM client's openai_api_base should point to HuggingFace
    client_base = agent._llm.bound.openai_api_base  # type: ignore[union-attr]
    assert "huggingface" in client_base or "router.huggingface" in client_base


def test_base_agent_uses_openrouter_base_url_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "openrouter_api_key", "or-test-key")
    agent = BaseAgent(agent_user(), AgentRepo())  # type: ignore[arg-type]
    client_base = agent._llm.bound.openai_api_base  # type: ignore[union-attr]
    assert "openrouter.ai" in client_base


@pytest.mark.anyio
async def test_call_llm_returns_none_without_hf_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "huggingface_api_key", "")
    user = agent_user(
        agent_config=agent_config(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            provider="huggingface",
        )
    )
    agent = BaseAgent(user, AgentRepo())  # type: ignore[arg-type]
    result = await agent._call_llm([{"role": "user", "content": "hello"}])
    assert result is None


@pytest.mark.anyio
async def test_call_llm_hf_invokes_llm_with_hf_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_agent.settings, "huggingface_api_key", "hf-test-token")

    async def noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr(base_agent.asyncio, "sleep", noop_sleep)

    user = agent_user(
        agent_config=agent_config(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            provider="huggingface",
        )
    )
    agent = BaseAgent(user, AgentRepo())  # type: ignore[arg-type]
    fake_response = MagicMock()
    fake_response.content = "HF reply"
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=fake_response)
    agent._llm = mock_llm

    result = await agent._call_llm([{"role": "user", "content": "hello"}])
    assert result == "HF reply"
    mock_llm.ainvoke.assert_called_once()


