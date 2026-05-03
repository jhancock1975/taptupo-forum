from __future__ import annotations

from datetime import UTC, datetime

from app.models.schemas import AgentConfig, NewsItem, Post, Thread, User


class TestUserModel:
    def test_create_user_defaults(self) -> None:
        user = User(username="testuser")
        assert user.username == "testuser"
        assert user.is_agent is False
        assert user.agent_config is None
        assert user.password_hash is None
        assert user.user_id
        assert user.created_at <= datetime.now(UTC)

    def test_create_agent_user(self) -> None:
        config = AgentConfig(
            model_id="test/model:free",
            persona_name="TestBot",
            expertise_areas=["tech"],
            personality_traits=["friendly"],
            response_probability=0.6,
            system_prompt="You are a test bot.",
        )
        user = User(username="TestBot", is_agent=True, agent_config=config)
        assert user.is_agent is True
        assert user.agent_config is not None
        assert user.agent_config.persona_name == "TestBot"
        assert user.agent_config.response_probability == 0.6


class TestThreadModel:
    def test_create_thread(self) -> None:
        thread = Thread(title="Test Thread", created_by="user-123")
        assert thread.title == "Test Thread"
        assert thread.created_by == "user-123"
        assert thread.thread_id
        assert thread.reply_count == 0
        assert thread.categories == []

    def test_thread_with_source(self) -> None:
        thread = Thread(
            title="HN Story",
            created_by="agent-1",
            source_url="https://example.com",
            source_type="hackernews",
            categories=["tech"],
        )
        assert thread.source_type == "hackernews"
        assert thread.source_url == "https://example.com"


class TestPostModel:
    def test_create_post(self) -> None:
        post = Post(
            thread_id="thread-1",
            author_id="user-1",
            content="Hello world",
        )
        assert post.thread_id == "thread-1"
        assert post.author_id == "user-1"
        assert post.content == "Hello world"
        assert post.parent_post_id is None

    def test_reply_post(self) -> None:
        post = Post(
            thread_id="thread-1",
            author_id="user-2",
            content="Reply",
            parent_post_id="post-1",
        )
        assert post.parent_post_id == "post-1"


class TestNewsItemModel:
    def test_create_news_item(self) -> None:
        item = NewsItem(
            source="hackernews",
            title="Test Story",
            url="https://example.com/story",
        )
        assert item.source == "hackernews"
        assert item.status == "new"
        assert item.promoted_thread_id is None


class TestAgentConfig:
    def test_defaults(self) -> None:
        config = AgentConfig(model_id="test/m", persona_name="T")
        assert config.expertise_areas == []
        assert config.personality_traits == []
        assert config.response_probability == 0.5
        assert config.system_prompt == ""
