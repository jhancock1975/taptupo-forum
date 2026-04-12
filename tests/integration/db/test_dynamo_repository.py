"""Integration tests for :class:`DynamoRepository` against moto.

moto's ``@mock_aws`` decorator monkey-patches ``botocore`` in-process,
but ``aiobotocore`` (used by ``aioboto3``) has its own response handling
that tries to ``await`` a ``bytes`` body — incompatible with moto's
in-process interception. The workaround is :class:`ThreadedMotoServer`,
which starts a real HTTP endpoint in a background thread; ``aioboto3``
is pointed at it via ``endpoint_url`` and everything behaves normally.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import boto3
import httpx
import pytest
import pytest_asyncio
from moto.server import ThreadedMotoServer

from app.db.dynamo import DynamoRepository
from app.db.interface import UserExistsError
from app.models import AgentConfig, NewsItem, Post, Thread, User
from scripts.setup_tables import create_tables

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _human(username: str = "alice") -> User:
    return User(username=username, password_hash="argon2$fake", is_agent=False)


def _agent(username: str = "bot") -> User:
    return User(
        username=username,
        is_agent=True,
        agent_config=AgentConfig(
            model_id="openrouter/free-model",
            persona_name="Botty",
            expertise_areas=["ai"],
            personality_traits=["curious"],
            response_probability=0.5,
            system_prompt="be helpful",
        ),
    )


@pytest_asyncio.fixture
async def repo() -> AsyncIterator[DynamoRepository]:
    server = ThreadedMotoServer(port=0)
    server.start()
    try:
        host, port = server.get_host_and_port()
        endpoint = f"http://{host}:{port}"
        # aioboto3 (via botocore) still needs *some* credentials to sign.
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
        # moto keeps backend state in process globals, so reset before use.
        httpx.post(f"{endpoint}/moto-api/reset")
        client = boto3.client("dynamodb", region_name="us-east-1", endpoint_url=endpoint)
        create_tables(client)
        yield DynamoRepository(endpoint_url=endpoint, region_name="us-east-1")
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
async def test_create_and_get_user(repo: DynamoRepository) -> None:
    user = _human("alice")
    await repo.create_user(user)
    got = await repo.get_user(user.user_id)
    assert got is not None
    assert got.username == "alice"


async def test_get_user_by_username_returns_none_for_missing(
    repo: DynamoRepository,
) -> None:
    assert await repo.get_user_by_username("nobody") is None


async def test_get_user_by_username_returns_user(repo: DynamoRepository) -> None:
    user = _human("alice")
    await repo.create_user(user)
    got = await repo.get_user_by_username("alice")
    assert got is not None
    assert got.user_id == user.user_id


async def test_create_duplicate_username_raises_user_exists_error(
    repo: DynamoRepository,
) -> None:
    await repo.create_user(_human("alice"))
    with pytest.raises(UserExistsError):
        await repo.create_user(_human("alice"))


async def test_list_agents_returns_only_agents(repo: DynamoRepository) -> None:
    await repo.create_user(_human("alice"))
    await repo.create_user(_agent("bot1"))
    await repo.create_user(_agent("bot2"))
    agents = await repo.list_agents()
    assert {a.username for a in agents} == {"bot1", "bot2"}


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------
def _thread(title: str = "t", created_by: str = "u1") -> Thread:
    return Thread(title=title, source_type="human", created_by=created_by)


async def test_create_and_get_thread(repo: DynamoRepository) -> None:
    t = _thread("Hello")
    await repo.create_thread(t)
    got = await repo.get_thread(t.thread_id)
    assert got is not None
    assert got.title == "Hello"


async def test_get_thread_returns_none_for_missing(repo: DynamoRepository) -> None:
    assert await repo.get_thread("missing") is None


async def test_list_threads_sorts_by_last_activity_desc_and_respects_limit(
    repo: DynamoRepository,
) -> None:
    from datetime import UTC, datetime, timedelta

    base = datetime.now(UTC)
    t1 = Thread(
        title="oldest",
        source_type="human",
        created_by="u",
        last_activity_at=base - timedelta(hours=2),
    )
    t2 = Thread(
        title="middle",
        source_type="human",
        created_by="u",
        last_activity_at=base - timedelta(hours=1),
    )
    t3 = Thread(
        title="newest", source_type="human", created_by="u", last_activity_at=base
    )
    for t in (t1, t2, t3):
        await repo.create_thread(t)
    listed = await repo.list_threads(limit=2)
    assert [t.title for t in listed] == ["newest", "middle"]


async def test_update_thread_activity_updates_field(repo: DynamoRepository) -> None:
    from datetime import UTC, datetime, timedelta

    t = _thread()
    await repo.create_thread(t)
    new_ts = datetime.now(UTC) + timedelta(hours=1)
    await repo.update_thread_activity(t.thread_id, new_ts)
    got = await repo.get_thread(t.thread_id)
    assert got is not None
    assert abs((got.last_activity_at - new_ts).total_seconds()) < 1


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------
async def test_create_post_bumps_thread_last_activity(repo: DynamoRepository) -> None:
    from datetime import UTC, datetime, timedelta

    t = Thread(
        title="x",
        source_type="human",
        created_by="u",
        last_activity_at=datetime.now(UTC) - timedelta(hours=1),
    )
    await repo.create_thread(t)
    original = (await repo.get_thread(t.thread_id)).last_activity_at  # type: ignore[union-attr]
    p = Post(thread_id=t.thread_id, author_id="u", content="hi")
    await repo.create_post(p)
    bumped = (await repo.get_thread(t.thread_id)).last_activity_at  # type: ignore[union-attr]
    assert bumped > original


async def test_get_posts_by_thread_returns_oldest_first(repo: DynamoRepository) -> None:
    from datetime import UTC, datetime, timedelta

    t = _thread()
    await repo.create_thread(t)
    base = datetime.now(UTC)
    p1 = Post(
        thread_id=t.thread_id, author_id="u", content="first", created_at=base
    )
    p2 = Post(
        thread_id=t.thread_id,
        author_id="u",
        content="second",
        created_at=base + timedelta(seconds=1),
    )
    p3 = Post(
        thread_id=t.thread_id,
        author_id="u",
        content="third",
        created_at=base + timedelta(seconds=2),
    )
    # Insert out of order to ensure GSI sort drives result order.
    for p in (p3, p1, p2):
        await repo.create_post(p)
    got = await repo.get_posts_by_thread(t.thread_id)
    assert [p.content for p in got] == ["first", "second", "third"]


async def test_get_post_returns_none_for_missing(repo: DynamoRepository) -> None:
    assert await repo.get_post("missing") is None


# ---------------------------------------------------------------------------
# NewsItems
# ---------------------------------------------------------------------------
def _news(url: str = "https://example.com/a", status: str = "new") -> NewsItem:
    return NewsItem(
        source="guardian",
        title="headline",
        url=url,
        raw_content="body",
        status=status,  # type: ignore[arg-type]
    )


async def test_create_and_get_news_item(repo: DynamoRepository) -> None:
    n = _news()
    await repo.create_news_item(n)
    got = await repo.get_news_item(n.item_id)
    assert got is not None
    assert got.url == "https://example.com/a"


async def test_list_new_news_items_filters_to_status_new(
    repo: DynamoRepository,
) -> None:
    new_item = _news("https://example.com/new", status="new")
    promoted = _news("https://example.com/promoted", status="promoted")
    skipped = _news("https://example.com/skipped", status="skipped")
    for i in (new_item, promoted, skipped):
        await repo.create_news_item(i)
    got = await repo.list_new_news_items()
    assert [i.url for i in got] == ["https://example.com/new"]


async def test_update_news_item_status_to_promoted_stores_thread_id(
    repo: DynamoRepository,
) -> None:
    n = _news()
    await repo.create_news_item(n)
    await repo.update_news_item_status(n.item_id, "promoted", promoted_thread_id="t-42")
    got = await repo.get_news_item(n.item_id)
    assert got is not None
    assert got.status == "promoted"
    assert got.promoted_thread_id == "t-42"


async def test_news_item_exists_by_url_true_false(repo: DynamoRepository) -> None:
    await repo.create_news_item(_news("https://example.com/here"))
    assert await repo.news_item_exists_by_url("https://example.com/here") is True
    assert await repo.news_item_exists_by_url("https://example.com/nope") is False
