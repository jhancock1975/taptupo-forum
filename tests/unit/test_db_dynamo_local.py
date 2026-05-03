from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from botocore.exceptions import ClientError

from app.db import dynamo_local
from app.db.dynamo_local import DynamoLocalRepository, _replace_floats, _restore_floats
from app.db.interface import RepositoryInterface
from app.models.schemas import AgentConfig, NewsItem, Post, Thread, User


class FakeTable:
    def __init__(self, name: str) -> None:
        self.name = name
        self.items: dict[str, dict[str, Any]] = {}
        self.update_calls: list[dict[str, Any]] = []

    def _key_name(self) -> str:
        return {
            "Users": "user_id",
            "Threads": "thread_id",
            "Posts": "post_id",
            "NewsItems": "item_id",
        }[self.name]

    def put_item(self, Item: dict[str, Any]) -> None:
        self.items[Item[self._key_name()]] = Item

    def get_item(self, Key: dict[str, str]) -> dict[str, Any]:
        key = next(iter(Key.values()))
        item = self.items.get(key)
        return {"Item": item} if item else {}

    def query(self, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        values = kwargs["ExpressionAttributeValues"]
        items = list(self.items.values())
        if ":u" in values:
            items = [item for item in items if item["username"] == values[":u"]]
        if ":t" in values:
            items = [item for item in items if item["thread_id"] == values[":t"]]
        if ":s" in values:
            items = [item for item in items if item["status"] == values[":s"]]
        return {"Items": items}

    def scan(self, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        items = list(self.items.values())
        values = kwargs.get("ExpressionAttributeValues", {})
        if values.get(":t") is True:
            items = [item for item in items if item.get("is_agent") is True]
        if ":u" in values:
            items = [item for item in items if values[":u"] in item.get("url", "")]
        if "Limit" in kwargs:
            items = items[: kwargs["Limit"]]
        return {"Items": items}

    def update_item(self, **kwargs: Any) -> None:
        self.update_calls.append(kwargs)
        key = next(iter(kwargs["Key"].values()))
        item = self.items[key]
        values = kwargs["ExpressionAttributeValues"]
        if self.name == "Threads":
            item["last_activity_at"] = values[":t"]
            item["reply_count"] += values[":inc"]
        if self.name == "NewsItems":
            item["status"] = values[":s"]
            if ":p" in values:
                item["promoted_thread_id"] = values[":p"]
        if self.name == "Users" and ":c" in values:
            item["agent_config"] = values[":c"]


class FakeClient:
    def __init__(
        self,
        table_names: list[str] | None = None,
        table_sizes: dict[str, int] | None = None,
    ) -> None:
        self.table_names = table_names or []
        self.table_sizes = table_sizes or {}

    def list_tables(self) -> dict[str, list[str]]:
        return {"TableNames": self.table_names}

    def describe_table(self, TableName: str) -> dict[str, Any]:
        size = self.table_sizes.get(TableName, 0)
        return {"Table": {"TableName": TableName, "TableSizeBytes": size}}


class FakeResource:
    def __init__(
        self,
        table_names: list[str] | None = None,
        create_error: ClientError | None = None,
        table_sizes: dict[str, int] | None = None,
    ) -> None:
        self.meta = type("Meta", (), {"client": FakeClient(table_names, table_sizes)})()  # type: ignore[arg-type]
        self.tables = {
            name: FakeTable(name) for name in ["Users", "Threads", "Posts", "NewsItems"]
        }
        self.created_tables: list[str] = []
        self.create_error = create_error

    def Table(self, name: str) -> FakeTable:
        return self.tables[name]

    def create_table(self, **table_def: Any) -> None:
        if self.create_error:
            raise self.create_error
        self.created_tables.append(table_def["TableName"])


def make_repo(
    monkeypatch: pytest.MonkeyPatch,
    resource: FakeResource,
    *,
    inline_run: bool = True,
) -> DynamoLocalRepository:
    monkeypatch.setattr(
        dynamo_local.boto3, "resource", lambda *args, **kwargs: resource
    )
    repo = DynamoLocalRepository()
    if inline_run:

        async def run_inline(fn: Any, *args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        repo._run = run_inline  # type: ignore[method-assign]
    return repo


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "CreateTable")


def test_float_conversion_helpers_handle_nested_values() -> None:
    converted = _replace_floats({"ratio": 0.25, "items": [1.5, {"plain": "x"}]})

    assert converted == {
        "ratio": Decimal("0.25"),
        "items": [Decimal("1.5"), {"plain": "x"}],
    }
    assert _restore_floats(converted) == {"ratio": 0.25, "items": [1.5, {"plain": "x"}]}
    assert _replace_floats("unchanged") == "unchanged"
    assert _restore_floats("unchanged") == "unchanged"


@pytest.mark.anyio
async def test_run_delegates_to_event_loop_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = FakeResource()
    repo = make_repo(monkeypatch, resource, inline_run=False)

    class Loop:
        def run_in_executor(self, executor: object, call: object) -> object:
            async def result() -> Any:
                return call()  # type: ignore[operator]

            return result()

    monkeypatch.setattr(dynamo_local.asyncio, "get_event_loop", lambda: Loop())

    assert await repo._run(lambda left, right: left + right, 2, 3) == 5


@pytest.mark.anyio
async def test_init_tables_creates_missing_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = FakeResource(table_names=["Users"])
    repo = make_repo(monkeypatch, resource)

    await repo.init_tables()

    assert resource.created_tables == ["Threads", "Posts", "NewsItems"]


@pytest.mark.anyio
async def test_init_tables_ignores_concurrent_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = FakeResource(create_error=client_error("ResourceInUseException"))
    repo = make_repo(monkeypatch, resource)

    await repo.init_tables()


@pytest.mark.anyio
async def test_init_tables_reraises_unexpected_client_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = FakeResource(create_error=client_error("AccessDeniedException"))
    repo = make_repo(monkeypatch, resource)

    with pytest.raises(ClientError):
        await repo.init_tables()


@pytest.mark.anyio
async def test_user_methods_round_trip_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = FakeResource()
    repo = make_repo(monkeypatch, resource)
    config = AgentConfig(
        model_id="test/model",
        persona_name="Agent",
        expertise_areas=["python"],
        response_probability=0.75,
    )
    agent = User(username="agent", is_agent=True, agent_config=config)
    human = User(username="human", password_hash="hash")

    await repo.create_user(agent)
    await repo.create_user(human)

    assert await repo.get_user("missing") is None
    loaded_agent = await repo.get_user(agent.user_id)
    assert loaded_agent == agent
    assert await repo.get_user_by_username("nobody") is None
    assert await repo.get_user_by_username("human") == human
    assert await repo.list_agents() == [agent]

    raw = human.model_dump(mode="json")
    raw["user_id"] = "json-agent"
    raw["username"] = "json-agent"
    raw["is_agent"] = True
    raw["agent_config"] = json.dumps(config.model_dump(mode="json"))
    resource.Table("Users").items["json-agent"] = raw

    loaded_json_agent = await repo.get_user("json-agent")
    assert loaded_json_agent is not None
    assert loaded_json_agent.agent_config == config


@pytest.mark.anyio
async def test_thread_methods_sort_and_update(monkeypatch: pytest.MonkeyPatch) -> None:
    resource = FakeResource()
    repo = make_repo(monkeypatch, resource)
    older = Thread(
        thread_id="older",
        title="Older",
        created_by="u1",
        last_activity_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    newer = Thread(
        thread_id="newer",
        title="Newer",
        created_by="u2",
        last_activity_at=datetime(2024, 1, 2, tzinfo=UTC),
    )

    await repo.create_thread(older)
    await repo.create_thread(newer)

    assert await repo.get_thread("missing") is None
    assert await repo.get_thread("older") == older
    assert await repo.list_threads(limit=50) == [newer, older]

    await repo.update_thread_activity("older")
    updated = await repo.get_thread("older")
    assert updated is not None
    assert updated.reply_count == 1
    assert updated.last_activity_at > older.last_activity_at


@pytest.mark.anyio
async def test_post_methods_sort_by_created_at(monkeypatch: pytest.MonkeyPatch) -> None:
    resource = FakeResource()
    repo = make_repo(monkeypatch, resource)
    late = Post(
        post_id="late",
        thread_id="thread-1",
        author_id="u1",
        content="late",
        created_at=datetime(2024, 1, 2, tzinfo=UTC),
    )
    early = Post(
        post_id="early",
        thread_id="thread-1",
        author_id="u2",
        content="early",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )

    await repo.create_post(late)
    await repo.create_post(early)

    assert await repo.get_posts_by_thread("thread-1") == [early, late]
    assert await repo.get_posts_by_thread("other") == []


@pytest.mark.anyio
async def test_news_item_methods_query_and_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = FakeResource()
    repo = make_repo(monkeypatch, resource)
    first = NewsItem(
        item_id="n1", source="hn", title="One", url="https://example.com/one"
    )
    second = NewsItem(
        item_id="n2", source="hn", title="Two", url="https://example.com/two"
    )

    await repo.create_news_item(first)
    await repo.create_news_item(second)

    assert await repo.get_news_items_by_status("new") == [first, second]
    assert await repo.get_news_item_by_url("missing") is None
    assert await repo.get_news_item_by_url("/two") == second

    await repo.update_news_item_status("n1", "promoted", "thread-1")
    await repo.update_news_item_status("n2", "skipped")

    promoted = await repo.get_news_item_by_url("/one")
    skipped = await repo.get_news_item_by_url("/two")
    assert promoted is not None
    assert skipped is not None
    assert promoted.status == "promoted"
    assert promoted.promoted_thread_id == "thread-1"
    assert skipped.status == "skipped"
    assert skipped.promoted_thread_id is None


@pytest.mark.anyio
async def test_get_storage_bytes_sums_table_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = {"Users": 1024, "Threads": 2048, "Posts": 4096, "NewsItems": 512}
    resource = FakeResource(table_sizes=sizes)
    repo = make_repo(monkeypatch, resource)

    total = await repo.get_storage_bytes()

    assert total == 1024 + 2048 + 4096 + 512


@pytest.mark.anyio
async def test_get_storage_bytes_returns_zero_when_no_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = FakeResource()
    repo = make_repo(monkeypatch, resource)

    total = await repo.get_storage_bytes()

    assert total == 0


class InterfaceProbe(RepositoryInterface):
    async def init_tables(self) -> None:
        return await super().init_tables()

    async def create_user(self, user: User) -> User:
        return await super().create_user(user)

    async def get_user(self, user_id: str) -> User | None:
        return await super().get_user(user_id)

    async def get_user_by_username(self, username: str) -> User | None:
        return await super().get_user_by_username(username)

    async def list_agents(self) -> list[User]:
        return await super().list_agents()

    async def create_thread(self, thread: Thread) -> Thread:
        return await super().create_thread(thread)

    async def get_thread(self, thread_id: str) -> Thread | None:
        return await super().get_thread(thread_id)

    async def list_threads(self, limit: int = 50) -> list[Thread]:
        return await super().list_threads(limit)

    async def update_thread_activity(self, thread_id: str) -> None:
        return await super().update_thread_activity(thread_id)

    async def create_post(self, post: Post) -> Post:
        return await super().create_post(post)

    async def get_posts_by_thread(self, thread_id: str) -> list[Post]:
        return await super().get_posts_by_thread(thread_id)

    async def create_news_item(self, item: NewsItem) -> NewsItem:
        return await super().create_news_item(item)

    async def get_news_items_by_status(self, status: str) -> list[NewsItem]:
        return await super().get_news_items_by_status(status)

    async def update_news_item_status(
        self,
        item_id: str,
        status: str,
        promoted_thread_id: str | None = None,
    ) -> None:
        return await super().update_news_item_status(
            item_id, status, promoted_thread_id
        )

    async def get_news_item_by_url(self, url: str) -> NewsItem | None:
        return await super().get_news_item_by_url(url)

    async def get_storage_bytes(self) -> int:
        return await super().get_storage_bytes()

    async def update_agent_config(self, user_id: str, config: AgentConfig) -> None:
        return await super().update_agent_config(user_id, config)


@pytest.mark.anyio
async def test_repository_interface_default_bodies_are_noops() -> None:
    probe = InterfaceProbe()
    user = User(username="user")
    thread = Thread(title="thread", created_by=user.user_id)
    post = Post(thread_id=thread.thread_id, author_id=user.user_id, content="post")
    item = NewsItem(source="hn", title="story", url="https://example.com")

    assert await probe.init_tables() is None
    assert await probe.create_user(user) is None
    assert await probe.get_user(user.user_id) is None
    assert await probe.get_user_by_username(user.username) is None
    assert await probe.list_agents() is None
    assert await probe.create_thread(thread) is None
    assert await probe.get_thread(thread.thread_id) is None
    assert await probe.list_threads() is None
    assert await probe.update_thread_activity(thread.thread_id) is None
    assert await probe.create_post(post) is None
    assert await probe.get_posts_by_thread(thread.thread_id) is None
    assert await probe.create_news_item(item) is None
    assert await probe.get_news_items_by_status("new") is None
    assert await probe.update_news_item_status(item.item_id, "skipped") is None
    assert await probe.get_news_item_by_url(item.url) is None
    assert await probe.get_storage_bytes() is None
    assert (
        await probe.update_agent_config(
            user.user_id, AgentConfig(model_id="x", persona_name="x")
        )
        is None
    )


@pytest.mark.anyio
async def test_update_agent_config_overwrites_config_in_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = FakeResource()
    repo = make_repo(monkeypatch, resource)
    config = AgentConfig(model_id="old/model:free", persona_name="Nova")
    agent = User(username="Nova", is_agent=True, agent_config=config)
    await repo.create_user(agent)

    new_config = AgentConfig(
        model_id="nvidia/new-model:free",
        persona_name="Nova",
        response_probability=0.7,
    )
    await repo.update_agent_config(agent.user_id, new_config)

    calls = resource.tables["Users"].update_calls
    assert len(calls) == 1
    assert calls[0]["Key"] == {"user_id": agent.user_id}
    assert calls[0]["UpdateExpression"] == "SET agent_config = :c"
    stored = resource.tables["Users"].items[agent.user_id]["agent_config"]
    # Value stored should contain the new model ID
    assert stored["model_id"] == "nvidia/new-model:free"
