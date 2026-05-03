from __future__ import annotations

import asyncio
import logging
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fastapi import FastAPI
from starlette.responses import Response
from starlette.websockets import WebSocketDisconnect

from app import logging_config, main, middleware
from app.middleware import CorrelationIdMiddleware
from app.models.schemas import AgentConfig, User
from app.routes import websocket as websocket_routes
from app.routes.websocket import ConnectionManager


def test_setup_logging_configures_structlog_and_root_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(logging_config.settings, "log_level", "WARNING")

    logging_config.setup_logging()

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


@pytest.mark.anyio
async def test_correlation_id_middleware_adds_generated_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def app(scope: object, receive: object, send: object) -> None:
        return None

    monkeypatch.setattr(middleware.uuid, "uuid4", lambda: "generated-id")
    request = SimpleNamespace(
        headers={},
        method="GET",
        url=SimpleNamespace(path="/status"),
    )

    async def call_next(request: object) -> Response:
        return Response("ok")

    response = await CorrelationIdMiddleware(app).dispatch(request, call_next)  # type: ignore[arg-type]

    assert response.headers["X-Correlation-ID"] == "generated-id"


@pytest.mark.anyio
async def test_correlation_id_middleware_preserves_existing_header() -> None:
    async def app(scope: object, receive: object, send: object) -> None:
        return None

    request = SimpleNamespace(
        headers={"X-Correlation-ID": "client-id"},
        method="POST",
        url=SimpleNamespace(path="/submit"),
    )

    async def call_next(request: object) -> Response:
        return Response("created", status_code=201)

    response = await CorrelationIdMiddleware(app).dispatch(request, call_next)  # type: ignore[arg-type]

    assert response.status_code == 201
    assert response.headers["X-Correlation-ID"] == "client-id"


class FakeWebSocket:
    def __init__(self, fail_send: bool = False) -> None:
        self.accepted = False
        self.fail_send = fail_send
        self.sent: list[str] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, html: str) -> None:
        if self.fail_send:
            raise RuntimeError("closed")
        self.sent.append(html)


@pytest.mark.anyio
async def test_connection_manager_accepts_disconnects_and_removes_dead_sockets() -> None:
    manager = ConnectionManager()
    live = FakeWebSocket()
    dead = FakeWebSocket(fail_send=True)

    await manager.connect("thread-1", live)  # type: ignore[arg-type]
    await manager.connect("thread-1", dead)  # type: ignore[arg-type]
    await manager.broadcast("thread-1", "<p>hello</p>")
    manager.disconnect("thread-1", live)  # type: ignore[arg-type]
    manager.disconnect("thread-1", live)  # no-op for already removed

    assert live.accepted
    assert dead.accepted
    assert live.sent == ["<p>hello</p>"]
    assert manager._connections["thread-1"] == []


class ReceivingWebSocket(FakeWebSocket):
    async def receive_text(self) -> str:
        raise WebSocketDisconnect()


@pytest.mark.anyio
async def test_thread_websocket_connects_until_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ConnectionManager()
    monkeypatch.setattr(websocket_routes, "ws_manager", manager)
    websocket = ReceivingWebSocket()

    await websocket_routes.thread_ws(websocket, "thread-1")  # type: ignore[arg-type]

    assert websocket.accepted
    assert manager._connections["thread-1"] == []


@pytest.mark.anyio
async def test_news_loop_fetches_promotes_logs_errors_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[int] = []

    async def fake_sleep(seconds: int) -> None:
        sleeps.append(seconds)

    class Aggregator:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch_all(self) -> int:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary")
            if self.calls == 2:
                return 2
            raise asyncio.CancelledError()

    class NewsAgent:
        def __init__(self) -> None:
            self.promotions: list[int] = []

        async def promote_news(self, max_items: int = 5) -> list[SimpleNamespace]:
            self.promotions.append(max_items)
            return [SimpleNamespace(thread_id="thread-a"), SimpleNamespace(thread_id="thread-b")]

    class Repo:
        def __init__(self) -> None:
            self.thread_lookups: list[str] = []

        async def get_posts_by_thread(self, thread_id: str) -> list[SimpleNamespace]:
            self.thread_lookups.append(thread_id)
            return [
                SimpleNamespace(
                    post_id=f"post-{thread_id}",
                    thread_id=thread_id,
                    author_id="nova",
                    content="opening",
                )
            ]

    class DiscussionEngine:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def on_new_post(self, thread_id: str, post: SimpleNamespace) -> None:
            self.calls.append((thread_id, post.post_id))

    aggregator = Aggregator()
    news_agent = NewsAgent()
    repo = Repo()
    discussion_engine = DiscussionEngine()
    app = SimpleNamespace(
        state=SimpleNamespace(
            news_aggregator=aggregator,
            news_agent=news_agent,
            repo=repo,
            discussion_engine=discussion_engine,
        )
    )
    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)

    await main._news_loop(app)  # type: ignore[arg-type]

    assert aggregator.calls == 3
    assert news_agent.promotions == [5]
    assert repo.thread_lookups == ["thread-a", "thread-b"]
    assert discussion_engine.calls == [
        ("thread-a", "post-thread-a"),
        ("thread-b", "post-thread-b"),
    ]
    interval = main.settings.news_fetch_interval_minutes * 60
    assert sleeps == [10, interval, 30, interval]


@pytest.mark.anyio
async def test_lifespan_initializes_app_state_and_cancels_news_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Repo:
        async def init_tables(self) -> None:
            calls.append("init_tables")

    class Templates:
        def __init__(self, directory: str) -> None:
            self.directory = directory

    class DiscussionEngine:
        def __init__(self, **kwargs: Any) -> None:
            calls.append("discussion")
            self.kwargs = kwargs

    class Aggregator:
        def __init__(self, repo: Repo) -> None:
            calls.append("aggregator")
            self.repo = repo

    class NewsAgent:
        def __init__(self, user: User, repo: Repo) -> None:
            calls.append(f"news_agent:{user.username}")
            self.user = user
            self.repo = repo

    async def register_agents(repo: Repo) -> list[User]:
        calls.append("register_agents")
        return [
            User(
                username="Nova",
                is_agent=True,
                agent_config=AgentConfig(model_id="test", persona_name="Nova"),
            )
        ]

    async def never_finishes(app: FastAPI) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            calls.append("news_task_cancelled")
            raise

    async def never_finishes_model(app: FastAPI) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            calls.append("model_task_cancelled")
            raise

    repo = Repo()
    monkeypatch.setattr(main, "setup_logging", lambda: calls.append("logging"))
    monkeypatch.setattr(main.settings, "openrouter_api_key", "")
    monkeypatch.setattr(main, "DynamoLocalRepository", lambda: repo)
    monkeypatch.setattr(main, "Jinja2Templates", Templates)
    monkeypatch.setattr(main, "register_agents", register_agents)
    monkeypatch.setattr(main, "DiscussionEngine", DiscussionEngine)
    monkeypatch.setattr(main, "NewsAggregator", Aggregator)
    monkeypatch.setattr(main, "NewsAgent", NewsAgent)
    monkeypatch.setattr(main, "_news_loop", never_finishes)
    monkeypatch.setattr(main, "_model_discovery_loop", never_finishes_model)
    app = FastAPI()

    async with main.lifespan(app):
        await asyncio.sleep(0)
        assert app.state.repo is repo
        assert isinstance(app.state.templates, Templates)
        assert app.state.ws_manager is main.ws_manager
        assert isinstance(app.state.discussion_engine, DiscussionEngine)
        assert isinstance(app.state.news_aggregator, Aggregator)
        assert isinstance(app.state.news_agent, NewsAgent)

    assert calls == [
        "logging",
        "init_tables",
        "register_agents",
        "discussion",
        "aggregator",
        "news_agent:Nova",
        "news_task_cancelled",
        "model_task_cancelled",
    ]


def test_docker_compose_dynamodb_local_can_write_to_volume() -> None:
    compose_path = pathlib.Path(__file__).parent.parent.parent / "docker-compose.yml"
    config = yaml.safe_load(compose_path.read_text())
    dynamo = config["services"]["dynamodb-local"]

    assert dynamo.get("user") == "root", (
        "dynamodb-local must run as root so it can write to the Docker named volume; "
        "without this the container crashes on startup with SQLiteException [14]"
    )
    assert "-dbPath" in dynamo["command"], (
        "dynamodb-local must use -dbPath for persistent storage (not -inMemory)"
    )
    volume_refs = [str(v) for v in dynamo.get("volumes", [])]
    assert any("dynamodb-data" in v for v in volume_refs), (
        "dynamodb-local must mount the dynamodb-data volume at the -dbPath location"
    )
