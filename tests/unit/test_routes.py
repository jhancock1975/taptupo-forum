from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.responses import HTMLResponse

from app.auth import routes as auth_routes
from app.auth.utils import create_access_token, decode_access_token, hash_password
from app.models.schemas import AgentConfig, Post, Thread, User
from app.routes import threads as thread_routes
from app.routes.threads import _discovery_status_html, _render_discovery_log, _storage_gauge_svg


class CapturingTemplates:
    def __init__(self) -> None:
        self.responses: list[HTMLResponse] = []
        self.rendered: list[tuple[str, dict[str, Any]]] = []

    def TemplateResponse(
        self,
        request: object,
        template_name: str,
        context: dict[str, Any],
        status_code: int = 200,
    ) -> HTMLResponse:
        response = HTMLResponse(template_name, status_code=status_code)
        response.template_name = template_name  # type: ignore[attr-defined]
        response.context = context  # type: ignore[attr-defined]
        self.responses.append(response)
        return response

    def get_template(self, template_name: str) -> object:
        parent = self

        class Template:
            def render(self, **context: Any) -> str:
                parent.rendered.append((template_name, context))
                return f"{template_name}:{context['post'].content}"

        return Template()


class FakeRequest:
    def __init__(
        self,
        *,
        repo: object,
        templates: CapturingTemplates,
        cookies: dict[str, str] | None = None,
        ws_manager: object | None = None,
        discussion_engine: object | None = None,
    ) -> None:
        self.cookies = cookies or {}
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                repo=repo,
                templates=templates,
                ws_manager=ws_manager,
                discussion_engine=discussion_engine,
            )
        )


class AuthRepo:
    def __init__(self, users: dict[str, User] | None = None) -> None:
        self.users = users or {}
        self.created: list[User] = []

    async def get_user_by_username(self, username: str) -> User | None:
        return self.users.get(username)

    async def create_user(self, user: User) -> User:
        self.created.append(user)
        self.users[user.username] = user
        return user


@pytest.mark.anyio
async def test_auth_pages_render_without_errors() -> None:
    templates = CapturingTemplates()
    request = FakeRequest(repo=AuthRepo(), templates=templates)

    login = await auth_routes.login_page(request)  # type: ignore[arg-type]
    register = await auth_routes.register_page(request)  # type: ignore[arg-type]

    assert login.status_code == 200
    assert login.template_name == "login.html"  # type: ignore[attr-defined]
    assert login.context == {"error": None}  # type: ignore[attr-defined]
    assert register.status_code == 200
    assert register.template_name == "register.html"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_login_rejects_unknown_or_bad_credentials() -> None:
    templates = CapturingTemplates()
    request = FakeRequest(repo=AuthRepo(), templates=templates)

    response = await auth_routes.login(  # type: ignore[arg-type]
        request,
        username="missing",
        password="secret",
    )

    assert response.status_code == 401
    assert response.template_name == "login.html"  # type: ignore[attr-defined]
    assert response.context == {"error": "Invalid username or password"}  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_login_sets_access_token_cookie_for_valid_credentials() -> None:
    user = User(username="alice", password_hash=hash_password("correct horse"))
    request = FakeRequest(repo=AuthRepo({"alice": user}), templates=CapturingTemplates())

    response = await auth_routes.login(  # type: ignore[arg-type]
        request,
        username="alice",
        password="correct horse",
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert "access_token=" in response.headers["set-cookie"]
    assert "HttpOnly" in response.headers["set-cookie"]


@pytest.mark.anyio
async def test_register_validates_username_password_and_duplicates() -> None:
    existing = User(username="taken", password_hash="hash")
    repo = AuthRepo({"taken": existing})
    templates = CapturingTemplates()
    request = FakeRequest(repo=repo, templates=templates)

    bad_username = await auth_routes.register(  # type: ignore[arg-type]
        request,
        username="no",
        password="password1",
        password_confirm="password1",
    )
    short_password = await auth_routes.register(  # type: ignore[arg-type]
        request,
        username="valid_user",
        password="short",
        password_confirm="short",
    )
    mismatch = await auth_routes.register(  # type: ignore[arg-type]
        request,
        username="valid_user",
        password="password1",
        password_confirm="password2",
    )
    duplicate = await auth_routes.register(  # type: ignore[arg-type]
        request,
        username="taken",
        password="password1",
        password_confirm="password1",
    )

    assert bad_username.status_code == 400
    assert short_password.status_code == 400
    assert mismatch.status_code == 400
    assert duplicate.status_code == 409
    assert [response.context["error"] for response in templates.responses] == [  # type: ignore[attr-defined]
        "Username must be 3-30 alphanumeric characters or underscores",
        "Password must be at least 8 characters",
        "Passwords do not match",
        "Username already taken",
    ]


@pytest.mark.anyio
async def test_register_creates_user_and_logs_them_in() -> None:
    repo = AuthRepo()
    request = FakeRequest(repo=repo, templates=CapturingTemplates())

    response = await auth_routes.register(  # type: ignore[arg-type]
        request,
        username="new_user",
        password="password1",
        password_confirm="password1",
    )

    assert response.status_code == 303
    assert repo.created[0].username == "new_user"
    assert repo.created[0].password_hash != "password1"
    assert "access_token=" in response.headers["set-cookie"]


@pytest.mark.anyio
async def test_logout_deletes_access_token_cookie() -> None:
    response = await auth_routes.logout()

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert "access_token=" in response.headers["set-cookie"]
    assert "Max-Age=0" in response.headers["set-cookie"]


class ForumRepo:
    def __init__(self) -> None:
        self.users: dict[str, User] = {}
        self.threads: dict[str, Thread] = {}
        self.posts: dict[str, list[Post]] = {}
        self.agents: list[User] = []
        self.created_threads: list[Thread] = []
        self.created_posts: list[Post] = []
        self.activity_updates: list[str] = []

    async def list_threads(self, limit: int = 50) -> list[Thread]:
        return list(self.threads.values())[:limit]

    async def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    async def get_thread(self, thread_id: str) -> Thread | None:
        return self.threads.get(thread_id)

    async def get_posts_by_thread(self, thread_id: str) -> list[Post]:
        return self.posts.get(thread_id, [])

    async def create_post(self, post: Post) -> Post:
        self.created_posts.append(post)
        self.posts.setdefault(post.thread_id, []).append(post)
        return post

    async def update_thread_activity(self, thread_id: str) -> None:
        self.activity_updates.append(thread_id)

    async def create_thread(self, thread: Thread) -> Thread:
        self.created_threads.append(thread)
        self.threads[thread.thread_id] = thread
        return thread

    async def list_agents(self) -> list[User]:
        return self.agents

    async def get_storage_bytes(self) -> int:
        return 0

    async def update_agent_config(self, user_id: str, config: Any) -> None:
        pass


class RecordingWSManager:
    def __init__(self) -> None:
        self.broadcasts: list[tuple[str, str]] = []

    async def broadcast(self, thread_id: str, html: str) -> None:
        self.broadcasts.append((thread_id, html))


class RecordingDiscussionEngine:
    def __init__(self) -> None:
        self.posts: list[tuple[str, Post]] = []

    async def on_new_post(self, thread_id: str, post: Post) -> None:
        self.posts.append((thread_id, post))


def request_for_forum(
    repo: ForumRepo,
    *,
    token: str | None = None,
    templates: CapturingTemplates | None = None,
    ws_manager: RecordingWSManager | None = None,
    discussion_engine: RecordingDiscussionEngine | None = None,
) -> FakeRequest:
    cookies = {"access_token": token} if token else {}
    return FakeRequest(
        repo=repo,
        templates=templates or CapturingTemplates(),
        cookies=cookies,
        ws_manager=ws_manager or RecordingWSManager(),
        discussion_engine=discussion_engine or RecordingDiscussionEngine(),
    )


def user_token(user: User) -> str:
    return create_access_token(user.user_id, user.username)


def test_current_user_decodes_cookie() -> None:
    user = User(username="alice")

    assert thread_routes._current_user(SimpleNamespace(cookies={})) is None  # type: ignore[arg-type]
    assert thread_routes._current_user(  # type: ignore[arg-type]
        SimpleNamespace(cookies={"access_token": user_token(user)})
    ) == {"user_id": user.user_id, "username": "alice"}
    assert decode_access_token(create_access_token("id-only", "bob")) == {
        "user_id": "id-only",
        "username": "bob",
    }


def test_decode_access_token_rejects_missing_claims() -> None:
    from jose import jwt

    from app.auth import utils

    token = jwt.encode({"sub": "user-without-name"}, utils.settings.secret_key, algorithm=utils.ALGORITHM)

    assert decode_access_token(token) is None


@pytest.mark.anyio
async def test_home_renders_threads_with_author_names() -> None:
    repo = ForumRepo()
    alice = User(user_id="u1", username="alice")
    repo.users[alice.user_id] = alice
    repo.threads["t1"] = Thread(thread_id="t1", title="Welcome", created_by=alice.user_id)
    repo.threads["t2"] = Thread(thread_id="t2", title="Missing author", created_by="missing")
    templates = CapturingTemplates()
    request = request_for_forum(repo, token=user_token(alice), templates=templates)

    response = await thread_routes.home(request)  # type: ignore[arg-type]

    assert response.template_name == "home.html"  # type: ignore[attr-defined]
    assert response.context["threads"] == list(repo.threads.values())  # type: ignore[attr-defined]
    assert response.context["authors"] == {alice.user_id: "alice"}  # type: ignore[attr-defined]
    assert response.context["user"]["username"] == "alice"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_thread_view_renders_404_for_missing_thread() -> None:
    user = User(username="alice")
    request = request_for_forum(ForumRepo(), token=user_token(user))

    response = await thread_routes.thread_view(request, "missing")  # type: ignore[arg-type]

    assert response.status_code == 404
    assert response.template_name == "404.html"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_thread_view_renders_posts_and_authors() -> None:
    repo = ForumRepo()
    alice = User(user_id="u1", username="alice")
    bob = User(user_id="u2", username="bob")
    repo.users = {alice.user_id: alice, bob.user_id: bob}
    repo.threads["t1"] = Thread(thread_id="t1", title="Welcome", created_by=alice.user_id)
    repo.posts["t1"] = [Post(thread_id="t1", author_id=bob.user_id, content="hello")]
    request = request_for_forum(repo, token=user_token(alice))

    response = await thread_routes.thread_view(request, "t1")  # type: ignore[arg-type]

    assert response.template_name == "thread.html"  # type: ignore[attr-defined]
    assert response.context["thread"] == repo.threads["t1"]  # type: ignore[attr-defined]
    assert response.context["posts"] == repo.posts["t1"]  # type: ignore[attr-defined]
    assert response.context["authors"] == {alice.user_id: alice, bob.user_id: bob}  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_create_post_redirects_anonymous_users() -> None:
    response = await thread_routes.create_post(  # type: ignore[arg-type]
        request_for_forum(ForumRepo()),
        thread_id="t1",
        content="hello",
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/auth/login"


@pytest.mark.anyio
async def test_create_post_renders_404_for_missing_thread() -> None:
    user = User(username="alice")
    request = request_for_forum(ForumRepo(), token=user_token(user))

    response = await thread_routes.create_post(  # type: ignore[arg-type]
        request,
        thread_id="missing",
        content="hello",
    )

    assert response.status_code == 404
    assert response.template_name == "404.html"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_create_post_persists_broadcasts_and_triggers_agents() -> None:
    repo = ForumRepo()
    alice = User(user_id="u1", username="alice")
    repo.users[alice.user_id] = alice
    repo.threads["t1"] = Thread(thread_id="t1", title="Welcome", created_by=alice.user_id)
    ws = RecordingWSManager()
    discussion = RecordingDiscussionEngine()
    templates = CapturingTemplates()
    request = request_for_forum(
        repo,
        token=user_token(alice),
        templates=templates,
        ws_manager=ws,
        discussion_engine=discussion,
    )

    response = await thread_routes.create_post(  # type: ignore[arg-type]
        request,
        thread_id="t1",
        content="  hello  ",
        parent_post_id="parent-1",
    )

    post = repo.created_posts[0]
    assert response.status_code == 303
    assert response.headers["location"] == "/thread/t1"
    assert post.content == "hello"
    assert post.parent_post_id == "parent-1"
    assert repo.activity_updates == ["t1"]
    assert ws.broadcasts == [("t1", "fragments/post.html:hello")]
    assert discussion.posts == [("t1", post)]
    assert templates.rendered[0][1]["author"] == alice


@pytest.mark.anyio
async def test_new_thread_page_requires_login_and_renders_form() -> None:
    user = User(username="alice")
    anonymous = await thread_routes.new_thread_page(request_for_forum(ForumRepo()))  # type: ignore[arg-type]
    authenticated = await thread_routes.new_thread_page(  # type: ignore[arg-type]
        request_for_forum(ForumRepo(), token=user_token(user))
    )

    assert anonymous.status_code == 303
    assert anonymous.headers["location"] == "/auth/login"
    assert authenticated.status_code == 200
    assert authenticated.template_name == "new_thread.html"  # type: ignore[attr-defined]
    assert authenticated.context["error"] is None  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_create_thread_requires_login_and_valid_content() -> None:
    user = User(username="alice")
    repo = ForumRepo()

    anonymous = await thread_routes.create_thread(  # type: ignore[arg-type]
        request_for_forum(repo),
        title="Title",
        content="Content",
    )
    invalid = await thread_routes.create_thread(  # type: ignore[arg-type]
        request_for_forum(repo, token=user_token(user)),
        title=" ",
        content="Content",
    )

    assert anonymous.status_code == 303
    assert anonymous.headers["location"] == "/auth/login"
    assert invalid.status_code == 400
    assert invalid.template_name == "new_thread.html"  # type: ignore[attr-defined]
    assert invalid.context["error"] == "Title and content are required"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_create_thread_persists_first_post_and_triggers_agents() -> None:
    user = User(user_id="u1", username="alice")
    repo = ForumRepo()
    discussion = RecordingDiscussionEngine()
    request = request_for_forum(repo, token=user_token(user), discussion_engine=discussion)

    response = await thread_routes.create_thread(  # type: ignore[arg-type]
        request,
        title="  A new topic  ",
        content="  First post  ",
    )

    thread = repo.created_threads[0]
    post = repo.created_posts[0]
    assert response.status_code == 303
    assert response.headers["location"] == f"/thread/{thread.thread_id}"
    assert thread.title == "A new topic"
    assert thread.created_by == user.user_id
    assert post.thread_id == thread.thread_id
    assert post.content == "First post"
    assert discussion.posts == [(thread.thread_id, post)]


@pytest.mark.anyio
async def test_agent_directory_renders_agents() -> None:
    repo = ForumRepo()
    user = User(username="alice")
    agent = User(
        username="Nova",
        is_agent=True,
        agent_config=AgentConfig(model_id="test", persona_name="Nova"),
    )
    repo.agents = [agent]
    request = request_for_forum(repo, token=user_token(user))

    response = await thread_routes.agent_directory(request)  # type: ignore[arg-type]

    assert response.template_name == "agents.html"  # type: ignore[attr-defined]
    assert response.context["agents"] == [agent]  # type: ignore[attr-defined]
    assert response.context["user"]["username"] == "alice"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_db_usage_returns_svg_gauge() -> None:
    class StorageRepo(ForumRepo):
        def __init__(self, used_bytes: int) -> None:
            super().__init__()
            self._used = used_bytes

        async def get_storage_bytes(self) -> int:
            return self._used

    # Empty DB: 0 bytes used → 0%
    request = request_for_forum(StorageRepo(0))
    response = await thread_routes.db_usage(request)  # type: ignore[arg-type]
    assert response.status_code == 200
    assert b"<svg" in response.body
    assert b"0%" in response.body
    assert b"#22c55e" in response.body  # green

    # Half full: 50 MB → ~50%, green
    request = request_for_forum(StorageRepo(50 * 1024 * 1024))
    response = await thread_routes.db_usage(request)  # type: ignore[arg-type]
    assert b"50%" in response.body
    assert b"#22c55e" in response.body

    # Warning zone: 80 MB → 80%, amber
    request = request_for_forum(StorageRepo(80 * 1024 * 1024))
    response = await thread_routes.db_usage(request)  # type: ignore[arg-type]
    assert b"80%" in response.body
    assert b"#f59e0b" in response.body

    # Critical zone: 95 MB → 95%, red
    request = request_for_forum(StorageRepo(95 * 1024 * 1024))
    response = await thread_routes.db_usage(request)  # type: ignore[arg-type]
    assert b"95%" in response.body
    assert b"#ef4444" in response.body

    # Over limit: 110 MB → capped at 100%
    request = request_for_forum(StorageRepo(110 * 1024 * 1024))
    response = await thread_routes.db_usage(request)  # type: ignore[arg-type]
    assert b"100%" in response.body


def test_storage_gauge_svg_shows_gb_units_for_large_quota() -> None:
    svg = _storage_gauge_svg(0, 2 * 1024 * 1024 * 1024, "db-gauge-svg", "S3 media")
    assert "GB" in svg
    assert "S3 media storage:" in svg


def test_storage_gauge_svg_shows_mb_units_for_small_quota() -> None:
    svg = _storage_gauge_svg(50 * 1024 * 1024, 100 * 1024 * 1024, "db-gauge-svg", "Database")
    assert "MB" in svg
    assert "Database storage:" in svg


@pytest.mark.anyio
async def test_s3_usage_endpoint_returns_svg(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.storage.s3 as s3_mod

    monkeypatch.setattr(s3_mod, "get_storage_bytes", AsyncMock(return_value=100 * 1024 * 1024))

    request = request_for_forum(ForumRepo())
    response = await thread_routes.s3_usage(request)  # type: ignore[arg-type]

    assert response.status_code == 200
    assert b"<svg" in response.body
    assert b"S3 media" in response.body


# ── Discovery log route ────────────────────────────────────────────────────────


def test_render_discovery_log_empty_state() -> None:
    html = _render_discovery_log([])
    assert "discovery-empty" in html
    assert "No discovery runs" in html


def test_render_discovery_log_models_fetched_event() -> None:
    events = [
        {
            "ts": "2026-05-02 11:00:00 UTC",
            "event": "models_fetched",
            "data": {"count": 5, "model_ids": ["openai/gpt-oss-20b:free", "openrouter/optimus-alpha"]},
        }
    ]
    html = _render_discovery_log(events)
    assert "📋" in html
    assert "5 models" in html
    assert "openrouter/optimus-alpha" in html
    assert "discovery-model-pill" in html


def test_render_discovery_log_agent_assigned_text_model() -> None:
    events = [
        {
            "ts": "2026-05-02 11:00:01 UTC",
            "event": "agent_assigned",
            "data": {
                "agent": "Nova",
                "model_label": "OpenAI · GPT-OSS 20B",
                "output_modality": "text",
            },
        }
    ]
    html = _render_discovery_log(events)
    assert "Nova" in html
    assert "OpenAI · GPT-OSS 20B" in html
    # No modality badge for text
    assert "discovery-modality-badge" not in html


def test_render_discovery_log_agent_assigned_image_model() -> None:
    events = [
        {
            "ts": "2026-05-02 11:00:02 UTC",
            "event": "agent_assigned",
            "data": {
                "agent": "Pixel",
                "model_label": "FLUX.1",
                "output_modality": "image",
            },
        }
    ]
    html = _render_discovery_log(events)
    assert "Pixel" in html
    assert "discovery-modality-badge" in html
    assert "image" in html


def test_render_discovery_log_job_complete_event() -> None:
    events = [
        {"ts": "2026-05-02 11:00:05 UTC", "event": "job_complete", "data": {"updated": 6}}
    ]
    html = _render_discovery_log(events)
    assert "6 agents updated" in html
    assert "🏁" in html


def test_render_discovery_log_error_row_class() -> None:
    events = [
        {"ts": "2026-05-02 11:00:00 UTC", "event": "fetch_failed", "data": {}}
    ]
    html = _render_discovery_log(events)
    assert "discovery-row-error" in html


def test_render_discovery_log_agent_failed_event() -> None:
    events = [
        {
            "ts": "2026-05-02 11:00:00 UTC",
            "event": "agent_failed",
            "data": {"agent": "Ember", "model_id": "bad/model"},
        }
    ]
    html = _render_discovery_log(events)
    assert "Ember" in html
    assert "bad/model" in html


def test_render_discovery_log_newest_first() -> None:
    events = [
        {"ts": "2026-05-02 11:00:00 UTC", "event": "job_started", "data": {}},
        {"ts": "2026-05-02 11:00:10 UTC", "event": "job_complete", "data": {"updated": 1}},
    ]
    html = _render_discovery_log(events)
    # job_complete (newer) should appear before job_started in the output
    assert html.index("🏁") < html.index("🔍")


@pytest.mark.anyio
async def test_discovery_log_endpoint_returns_html(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    log = [
        {"ts": "2026-05-02 11:00:00 UTC", "event": "job_complete", "data": {"updated": 3}}
    ]

    request = request_for_forum(ForumRepo())
    request.app = SimpleNamespace(state=SimpleNamespace(discovery_log=log))  # type: ignore[attr-defined]

    response = await thread_routes.discovery_log(request)  # type: ignore[arg-type]
    assert response.status_code == 200
    assert b"3 agents updated" in response.body


@pytest.mark.anyio
async def test_discovery_log_endpoint_returns_empty_state_when_no_log() -> None:
    from types import SimpleNamespace

    request = request_for_forum(ForumRepo())
    # app.state has no discovery_log attribute
    request.app = SimpleNamespace(state=SimpleNamespace())  # type: ignore[attr-defined]

    response = await thread_routes.discovery_log(request)  # type: ignore[arg-type]
    assert response.status_code == 200
    assert b"No discovery runs" in response.body


# ── Discovery status nav pill ─────────────────────────────────────────────────


def test_discovery_status_idle_when_no_log() -> None:
    html = _discovery_status_html([])
    assert "discovery-status-idle" in html
    assert "⏳" in html
    assert 'href="/agents"' in html


def test_discovery_status_ok_after_successful_run() -> None:
    log = [
        {"ts": "2026-05-02 11:00:00 UTC", "event": "job_started", "data": {}},
        {"ts": "2026-05-02 11:00:01 UTC", "event": "models_fetched", "data": {"count": 12, "model_ids": []}},
        {"ts": "2026-05-02 11:00:05 UTC", "event": "job_complete", "data": {"updated": 6}},
    ]
    html = _discovery_status_html(log)
    assert "discovery-status-ok" in html
    assert "🤖" in html
    assert "6 agents" in html
    assert "12 free models" in html


def test_discovery_status_error_after_fetch_failed() -> None:
    log = [
        {"ts": "2026-05-02 11:00:00 UTC", "event": "job_started", "data": {}},
        {"ts": "2026-05-02 11:00:01 UTC", "event": "fetch_failed", "data": {}},
    ]
    html = _discovery_status_html(log)
    assert "discovery-status-error" in html
    assert "⚠️" in html


def test_discovery_status_running_when_started_not_completed() -> None:
    log = [
        {"ts": "2026-05-02 11:00:00 UTC", "event": "job_complete", "data": {"updated": 3}},
        {"ts": "2026-05-02 11:00:10 UTC", "event": "job_started", "data": {}},
    ]
    html = _discovery_status_html(log)
    assert "discovery-status-running" in html
    assert "🔄" in html


@pytest.mark.anyio
async def test_discovery_status_endpoint_returns_html() -> None:
    from types import SimpleNamespace

    log = [
        {"ts": "2026-05-02 11:00:00 UTC", "event": "job_complete", "data": {"updated": 4}},
    ]
    request = request_for_forum(ForumRepo())
    request.app = SimpleNamespace(state=SimpleNamespace(discovery_log=log))  # type: ignore[attr-defined]

    response = await thread_routes.discovery_status(request)  # type: ignore[arg-type]
    assert response.status_code == 200
    assert b"discovery-status" in response.body
    assert b"/agents" in response.body
