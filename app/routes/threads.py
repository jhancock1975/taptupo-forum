from __future__ import annotations

import math

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth.utils import decode_access_token
from app.config import settings
from app.models.schemas import Post, Thread

router = APIRouter(tags=["threads"])


def _current_user(request: Request) -> dict[str, str] | None:
    token = request.cookies.get("access_token")
    if not token:
        return None
    return decode_access_token(token)


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    repo = request.app.state.repo
    user = _current_user(request)
    threads = await repo.list_threads(limit=50)

    # build author lookup
    author_ids = {t.created_by for t in threads}
    authors: dict[str, str] = {}
    for uid in author_ids:
        u = await repo.get_user(uid)
        if u:
            authors[uid] = u.username

    return templates.TemplateResponse(
        request,
        "home.html",
        {"threads": threads, "user": user, "authors": authors},
    )


@router.get("/thread/{thread_id}", response_class=HTMLResponse)
async def thread_view(request: Request, thread_id: str) -> HTMLResponse:
    templates = request.app.state.templates
    repo = request.app.state.repo
    user = _current_user(request)

    thread = await repo.get_thread(thread_id)
    if not thread:
        return templates.TemplateResponse(
            request, "404.html", {"user": user}, status_code=404
        )

    posts = await repo.get_posts_by_thread(thread_id)

    # build author lookup for all posts + thread creator
    author_ids = {p.author_id for p in posts} | {thread.created_by}
    authors: dict[str, object] = {}
    for uid in author_ids:
        u = await repo.get_user(uid)
        if u:
            authors[uid] = u

    return templates.TemplateResponse(
        request,
        "thread.html",
        {"thread": thread, "posts": posts, "user": user, "authors": authors},
    )


@router.post("/thread/{thread_id}/post", response_model=None)
async def create_post(
    request: Request,
    thread_id: str,
    content: str = Form(...),
    parent_post_id: str = Form(default=""),
) -> RedirectResponse | HTMLResponse:
    templates = request.app.state.templates
    repo = request.app.state.repo
    user = _current_user(request)

    if not user:
        return RedirectResponse(url="/auth/login", status_code=303)

    thread = await repo.get_thread(thread_id)
    if not thread:
        return templates.TemplateResponse(
            request, "404.html", {"user": user}, status_code=404
        )

    post = Post(
        thread_id=thread_id,
        author_id=user["user_id"],
        content=content.strip(),
        parent_post_id=parent_post_id or None,
    )
    await repo.create_post(post)
    await repo.update_thread_activity(thread_id)

    # Broadcast via WebSocket
    ws_manager = request.app.state.ws_manager
    author = await repo.get_user(user["user_id"])
    html_fragment = templates.get_template("fragments/post.html").render(
        post=post, author=author, user=user
    )
    await ws_manager.broadcast(thread_id, html_fragment)

    discussion_engine = request.app.state.discussion_engine
    await discussion_engine.on_new_post(thread_id, post)

    return RedirectResponse(url=f"/thread/{thread_id}", status_code=303)


@router.get("/new-thread", response_class=HTMLResponse, response_model=None)
async def new_thread_page(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    user = _current_user(request)
    if not user:
        return RedirectResponse(url="/auth/login", status_code=303)
    return templates.TemplateResponse(
        request, "new_thread.html", {"user": user, "error": None}
    )


@router.post("/new-thread", response_model=None)
async def create_thread(
    request: Request,
    title: str = Form(...),
    content: str = Form(...),
) -> RedirectResponse | HTMLResponse:
    templates = request.app.state.templates
    repo = request.app.state.repo
    user = _current_user(request)

    if not user:
        return RedirectResponse(url="/auth/login", status_code=303)

    title = title.strip()
    content = content.strip()

    if not title or not content:
        return templates.TemplateResponse(
            request,
            "new_thread.html",
            {"user": user, "error": "Title and content are required"},
            status_code=400,
        )

    thread = Thread(
        title=title,
        created_by=user["user_id"],
    )
    await repo.create_thread(thread)

    first_post = Post(
        thread_id=thread.thread_id,
        author_id=user["user_id"],
        content=content,
    )
    await repo.create_post(first_post)

    discussion_engine = request.app.state.discussion_engine
    await discussion_engine.on_new_post(thread.thread_id, first_post)

    return RedirectResponse(url=f"/thread/{thread.thread_id}", status_code=303)


@router.get("/agents", response_class=HTMLResponse)
async def agent_directory(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    repo = request.app.state.repo
    user = _current_user(request)
    agents = await repo.list_agents()
    return templates.TemplateResponse(
        request, "agents.html", {"agents": agents, "user": user}
    )


@router.get("/api/db-usage", response_class=HTMLResponse)
async def db_usage(request: Request) -> HTMLResponse:
    """Return an SVG gauge showing DynamoDB storage usage vs the configured limit."""
    repo = request.app.state.repo
    limit_bytes = settings.db_size_limit_mb * 1024 * 1024
    used_bytes = await repo.get_storage_bytes()
    svg = _storage_gauge_svg(used_bytes, limit_bytes, "db-gauge-svg", "Database")
    return HTMLResponse(content=svg)


def _storage_gauge_svg(
    used_bytes: int, limit_bytes: int, css_class: str, label_prefix: str
) -> str:
    """Shared donut-gauge SVG renderer."""
    pct = min(100.0, used_bytes / limit_bytes * 100) if limit_bytes > 0 else 0.0
    circ = 2 * math.pi * 14
    filled = round(pct / 100 * circ, 2)
    rest = round(circ - filled, 2)

    if pct < 70:
        color = "#22c55e"
    elif pct < 90:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    used_mb = round(used_bytes / (1024 * 1024), 2)
    limit_val = limit_bytes / (1024 * 1024)
    unit = "MB"
    if limit_val >= 1024:
        limit_val /= 1024
        used_mb = round(used_bytes / (1024**3), 2)
        unit = "GB"

    label = f"{used_mb} {unit} / {limit_val:.0f} {unit}"
    pct_str = f"{pct:.0f}%"

    return (
        f'<svg width="40" height="40" viewBox="0 0 36 36" class="{css_class}">'
        f"<title>{label_prefix} storage: {label}</title>"
        f'<circle cx="18" cy="18" r="14" fill="none"'
        f' stroke="var(--db-gauge-track)" stroke-width="3.5"/>'
        f'<circle cx="18" cy="18" r="14" fill="none"'
        f' stroke="{color}" stroke-width="3.5"'
        f' stroke-dasharray="{filled} {rest}"'
        f' transform="rotate(-90 18 18)"/>'
        f'<text x="18" y="21.5" text-anchor="middle"'
        f' font-size="7" font-weight="600" fill="{color}">{pct_str}</text>'
        f"</svg>"
    )


@router.get("/api/s3-usage", response_class=HTMLResponse)
async def s3_usage(request: Request) -> HTMLResponse:
    """Return an SVG gauge showing S3 media storage usage vs the configured quota."""
    from app.storage import s3 as s3_store

    used_bytes = await s3_store.get_storage_bytes()
    limit_bytes = int(settings.s3_quota_gb * 1024 * 1024 * 1024)
    svg = _storage_gauge_svg(used_bytes, limit_bytes, "db-gauge-svg", "S3 media")
    return HTMLResponse(content=svg)


# ── Event labels for the discovery log ───────────────────────────────────────

_EVENT_LABELS: dict[str, str] = {
    "job_started": "🔍 Discovery job started",
    "models_fetched": "📋 Models fetched from OpenRouter",
    "fetch_failed": "❌ Failed to fetch models",
    "no_free_models": "⚠️ No free models found",
    "no_agent_pairs": "⚠️ No agent personas to assign",
    "no_selected_models": "⚠️ Model selection returned empty",
    "agent_assigned": "✅ Model assigned",
    "agent_failed": "❌ Agent update failed",
    "job_complete": "🏁 Job complete",
}


def _render_discovery_log(events: list[dict]) -> str:
    """Render discovery log events as an HTML fragment."""
    if not events:
        return (
            '<p class="discovery-empty">'
            "No discovery runs recorded yet. The background job will run shortly after startup."
            "</p>"
        )

    rows: list[str] = []
    for ev in reversed(events):  # newest first
        event = ev.get("event", "")
        label = _EVENT_LABELS.get(event, event)
        ts = ev.get("ts", "")
        data = ev.get("data", {})

        detail_parts: list[str] = []
        if event == "models_fetched":
            count = data.get("count", 0)
            ids = data.get("model_ids", [])
            detail_parts.append(f"{count} models")
            if ids:
                pills = "".join(
                    f'<span class="discovery-model-pill">{mid}</span>' for mid in ids
                )
                detail_parts.append(f'<div class="discovery-model-list">{pills}</div>')
        elif event == "agent_assigned":
            agent = data.get("agent", "")
            model_label = data.get("model_label") or data.get("model_id", "")
            modality = data.get("output_modality", "text")
            mod_badge = (
                f'<span class="discovery-modality-badge">{modality}</span>'
                if modality != "text"
                else ""
            )
            detail_parts.append(
                f"<strong>{agent}</strong> → "
                f'<span class="discovery-model-name">{model_label}</span>{mod_badge}'
            )
        elif event == "agent_failed":
            detail_parts.append(
                f'<strong>{data.get("agent", "")}</strong> '
                f'({data.get("model_id", "")})'
            )
        elif event == "job_complete":
            detail_parts.append(f'{data.get("updated", 0)} agents updated')
        elif event == "fetch_failed" or event == "job_started":
            pass  # label is enough

        detail_html = " ".join(detail_parts)
        row_class = "discovery-row-error" if "fail" in event or "error" in event else ""
        rows.append(
            f'<div class="discovery-row {row_class}">'
            f'<span class="discovery-ts">{ts}</span>'
            f'<span class="discovery-label">{label}</span>'
            f'<span class="discovery-detail">{detail_html}</span>'
            f"</div>"
        )

    return "\n".join(rows)


@router.get("/api/discovery-log", response_class=HTMLResponse)
async def discovery_log(request: Request) -> HTMLResponse:
    """Return an HTML fragment of recent model discovery activity."""
    log: list[dict] = getattr(request.app.state, "discovery_log", [])
    html = _render_discovery_log(log)
    return HTMLResponse(content=html)


def _discovery_status_html(log: list[dict]) -> str:
    """Return a compact nav pill summarising the latest discovery run."""
    # Find the most recent job_complete / fetch_failed event
    last_complete = next(
        (
            e
            for e in reversed(log)
            if e.get("event") in ("job_complete", "fetch_failed")
        ),
        None,
    )
    # Check whether a job is currently in-flight (started but not finished)
    last_started_idx = next(
        (i for i, e in enumerate(reversed(log)) if e.get("event") == "job_started"),
        None,
    )
    last_complete_idx = next(
        (
            i
            for i, e in enumerate(reversed(log))
            if e.get("event") in ("job_complete", "fetch_failed")
        ),
        None,
    )
    running = last_started_idx is not None and (
        last_complete_idx is None or last_started_idx < last_complete_idx
    )

    if running:
        icon, label, css = "🔄", "Discovering…", "discovery-status-running"
    elif last_complete is None:
        icon, label, css = "⏳", "Model discovery", "discovery-status-idle"
    elif last_complete["event"] == "fetch_failed":
        ts = last_complete.get("ts", "")
        icon, label, css = "⚠️", f"Discovery failed · {ts}", "discovery-status-error"
    else:
        data = last_complete.get("data", {})
        updated = data.get("updated", 0)
        # Count distinct models seen in this run by scanning back to the last job_started
        model_count = next(
            (
                e["data"].get("count", 0)
                for e in reversed(log)
                if e.get("event") == "models_fetched"
            ),
            0,
        )
        ts = last_complete.get("ts", "")
        label = f"{updated} agents · {model_count} free models"
        if ts:
            label += f" · {ts}"
        icon, css = "🤖", "discovery-status-ok"

    return (
        f'<a href="/agents" class="discovery-status-pill {css}" '
        f'title="Model discovery: {label}">'
        f'<span class="discovery-status-icon">{icon}</span>'
        f'<span class="discovery-status-label">{label}</span>'
        f"</a>"
    )


@router.get("/api/discovery-status", response_class=HTMLResponse)
async def discovery_status(request: Request) -> HTMLResponse:
    """Return a compact nav pill showing the latest model discovery status."""
    log: list[dict] = getattr(request.app.state, "discovery_log", [])
    return HTMLResponse(content=_discovery_status_html(log))
