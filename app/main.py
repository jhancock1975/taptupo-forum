from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agents.discussion import DiscussionEngine
from app.agents.news_agent import NewsAgent
from app.agents.registry import PERSONA_PRESETS, register_agents
from app.auth.routes import router as auth_router
from app.config import settings
from app.db.dynamo_local import DynamoLocalRepository
from app.logging_config import setup_logging
from app.mcp.catalog import MCPToolCatalog
from app.middleware import CorrelationIdMiddleware
from app.news.aggregator import NewsAggregator
from app.rendering import render_post_content
from app.routes.threads import router as threads_router
from app.routes.websocket import router as ws_router
from app.routes.websocket import ws_manager

logger = structlog.get_logger()

BASE_DIR = Path(__file__).resolve().parent


async def _news_loop(app: FastAPI) -> None:
    """Background task that periodically fetches news and promotes items."""
    await asyncio.sleep(10)  # initial delay for startup
    while True:
        try:
            aggregator: NewsAggregator = app.state.news_aggregator
            new_count = await aggregator.fetch_all()
            if new_count > 0:
                news_agent: NewsAgent = app.state.news_agent
                promoted_threads = await news_agent.promote_news(max_items=5)
                discussion_engine: DiscussionEngine = app.state.discussion_engine
                repo = app.state.repo
                for i, thread in enumerate(promoted_threads):
                    if i > 0:
                        await asyncio.sleep(30)
                    posts = await repo.get_posts_by_thread(thread.thread_id)
                    if not posts:
                        logger.warning(
                            "promoted_thread_missing_opening_post",
                            thread_id=thread.thread_id,
                        )
                        continue
                    await discussion_engine.on_new_post(thread.thread_id, posts[0])
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("news_loop_error")
        await asyncio.sleep(settings.news_fetch_interval_minutes * 60)


async def _model_discovery_loop(app: FastAPI) -> None:
    """Background task that periodically fetches free models from OpenRouter
    and assigns one per agent persona, generating a per-model skill prompt."""
    from app.agents.model_discovery import ModelDiscoveryService

    await asyncio.sleep(3)  # brief delay so DB is fully ready
    while True:
        try:
            if settings.openrouter_api_key:
                service = ModelDiscoveryService(settings.openrouter_api_key)
                agents = await app.state.repo.list_agents()
                updated = await service.refresh_agent_models(
                    agents,
                    app.state.repo,
                    PERSONA_PRESETS,
                    log=app.state.discovery_log,
                    hf_api_key=settings.huggingface_api_key,
                )
                if updated > 0:
                    fresh_agents = await app.state.repo.list_agents()
                    app.state.discussion_engine.reload_agents(fresh_agents)
                    logger.info("model_discovery_complete", updated=updated)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("model_discovery_loop_error")
        await asyncio.sleep(settings.model_refresh_interval_hours * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    logger.info("starting_app", db_backend=settings.db_backend)
    if not settings.openrouter_api_key:
        logger.warning(
            "openrouter_key_missing",
            message=(
                "OPENROUTER_API_KEY is not configured. "
                "Agents will not generate LLM replies."
            ),
        )

    # Database
    repo = DynamoLocalRepository()
    await repo.init_tables()
    app.state.repo = repo

    # Discovery log (persists in-process, shared with background job)
    app.state.discovery_log: list[dict] = []

    # S3 media bucket
    from app.storage import s3 as s3_store

    await s3_store.ensure_bucket_exists()

    # Templates
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.filters["render_post_content"] = render_post_content
    app.state.templates = templates

    # WebSocket manager
    app.state.ws_manager = ws_manager

    # Register agents
    agents = await register_agents(repo)

    # News
    app.state.news_aggregator = NewsAggregator(repo)
    news_user = next((a for a in agents if a.username == "Nova"), agents[0])
    app.state.news_agent = NewsAgent(news_user, repo)

    # In-process MCP tool catalog
    app.state.tool_catalog = MCPToolCatalog(
        repo=repo,
        news_aggregator=app.state.news_aggregator,
    )

    # Discussion engine
    app.state.discussion_engine = DiscussionEngine(
        repo=repo,
        agents=agents,
        ws_manager=ws_manager,
        templates=templates,
        tool_catalog=app.state.tool_catalog,
    )

    # Start background tasks
    news_task = asyncio.create_task(_news_loop(app))
    model_discovery_task = asyncio.create_task(_model_discovery_loop(app))

    logger.info("app_ready")
    yield

    news_task.cancel()
    model_discovery_task.cancel()
    try:
        await news_task
    except asyncio.CancelledError:
        pass
    try:
        await model_discovery_task
    except asyncio.CancelledError:
        pass
    logger.info("app_shutdown")


app = FastAPI(title="Taptupo Forum", lifespan=lifespan)

# Middleware
app.add_middleware(CorrelationIdMiddleware)

# Static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Routers
app.include_router(auth_router)
app.include_router(threads_router)
app.include_router(ws_router)
