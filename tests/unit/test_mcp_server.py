from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.mcp.server import create_forum_mcp_server


class FakeFastMCP:
    def __init__(self, name: str, instructions: str) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}

    def tool(self) -> Any:
        def decorator(func: Any) -> Any:
            self.tools[func.__name__] = func
            return func

        return decorator


@pytest.fixture
def fake_fastmcp_module(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP
    server_module.fastmcp = fastmcp_module
    mcp_module.server = server_module

    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)


@pytest.mark.anyio
async def test_create_forum_mcp_server_registers_tools_and_proxies_calls(
    fake_fastmcp_module: None,
) -> None:
    _ = fake_fastmcp_module
    invoke = AsyncMock(side_effect=lambda name, args: {"tool": name, "args": args})
    catalog = types.SimpleNamespace(invoke=invoke)

    server = create_forum_mcp_server(catalog)

    assert isinstance(server, FakeFastMCP)
    assert server.name == "taptupo-forum-tools"
    assert set(server.tools) == {
        "meta_list_tools",
        "forum_news_refresh",
        "forum_news_latest",
        "hn_top_stories",
        "wikipedia_summary",
        "weather_current",
        "newsapi_top_headlines",
        "guardian_search",
    }

    assert await server.tools["meta_list_tools"]() == {
        "tool": "meta.list_tools",
        "args": {},
    }
    assert await server.tools["forum_news_refresh"]() == {
        "tool": "forum.news.refresh",
        "args": {},
    }
    assert await server.tools["forum_news_latest"](9) == {
        "tool": "forum.news.latest",
        "args": {"limit": 9},
    }
    assert await server.tools["hn_top_stories"](4) == {
        "tool": "hn.top_stories",
        "args": {"limit": 4},
    }
    assert await server.tools["wikipedia_summary"]("Ada Lovelace") == {
        "tool": "wikipedia.summary",
        "args": {"query": "Ada Lovelace"},
    }
    assert await server.tools["weather_current"]("Berlin", 52.5, 13.4) == {
        "tool": "weather.current",
        "args": {
            "location": "Berlin",
            "latitude": 52.5,
            "longitude": 13.4,
        },
    }
    assert await server.tools["newsapi_top_headlines"]("us", "science", 3) == {
        "tool": "newsapi.top_headlines",
        "args": {
            "country": "us",
            "category": "science",
            "limit": 3,
        },
    }
    assert await server.tools["guardian_search"]("space", 2) == {
        "tool": "guardian.search",
        "args": {"query": "space", "limit": 2},
    }

    assert invoke.await_count == 8
