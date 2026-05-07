from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.mcp.catalog import MCPToolCatalog


def create_forum_mcp_server(tool_catalog: MCPToolCatalog) -> Any:
    """Create a FastMCP server exposing the forum tool catalog."""
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "MCP SDK is not installed. Add the 'mcp' package to run this server."
        ) from exc

    mcp = FastMCP(
        "taptupo-forum-tools",
        instructions=(
            "Tool gateway for Taptupo Forum agents. Use meta_list_tools first to "
            "inspect the available tool list and required arguments."
        ),
    )

    @mcp.tool()
    async def meta_list_tools() -> dict[str, Any]:
        return await tool_catalog.invoke("meta.list_tools", {})

    @mcp.tool()
    async def forum_news_refresh() -> dict[str, Any]:
        return await tool_catalog.invoke("forum.news.refresh", {})

    @mcp.tool()
    async def forum_news_latest(limit: int = 5) -> dict[str, Any]:
        return await tool_catalog.invoke("forum.news.latest", {"limit": limit})

    @mcp.tool()
    async def hn_top_stories(limit: int = 5) -> dict[str, Any]:
        return await tool_catalog.invoke("hn.top_stories", {"limit": limit})

    @mcp.tool()
    async def wikipedia_summary(query: str) -> dict[str, Any]:
        return await tool_catalog.invoke("wikipedia.summary", {"query": query})

    @mcp.tool()
    async def weather_current(
        location: str = "",
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"location": location}
        if latitude is not None:
            args["latitude"] = latitude
        if longitude is not None:
            args["longitude"] = longitude
        return await tool_catalog.invoke("weather.current", args)

    @mcp.tool()
    async def newsapi_top_headlines(
        country: str = "us",
        category: str = "technology",
        limit: int = 5,
    ) -> dict[str, Any]:
        return await tool_catalog.invoke(
            "newsapi.top_headlines",
            {
                "country": country,
                "category": category,
                "limit": limit,
            },
        )

    @mcp.tool()
    async def guardian_search(query: str, limit: int = 5) -> dict[str, Any]:
        return await tool_catalog.invoke(
            "guardian.search",
            {
                "query": query,
                "limit": limit,
            },
        )

    return mcp
