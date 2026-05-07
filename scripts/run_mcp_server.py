from __future__ import annotations

import asyncio
from typing import Any

from app.db.dynamo_local import DynamoLocalRepository
from app.mcp.catalog import MCPToolCatalog
from app.mcp.server import create_forum_mcp_server
from app.news.aggregator import NewsAggregator


async def _build_server() -> Any:
    repo = DynamoLocalRepository()
    await repo.init_tables()
    aggregator = NewsAggregator(repo)
    catalog = MCPToolCatalog(repo=repo, news_aggregator=aggregator)
    return create_forum_mcp_server(catalog)


def main() -> None:
    server: Any = asyncio.run(_build_server())
    server.run()


if __name__ == "__main__":
    main()
