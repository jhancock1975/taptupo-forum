"""DynamoDB-backed implementation of :class:`RepositoryInterface`."""

from __future__ import annotations

import aioboto3  # type: ignore[import-untyped]

from app.db.interface import RepositoryInterface


class DynamoRepository(RepositoryInterface):  # type: ignore[abstract]
    """DynamoDB repository (methods added in subsequent commits)."""

    endpoint_url: str | None
    region_name: str
    _session: aioboto3.Session

    def __init__(self, endpoint_url: str | None, region_name: str = "us-east-1") -> None:
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self._session = aioboto3.Session()
