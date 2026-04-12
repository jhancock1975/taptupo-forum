"""Repository factory — select concrete implementation from environment.

Reads the following environment variables:

``DB_BACKEND``
    One of ``"local"`` (default) or ``"aws"``. Any other value raises
    :class:`ValueError`.
``DYNAMODB_ENDPOINT``
    Used only when ``DB_BACKEND=local``. Defaults to
    ``http://localhost:8000`` so dev setups with ``dynamodb-local`` work
    out of the box.
``AWS_REGION``
    Defaults to ``us-east-1``.
"""

from __future__ import annotations

import os

from app.db.dynamo import DynamoRepository
from app.db.interface import RepositoryInterface

_DEFAULT_LOCAL_ENDPOINT = "http://localhost:8000"
_DEFAULT_REGION = "us-east-1"


def get_repository() -> RepositoryInterface:
    """Return a repository configured from environment variables."""
    backend = os.environ.get("DB_BACKEND", "local")
    region = os.environ.get("AWS_REGION", _DEFAULT_REGION)

    if backend == "local":
        endpoint = os.environ.get("DYNAMODB_ENDPOINT", _DEFAULT_LOCAL_ENDPOINT)
        return DynamoRepository(endpoint_url=endpoint, region_name=region)
    if backend == "aws":
        return DynamoRepository(endpoint_url=None, region_name=region)

    raise ValueError(f"unknown DB_BACKEND: {backend!r}")
