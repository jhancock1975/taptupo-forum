"""Data-access layer for taptupo-forum."""

from app.db.dynamo import DynamoRepository
from app.db.factory import get_repository
from app.db.interface import (
    RepositoryError,
    RepositoryInterface,
    UserExistsError,
)

__all__ = [
    "DynamoRepository",
    "RepositoryError",
    "RepositoryInterface",
    "UserExistsError",
    "get_repository",
]
