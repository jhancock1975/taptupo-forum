"""Argon2id password hashing via passlib.

Argon2 is the current OWASP recommendation for password storage. We use
passlib's CryptContext so future algorithm changes (e.g., rotating cost
parameters) remain transparent to callers: ``verify_password`` will also
return ``False`` for malformed hashes rather than raising.
"""

from __future__ import annotations

from passlib.context import CryptContext  # type: ignore[import-untyped,unused-ignore]
from passlib.exc import UnknownHashError  # type: ignore[import-untyped,unused-ignore]

_CONTEXT = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(plaintext: str) -> str:
    """Return an argon2id hash of ``plaintext`` suitable for storage."""
    return str(_CONTEXT.hash(plaintext))


def verify_password(plaintext: str, hashed: str) -> bool:
    """Return True iff ``plaintext`` matches ``hashed``.

    Returns ``False`` (never raises) for hashes that are not in a
    recognised format.
    """
    try:
        return bool(_CONTEXT.verify(plaintext, hashed))
    except (ValueError, UnknownHashError):
        return False
