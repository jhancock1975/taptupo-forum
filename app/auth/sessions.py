"""Signed session cookies via itsdangerous.

We serialise a small JSON-compatible payload (user_id, username) into a
URL-safe, HMAC-signed token. ``decode_session`` surfaces every failure
mode (tampering, wrong secret, malformed input) as a single
``SessionError`` so callers can treat them uniformly.
"""

from __future__ import annotations

from typing import Any

from itsdangerous import BadSignature, URLSafeSerializer

_SALT = "taptupo-session-v1"


class SessionError(Exception):
    """Raised when a session token cannot be verified or decoded."""


def encode_session(payload: dict[str, Any], secret: str) -> str:
    """Return a signed, URL-safe token carrying ``payload``."""
    serializer = URLSafeSerializer(secret, salt=_SALT)
    return str(serializer.dumps(payload))


def decode_session(token: str, secret: str) -> dict[str, Any]:
    """Verify and return the payload embedded in ``token``.

    Raises ``SessionError`` if the token is tampered, signed with a
    different secret, or otherwise malformed.
    """
    serializer = URLSafeSerializer(secret, salt=_SALT)
    try:
        loaded = serializer.loads(token)
    except BadSignature as exc:
        raise SessionError("invalid session token") from exc
    if not isinstance(loaded, dict):
        raise SessionError("session payload is not a dict")
    return loaded
