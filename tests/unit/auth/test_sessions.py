"""Unit tests for app.auth.sessions."""

from __future__ import annotations

import pytest

from app.auth.sessions import SessionError, decode_session, encode_session

SECRET = "test-secret-do-not-use-in-prod"  # pragma: allowlist secret


@pytest.mark.unit
def test_encode_decode_round_trip() -> None:
    token = encode_session({"user_id": "u-123", "username": "alice"}, SECRET)
    assert isinstance(token, str)
    assert token != ""
    decoded = decode_session(token, SECRET)
    assert decoded == {"user_id": "u-123", "username": "alice"}


@pytest.mark.unit
def test_decode_rejects_tampered_payload() -> None:
    token = encode_session({"user_id": "u-123"}, SECRET)
    tampered = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
    with pytest.raises(SessionError):
        decode_session(tampered, SECRET)


@pytest.mark.unit
def test_decode_rejects_wrong_secret() -> None:
    token = encode_session({"user_id": "u-123"}, SECRET)
    with pytest.raises(SessionError):
        decode_session(token, "different-secret")  # pragma: allowlist secret


@pytest.mark.unit
def test_decode_rejects_garbage() -> None:
    with pytest.raises(SessionError):
        decode_session("not a real token", SECRET)
