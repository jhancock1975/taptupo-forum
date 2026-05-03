from __future__ import annotations

from app.auth.utils import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_and_verify(self) -> None:
        hashed = hash_password("my-secret-pass")
        assert hashed != "my-secret-pass"
        assert verify_password("my-secret-pass", hashed)

    def test_wrong_password(self) -> None:
        hashed = hash_password("correct")
        assert not verify_password("wrong", hashed)


class TestJWT:
    def test_create_and_decode(self) -> None:
        token = create_access_token("user-123", "alice")
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["user_id"] == "user-123"
        assert payload["username"] == "alice"

    def test_invalid_token(self) -> None:
        result = decode_access_token("not.a.valid.token")
        assert result is None
