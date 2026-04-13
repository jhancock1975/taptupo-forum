"""Unit tests for app.auth.passwords."""

from __future__ import annotations

import pytest

from app.auth.passwords import hash_password, verify_password


@pytest.mark.unit
def test_hash_password_returns_non_empty_string_distinct_from_plaintext() -> None:
    plaintext = "correct horse battery staple"  # pragma: allowlist secret
    hashed = hash_password(plaintext)
    assert isinstance(hashed, str)
    assert hashed != plaintext
    assert hashed != ""


@pytest.mark.unit
def test_hash_password_uses_argon2() -> None:
    hashed = hash_password("whatever")  # pragma: allowlist secret
    assert hashed.startswith("$argon2")


@pytest.mark.unit
def test_hash_password_produces_unique_hashes_for_same_plaintext() -> None:
    plaintext = "correct horse battery staple"  # pragma: allowlist secret
    assert hash_password(plaintext) != hash_password(plaintext)


@pytest.mark.unit
def test_verify_password_accepts_correct_password() -> None:
    plaintext = "correct horse battery staple"  # pragma: allowlist secret
    hashed = hash_password(plaintext)
    assert verify_password(plaintext, hashed) is True


@pytest.mark.unit
def test_verify_password_rejects_wrong_password() -> None:
    hashed = hash_password("right")  # pragma: allowlist secret
    assert verify_password("wrong", hashed) is False  # pragma: allowlist secret


@pytest.mark.unit
def test_verify_password_rejects_invalid_hash_format() -> None:
    assert verify_password("anything", "not-a-real-hash") is False
