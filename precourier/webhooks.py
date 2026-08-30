"""Provider-neutral HMAC webhook primitives."""

from __future__ import annotations

import hashlib
import hmac


def sign_payload(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify_payload(secret: str, body: bytes, signature: str) -> bool:
    expected = sign_payload(secret, body)
    supplied = signature.strip().removeprefix("sha256=")
    return hmac.compare_digest(expected, supplied)


__all__ = ["sign_payload", "verify_payload"]
