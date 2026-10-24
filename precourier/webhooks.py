"""Provider-neutral HMAC webhook primitives."""

from __future__ import annotations

import hashlib
import hmac
import time


def sign_payload(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify_payload(secret: str, body: bytes, signature: str) -> bool:
    expected = sign_payload(secret, body)
    supplied = signature.strip().removeprefix("sha256=")
    return hmac.compare_digest(expected, supplied)


def verify_replay_safe(secret: str, body: bytes, signature: str, timestamp: int,
                       *, now: int | None = None, window_seconds: int = 300,
                       seen: set[str] | None = None) -> bool:
    """Verify HMAC plus timestamp freshness and one-time delivery identity."""
    current = int(time.time()) if now is None else int(now)
    if abs(current - int(timestamp)) > window_seconds:
        return False
    if not verify_payload(secret, body, signature):
        return False
    identity = hashlib.sha256(str(timestamp).encode() + body).hexdigest()
    if seen is not None:
        if identity in seen:
            return False
        seen.add(identity)
    return True


__all__ = ["sign_payload", "verify_payload", "verify_replay_safe"]
