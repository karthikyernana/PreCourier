"""Canonical JSON: one byte sequence per value, on every machine.

Three things in this system hash or sign a structure, and all three break if the
same data can serialise two ways:

*   the audit chain, `sha256(prev_hash + canonical_json(payload))`
*   outbound webhook signatures, `hmac_sha256(secret, canonical_json(body))`
*   the backtest's result digest, which PRD section 13 requires to be identical
    across runs and platforms

`json.dumps` defaults are not sufficient for that. It emits `", "` separators,
preserves insertion order, and renders floats through `repr`, which historically
differed between platforms. So this module fixes all three: sorted keys, tight
separators, and floats rejected outright.

Rejecting floats is the unusual choice and the important one. A float in a hashed
payload means a signature that depends on binary rounding; a float in a money
field means an audit trail that does not add up. Since `precourier.money` already
represents every quantity as an integer, a float reaching this module is a bug
upstream, and failing loudly here is how it gets found.
"""

from __future__ import annotations

import json
from typing import Any


class NotCanonical(TypeError):
    """A value that has no single unambiguous byte representation."""


def _check(value: Any, path: str = "$") -> None:
    """Reject anything whose serialisation is not fully determined."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int):
        return
    if isinstance(value, str):
        return
    if isinstance(value, float):
        raise NotCanonical(
            "float at %s: %r. Money is integer paise, rates are integer bps and "
            "probabilities are integer ppm -- see precourier.money. A float here "
            "would make the hash depend on binary rounding." % (path, value))
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise NotCanonical(
                    "non-string key at %s: %r. JSON coerces it, so two different "
                    "keys could collide after coercion." % (path, key))
            _check(item, "%s.%s" % (path, key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check(item, "%s[%d]" % (path, index))
        return
    if isinstance(value, (set, frozenset)):
        raise NotCanonical(
            "set at %s: iteration order is not defined. Sort it into a list at "
            "the call site, where the intended order is known." % (path,))
    raise NotCanonical("cannot canonicalise %s at %s"
                       % (type(value).__name__, path))


def dumps(payload: Any) -> str:
    """Canonical JSON text: sorted keys, no whitespace, no floats, no NaN."""
    _check(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def dumpb(payload: Any) -> bytes:
    """Canonical JSON bytes -- what actually gets hashed."""
    return dumps(payload).encode("utf-8")


def loads(text: str) -> Any:
    """Plain `json.loads`. Present so callers need one import, not two."""
    return json.loads(text)


def sha256_hex(*chunks: bytes) -> str:
    """Hex SHA-256 over the concatenation. The audit chain's single primitive."""
    import hashlib
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()
