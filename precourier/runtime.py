"""Clock and identity -- the two ambient dependencies, made injectable.

`datetime.now()` and `uuid4()` are the usual way a system becomes irreproducible.
PRD section 13 requires that `--seed 42` reproduce every published number
byte-for-byte, which is impossible if any row carries a wall-clock timestamp or a
random id. So neither is ever called directly outside this module: everything
that needs the time or a new id takes a `Clock` and an `Ids`.

Two implementations of each, chosen at the edge:

    live API      SystemClock + RandomIds     real time, unguessable ids
    backtest      SimClock    + CountingIds   simulated time, deterministic ids

The seam is deliberately narrow. If a module imports `datetime` or `uuid`
directly it has broken reproducibility, and `tests/test_determinism.py` walks the
package AST to say so.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import hmac
import secrets
from typing import Dict, Optional


UTC = _datetime.timezone.utc

#: Timestamp format used in every stored row and every API response. ISO-8601,
#: UTC, second precision, 'Z' suffix -- lexicographically sortable, which means
#: `ORDER BY created_at` is correct without a date function.
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def format_ts(moment: _datetime.datetime) -> str:
    """Render an aware datetime in the storage format, normalising to UTC."""
    if moment.tzinfo is None:
        raise ValueError("refusing to format a naive datetime: %r" % (moment,))
    return moment.astimezone(UTC).strftime(TS_FORMAT)


def parse_ts(text: str) -> _datetime.datetime:
    """Inverse of `format_ts`. Accepts a trailing 'Z' or an explicit offset."""
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = _datetime.datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


# ==========================================================================
# clocks
# ==========================================================================

class Clock:
    """Source of the current instant.

    Subclasses implement `now`. `stamp` is what callers actually use, so a
    timestamp's string form is decided in exactly one place.
    """

    def now(self) -> _datetime.datetime:
        raise NotImplementedError

    def stamp(self) -> str:
        return format_ts(self.now())

    def plus_minutes(self, minutes: int) -> str:
        return format_ts(self.now() + _datetime.timedelta(minutes=minutes))

    def plus_hours(self, hours: int) -> str:
        return format_ts(self.now() + _datetime.timedelta(hours=hours))


class SystemClock(Clock):
    """Wall time, UTC. The only place in the package that reads the real clock."""

    def now(self) -> _datetime.datetime:
        return _datetime.datetime.now(tz=UTC)


class SimClock(Clock):
    """Simulated time that advances only when told to.

    The backtest owns one of these, so a run's timestamps depend on the seed and
    the event sequence rather than on when the run happened. Also what tests use
    to reach a hold expiry without sleeping four hours.
    """

    def __init__(self, start: str = "2026-01-01T00:00:00Z") -> None:
        self._at = parse_ts(start)

    def now(self) -> _datetime.datetime:
        return self._at

    def advance(self, *, minutes: int = 0, hours: int = 0,
                days: int = 0, seconds: int = 0) -> "SimClock":
        self._at += _datetime.timedelta(minutes=minutes, hours=hours,
                                        days=days, seconds=seconds)
        return self

    def set_to(self, moment: str) -> "SimClock":
        self._at = parse_ts(moment)
        return self


# ==========================================================================
# identity
# ==========================================================================

#: Short, typed prefixes. A stray id in a log or a URL says what it is, and a
#: test can assert an endpoint was handed an order id rather than a buyer id.
PREFIXES: Dict[str, str] = {
    "merchant": "mch",
    "api_key": "key",
    "user": "usr",
    "buyer": "byr",
    "order": "ord",
    "address": "adr",
    "transition": "trn",
    "features": "ftr",
    "risk": "rsk",
    "diagnosis": "dgn",
    "decision": "dcn",
    "action": "act",
    "message": "msg",
    "payment": "pay",
    "outcome": "out",
    "ledger": "led",
    "config": "cfg",
    "model": "mdl",
    "webhook": "whk",
    "delivery": "dlv",
}


class Ids:
    """Source of new identifiers."""

    def new(self, kind: str) -> str:
        raise NotImplementedError

    @staticmethod
    def _prefix(kind: str) -> str:
        try:
            return PREFIXES[kind]
        except KeyError:
            raise ValueError(
                "unknown id kind %r -- add it to runtime.PREFIXES rather than "
                "inventing a prefix at the call site" % (kind,))


class RandomIds(Ids):
    """Unguessable ids for the live API.

    128 bits of `secrets` entropy. Order ids appear in buyer-facing links, so an
    incrementing integer would let anyone enumerate a merchant's orders.
    """

    def new(self, kind: str) -> str:
        return "%s_%s" % (self._prefix(kind), secrets.token_hex(16))


class CountingIds(Ids):
    """Deterministic ids: `ord_000001`, `ord_000002`, ...

    Per-kind counters, so adding a diagnosis does not shift every subsequent
    order id and a diff between two backtest runs stays readable.
    """

    def __init__(self, width: int = 6) -> None:
        self._width = width
        self._counters: Dict[str, int] = {}

    def new(self, kind: str) -> str:
        prefix = self._prefix(kind)
        nxt = self._counters.get(kind, 0) + 1
        self._counters[kind] = nxt
        return "%s_%0*d" % (prefix, self._width, nxt)

    def peek(self, kind: str) -> int:
        return self._counters.get(kind, 0)


# ==========================================================================
# secrets handling
# ==========================================================================

def hash_secret(plaintext: str, *, pepper: str = "") -> str:
    """One-way hash for an API key or webhook secret at rest.

    Plain SHA-256, no salt, because lookup is by hash: a per-row salt would force
    a table scan and a bcrypt-style KDF would cost milliseconds on the p95 hot
    path. That is a defensible trade for a 128-bit random key -- there is no
    dictionary to attack. It would not be defensible for a password, and this
    function is never used for one.

    `pepper` comes from the environment when set, so a stolen database alone does
    not permit offline verification of guessed keys.
    """
    import hashlib
    digest = hashlib.sha256()
    if pepper:
        digest.update(pepper.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(plaintext.encode("utf-8"))
    return digest.hexdigest()


def secrets_equal(left: str, right: str) -> bool:
    """Constant-time comparison, so a timing signal cannot leak a key."""
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def mint_api_key(mode: str, *, ids: Optional[Ids] = None) -> str:
    """A fresh `pk_test_...` / `pk_live_...` key. Shown once, then only hashed.

    Uses `secrets` regardless of the injected `Ids`, because a deterministic
    credential is not a credential. Reproducibility applies to the numbers a
    backtest publishes, never to a key.
    """
    if mode not in ("test", "live"):
        raise ValueError("mode must be 'test' or 'live', got %r" % (mode,))
    return "pk_%s_%s" % (mode, secrets.token_hex(20))


def key_prefix(plaintext: str, keep: int = 6) -> str:
    """The displayable head of a key: `pk_test_a1b2c3`.

    Enough for a merchant to tell two keys apart in the console, far too little
    to reconstruct one.
    """
    head, _, tail = plaintext.rpartition("_")
    if not head:
        return plaintext[:keep]
    return "%s_%s" % (head, tail[:keep])


# ==========================================================================
# bundle
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class Runtime:
    """The ambient dependencies, passed as one argument instead of three.

    Constructed once at each entry point -- `api.app`, `backtest.__main__`, a
    test fixture -- and threaded down. Nothing below the entry point decides for
    itself what time it is.
    """

    clock: Clock
    ids: Ids
    pepper: str = ""

    @classmethod
    def live(cls, pepper: str = "") -> "Runtime":
        return cls(clock=SystemClock(), ids=RandomIds(), pepper=pepper)

    @classmethod
    def deterministic(cls, start: str = "2026-01-01T00:00:00Z") -> "Runtime":
        return cls(clock=SimClock(start), ids=CountingIds())

    def stamp(self) -> str:
        return self.clock.stamp()

    def new_id(self, kind: str) -> str:
        return self.ids.new(kind)

    def hash_secret(self, plaintext: str) -> str:
        return hash_secret(plaintext, pepper=self.pepper)
