"""Wiring: application state, authentication, and the idempotency wrapper.

The single most important line in this file is the authentication dependency
returning a `Tenant` built from a key *hash*. Nothing downstream ever sees a
`merchant_id` that came off a request, which is what makes the isolation guarantee
in `tenancy.py` reach all the way to the socket. A route that wanted to accept a
`merchant_id` parameter could not use it: every repository takes a `Tenant`.

Authentication failures are deliberately uniform -- same status, same body, same
timing characteristics as far as this layer controls. A 401 that distinguishes
"unknown key" from "revoked key" from "suspended merchant" is an oracle for
enumerating the other two.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Any, Dict, Optional

from fastapi import Depends, Header, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from .. import states
from ..audit import AuditLog
from ..runtime import Runtime
from ..store.db import Database
from ..store.repo import Conflict, IdempotencyRepo
from ..tenancy import AuthError, Tenant, authenticate

#: Returned for every authentication failure, whatever the cause.
UNAUTHORIZED = "invalid or missing API key"

#: 422. Starlette renamed its constant from `HTTP_422_UNPROCESSABLE_ENTITY` to
#: `..._CONTENT` (RFC 9110's wording) and deprecated the old spelling, so the number
#: is written once here rather than importing a name that moves between versions.
#: The status code is the contract; what the RFC calls it is not.
HTTP_422 = 422

_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclasses.dataclass
class AppState:
    """Everything the app holds for its lifetime.

    `lock` serialises database work across the threadpool that serves sync
    endpoints. One SQLite connection is not thread-safe, and holding a lock is the
    honest version of that constraint -- the alternative, a connection pool, would
    imply a concurrency this deployment does not have.

    `decider` and `actor` are the two seams. M4 replaces the first, M7 the second,
    and neither touches a route.
    """

    db: Database
    runtime: Runtime
    audit: AuditLog
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    decider: Optional[states.Decider] = None
    actor: Optional[states.Actor] = None

    #: Whether the data in this deployment is synthetic. Rendered on every console
    #: page and returned by `/health`, because PRD section 13 requires synthetic
    #: data to be disclosed as synthetic wherever it is displayed -- a demo that
    #: looks like production data is a lie by omission.
    synthetic: bool = False
    build: str = "precourier-p0"

    def service(self, tenant: Tenant, *, arm: str = "live") -> "states.OrderService":
        return states.OrderService(tenant, decider=self.decider,
                                   actor=self.actor, arm=arm)


def state_of(request: Request) -> AppState:
    return request.app.state.precourier          # type: ignore[no-any-return]


# ==========================================================================
# authentication
# ==========================================================================

def bearer_token(
        credentials: Optional[HTTPAuthorizationCredentials] = Security(
            _bearer_scheme),
        x_api_key: Optional[str] = Security(_api_key_scheme),
        authorization: Optional[str] = Header(default=None)) -> str:
    """Pull the key from `Authorization: Bearer ...` or `X-API-Key`.

    Both accepted because order-feed integrations differ and a merchant should not
    have to argue with a header name to get started. The key itself is never
    logged, never echoed, and never stored -- `authenticate` hashes it.
    """
    if credentials is not None and credentials.scheme.lower() == "bearer":
        if credentials.credentials.strip():
            return credentials.credentials.strip()
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, UNAUTHORIZED,
                        headers={"WWW-Authenticate": "Bearer"})


def get_tenant(request: Request, token: str = Depends(bearer_token)) -> Tenant:
    app_state = state_of(request)
    try:
        with app_state.lock:
            return authenticate(app_state.db, app_state.runtime, token)
    except AuthError:
        # No detail. See the module docstring: the difference between an unknown
        # key and a revoked one is information an attacker wants and a legitimate
        # caller does not need.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, UNAUTHORIZED,
                            headers={"WWW-Authenticate": "Bearer"})


# ==========================================================================
# idempotency
# ==========================================================================

class Idempotent:
    """Replay protection for one endpoint, as a context manager.

    Usage is deliberately explicit rather than middleware:

        with Idempotent(tenant, "POST /v1/orders", key, payload) as slot:
            if slot.replayed:
                return slot.response
            ...
            slot.store(response)

    Middleware would have to guess which routes are idempotent and would have to
    re-serialise every response to hash it. Being explicit means the *endpoint*
    declares its own replay semantics, which is where that knowledge lives: an
    outcome webhook and an order feed need the same protection for different
    reasons.

    A missing `Idempotency-Key` is allowed and simply does nothing. Requiring one
    would break the first integration a merchant writes, and the cost of the
    default is bounded -- `OrderRepo.create` already refuses a duplicate
    `external_id`, and `OutcomeRepo.record` is idempotent on the outcome itself.

    A key reused with a *changed* body is a 409, not a replay. The two wrong answers
    here are both silent: returning the first response drops the second order, and
    ignoring the key creates a duplicate the merchant believes was suppressed. A
    conflict is the only answer that surfaces the caller's bug without losing their
    order -- they can retry it under a correct key.
    """

    def __init__(self, tenant: Tenant, endpoint: str, key: Optional[str],
                 payload: Any) -> None:
        self._repo = IdempotencyRepo(tenant)
        self._endpoint = endpoint
        self._key = (key or "").strip()
        self._hash = IdempotencyRepo.hash_request(payload)
        self.replayed = False
        self.status_code = status.HTTP_200_OK
        self.response: Dict[str, Any] = {}

    def __enter__(self) -> "Idempotent":
        if not self._key:
            return self
        try:
            found = self._repo.lookup(self._key, self._endpoint, self._hash)
        except Conflict as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
        if found is not None:
            self.status_code, self.response = found
            self.replayed = True
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def store(self, response: Dict[str, Any], *,
              status_code: int = status.HTTP_200_OK) -> None:
        if not self._key:
            return
        self._repo.remember(self._key, self._endpoint, self._hash,
                            status_code=status_code, response=response)


def idempotency_key(idempotency_key: Optional[str] = Header(default=None)
                    ) -> Optional[str]:
    return idempotency_key
