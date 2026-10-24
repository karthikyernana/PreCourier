"""The ASGI application. One factory, three guarantees wired at the edges.

`create_app` is the only place that constructs a `Runtime`, opens a `Database`, and
reads the environment. Everything below it is handed what it needs, which is what
makes the same code run under a live clock in production and a simulated one in a
test without a flag.

The three guarantees, and where each is installed:

1.  **Fail-open.** `_unhandled` catches anything a route did not, and answers with
    `release: true`. `states.OrderService` already fails open around the decision
    pipeline; this catches the layer above it -- a bug in serialisation, a broken
    dependency, a `MemoryError`. A 500 that does not say what to do with the parcel
    is a stalled dispatch queue.

2.  **The timer.** `_sweeper` runs `sweep()` for every merchant on an interval, so
    `max_hold_minutes` is honoured by a deployment nobody is watching. Off when the
    clock is simulated: a `SimClock` does not advance on its own, so a background
    loop against one is a busy-wait that can never fire.

3.  **Latency.** `X-Response-Time-Ms` on every response, and a middleware that logs
    anything over the p95 target. PRD section 1 puts a number on this; a system that
    claims a latency budget without measuring it does not have one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import states
from ..audit import AuditLog
from ..runtime import Runtime, SystemClock
from ..store.db import SCHEMA_VERSION, Database
from ..tenancy import list_merchants, load_tenant
from . import schemas, v1
from .deps import HTTP_422, AppState
from ..console import CONSOLE_HTML

log = logging.getLogger("precourier.api")

#: The p95 target from PRD section 1, in milliseconds. Exceeding it is logged at
#: WARNING with the path, because the alternative -- discovering it from a
#: merchant's complaint -- is not a monitoring strategy.
LATENCY_BUDGET_MS = 5_000

#: How often the background sweeper looks for expired holds. Sixty seconds against
#: a 240-minute cap is 0.4% overshoot in the worst case, which is well inside the
#: promise and cheap enough to run on one thread.
SWEEP_INTERVAL_SECONDS = 60


def create_app(*, db: Optional[Database] = None,
               runtime: Optional[Runtime] = None,
               decider: Optional[states.Decider] = None,
               actor: Optional[states.Actor] = None,
               db_path: Optional[str] = None,
               synthetic: bool = False,
               sweeper: Optional[bool] = None) -> FastAPI:
    """Build the app.

    Every dependency is injectable and every default is the production one, so a
    test constructs the real application with a simulated clock rather than a
    stand-in for it. There is no second code path to keep in sync.

    `sweeper` defaults to on for a real clock and off for a simulated one. A test
    that wants the timer drives `POST /v1/holds/sweep` or calls `sweep()`, which is
    the same method the background task calls.
    """
    runtime = runtime or Runtime.live(pepper=os.environ.get("PRECOURIER_PEPPER", ""))
    db = db or Database.connect(
        db_path or os.environ.get("PRECOURIER_DB", "var/precourier.db"))

    if sweeper is None:
        sweeper = isinstance(runtime.clock, SystemClock)

    app_state = AppState(
        db=db, runtime=runtime, audit=AuditLog(db, runtime),
        # Stage 1 is the default live order path. The older one-cause decider
        # remains injectable for backwards-compatible deployments and tests.
        decider=decider if decider is not None else states.ProfitDecisionDecider(),
        actor=actor if actor is not None else states.NoChannelActor(),
        synthetic=synthetic)

    app = FastAPI(
        title="PreCourier",
        version="0.1.0",
        summary="Pre-dispatch RTO prevention for Indian COD commerce.",
        description=_DESCRIPTION,
        docs_url="/docs",
        redoc_url=None,
        lifespan=_lifespan_with_sweeper if sweeper else None,
    )
    app.state.precourier = app_state
    app.include_router(v1.router)

    _install_handlers(app)
    _install_latency_header(app)
    _install_health(app)
    app.add_api_route("/console", lambda: HTMLResponse(CONSOLE_HTML),
                     methods=["GET"], include_in_schema=False)
    return app


# ==========================================================================
# error handling -- the fail-open of last resort
# ==========================================================================

def _error(status_code: int, error: str, detail: str = "",
           order_id: Optional[str] = None) -> JSONResponse:
    body = schemas.ErrorOut(error=error, detail=detail,
                            release=True, order_id=order_id)
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _install_handlers(app: FastAPI) -> None:

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Reshaped into `ErrorOut` so that a 404 and a 500 carry the same fields.
        # An integration should not need two error parsers.
        response = _error(exc.status_code, _slug(exc.status_code),
                          str(exc.detail))
        for header, value in (exc.headers or {}).items():
            response.headers[header] = value
        return response

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request,
                       exc: RequestValidationError) -> JSONResponse:
        # Field-level detail, flattened. Pydantic's own structure is nested and
        # includes the input value, which for these payloads means echoing a phone
        # number into an error body.
        problems = "; ".join(
            "%s: %s" % (".".join(str(p) for p in err.get("loc", ())[1:]) or "body",
                        err.get("msg", "invalid"))
            for err in exc.errors())
        return _error(HTTP_422, "invalid_request", problems)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The last line of the fulfilment promise. Logged with a traceback because
        # this should never happen, and answered with `release: true` because when
        # it does, a merchant's parcels must keep moving.
        log.exception("unhandled error on %s %s", request.method,
                      request.url.path)
        return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error",
                      "this request failed and no decision was made; print the "
                      "label")


def _slug(status_code: int) -> str:
    return {400: "bad_request", 401: "unauthorized", 403: "forbidden",
            404: "not_found", 409: "conflict",
            422: "invalid_request", 429: "rate_limited",
            }.get(status_code, "error")


# ==========================================================================
# latency
# ==========================================================================

def _install_latency_header(app: FastAPI) -> None:

    @app.middleware("http")
    async def _timed(request: Request, call_next: Any) -> Any:
        # `time.monotonic` rather than the injected clock: this measures a duration
        # for an operator, not an event for the ledger. A `SimClock` would report
        # every request as instantaneous, which is exactly the wrong answer here.
        started = time.monotonic()
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
        if elapsed_ms > LATENCY_BUDGET_MS:
            log.warning("%s %s took %dms, over the %dms budget", request.method,
                        request.url.path, elapsed_ms, LATENCY_BUDGET_MS)
        return response


# ==========================================================================
# health
# ==========================================================================

def _install_health(app: FastAPI) -> None:

    @app.get("/health", response_model=schemas.HealthOut, tags=["ops"])
    @app.get("/v1/health", response_model=schemas.HealthOut, tags=["ops"])
    def health(request: Request) -> Dict[str, Any]:
        """Unauthenticated, and deliberately says almost nothing.

        Enough for a load balancer and for a merchant to confirm which build they
        are talking to. No merchant names, no counts, no schema detail -- an
        unauthenticated endpoint that reports how many orders exist is a business
        intelligence feed for anyone who finds the URL.

        `synthetic_data` is here because PRD section 13 requires synthetic data to
        be disclosed wherever it is shown, and an API response is a surface.
        """
        app_state: AppState = request.app.state.precourier
        problems = app_state.db.integrity_check()
        return schemas.HealthOut(
            status="ok" if not problems else "degraded", schema_version=SCHEMA_VERSION,
            mode="live" if isinstance(app_state.runtime.clock, SystemClock)
                 else "deterministic",
            synthetic_data=app_state.synthetic,
            build=app_state.build).model_dump()


# ==========================================================================
# the timer
# ==========================================================================

@contextlib.asynccontextmanager
async def _lifespan_with_sweeper(app: FastAPI) -> AsyncIterator[None]:
    """Run the auto-release timer for as long as the app is up.

    A lifespan rather than `@app.on_event`, so the task is cancelled on the way down
    even if startup raised. An orphaned sweeper holding the database lock would
    outlive the process it was serving.
    """
    task = asyncio.create_task(_sweep_forever(app))
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):     # noqa: BLE001
            pass


async def _sweep_forever(app: FastAPI) -> None:
    while True:
        try:
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
            # Off the event loop: `sweep` is synchronous and takes the database
            # lock, and blocking the loop would add latency to every concurrent
            # decision -- a timer that protects fulfilment latency by damaging it.
            released = await asyncio.get_running_loop().run_in_executor(
                None, sweep_all, app.state.precourier)
            if released:
                log.info("auto-released %d order(s) whose hold expired", released)
        except asyncio.CancelledError:
            raise
        except Exception:                       # noqa: BLE001
            # A failed sweep must not kill the timer. The next tick tries again,
            # and until then `max_hold_minutes` is enforced by whoever calls
            # `POST /v1/holds/sweep`.
            log.exception("hold sweep failed; retrying on the next tick")


def sweep_all(app_state: AppState) -> int:
    """Sweep expired holds for every active merchant. Returns the count released.

    Also the entry point for a cron-driven deployment: `python -m precourier.api
    --sweep` and this function are the same work.
    """
    total = 0
    with app_state.lock:
        merchants: List[Dict[str, Any]] = list_merchants(app_state.db)
        for row in merchants:
            if str(row.get("status", "active")) != "active":
                continue
            tenant = load_tenant(app_state.db, app_state.runtime,
                                 str(row["id"]))
            total += len(app_state.service(tenant).sweep())
    return total


_DESCRIPTION = """
A decision API that sits between *order placed* and *label printed*.

`POST /v1/orders` answers one question: **may this label be printed?** The answer
carries the probability, the threshold it was compared against, the arithmetic that
produced the threshold, and what was done about it.

### The integration contract

*   Branch on `release`. `true` means print. `false` means hold until `held_until`.
*   **On any error, or any timeout, print the label.** Every error body from this API
    carries `release: true` for that reason. A hold is never worth a stalled
    dispatch queue, and every order is auto-released after `max_hold_minutes`
    regardless of what this service does or fails to do.
*   Send `Idempotency-Key` on writes. A replayed key returns the original response
    rather than deciding twice.
*   Money is an integer number of paise, or a decimal string. Never a JSON number.

### Modes

*   `shadow` -- reason fully, record the counterfactual, touch nothing, release.
*   `advisory` -- act, then release without waiting for the result.
*   `gateway` -- act, and hold the label until the order resolves or times out.
""".strip()


__all__ = ["create_app", "sweep_all", "AppState", "LATENCY_BUDGET_MS",
           "SWEEP_INTERVAL_SECONDS"]
