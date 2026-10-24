# PreCourier

PreCourier is India-first, pre-dispatch COD decision infrastructure. It evaluates an order before label creation, explains the likely delivery constraint, and applies merchant-specific rupee economics before choosing whether to intervene.

## Current MVP

The current repository contains a tested vertical slice:

```text
order intake -> observable rule risk -> cause evidence -> ledger gate
  -> simulated address-repair message -> buyer reply -> PIN verification
  -> append-only address revision -> release -> delivered/RTO outcome -> report
```

The default application uses a deterministic rule model and no outbound provider. It does not call WhatsApp, SMS, Telegram, carriers, stores, or payment providers. Synthetic backtest results are labelled synthetic and are not customer evidence. The simulated channel exists only in automated verification tests and is not selected by a normal application launch.

## Run

```bash
python -m venv .venv
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/pytest -q
./.venv/bin/python -m precourier.backtest --seed 42
```

The backtest prints JSON for `ship_all`, best-tuned `block_best`, and `agent`, plus an optional sensitivity grid and ten-seed summary:

```bash
./.venv/bin/python -m precourier.backtest --seed 42 --sensitivity --ten-seeds
```

To run the API locally with the installed environment:

```bash
export PRECOURIER_DB=var/precourier.db
export PRECOURIER_PEPPER="local-development-secret"
./.venv/bin/python tools/provision_local.py
./.venv/bin/uvicorn precourier.api:create_app --factory
```

Open `http://127.0.0.1:8000/docs` to inspect the production API contract, or
`http://127.0.0.1:8000/console` for the read-only operator view. The console asks
for a merchant API key in the browser and then loads only that merchant's queue,
reasoning, gate arithmetic, messages, outcomes, and audit status. A merchant must
be provisioned and issued a key by the local setup command above before making
authenticated requests. The default app has no message provider, so an
address-repair decision will release rather than pretend a buyer was contacted.

For a clean local run, change `PRECOURIER_DB` to a new filename, such as
`var/precourier-check-01.db`. Keep the same `PRECOURIER_PEPPER` for provisioning
and server startup. The printed test key is used only for this local inspection.

The API defaults to `var/precourier.db` unless `PRECOURIER_DB` is set. The pepper
must be stable for the lifetime of the database and must come from a secret store
in any shared environment.

```bash
curl http://127.0.0.1:8000/health
```

## API inspection flow

1. In Swagger, authorize with the `PRECOURIER_API_KEY` printed by `tools/provision_local.py`.
2. Call `POST /v1/orders` with a merchant order. The default app has no outbound channel, so this verifies scoring and safe release behavior only.
3. Copy the returned `order_id` and call `GET /v1/orders/{order_id}/messages`.
4. Confirm the response says `release: true`, `state: released_for_dispatch`, and `blocked_reason: no_channel` when no channel is configured.
5. Call `POST /v1/outcomes` with `kind: delivered` or `rto`.
6. Call `GET /v1/orders/{order_id}` to inspect features, risk, diagnosis, messages, address revisions, transitions, and ledger-related outcome data.
7. Call `GET /v1/reports/roi` and `GET /v1/audit` to inspect the merchant report and verified audit chain.

Order body:

```json
{
  "phone": "+919876543210",
  "value": "1200.00",
  "consent": {"simulated": true},
  "address": {
    "raw": "near temple",
    "city": "Bangalore",
    "pincode": "110001"
  }
}
```

Outcome body:

```json
{
  "order_id": "<returned-order-id>",
  "kind": "delivered"
}
```

For the deterministic synthetic evaluation, run:

```bash
./.venv/bin/python -m precourier.backtest --seed 42 --sensitivity --ten-seeds
```

The result contains three arms, nine sensitivity cells, ten fixed seeds, and a
synthetic-data disclosure. It is not a merchant performance claim.

```bash
curl http://127.0.0.1:8000/health
```

## Documentation

- `PRD.md` is the canonical product requirements document and is preserved in full.
- `docs/CURRENT-STATE.md` records source-backed implementation status.
- `docs/MVP.md` defines the current release gate.
- `docs/POST-MVP.md` records all remaining features in implementation order.
- `docs/INDIA-READINESS.md` records Indian integration, privacy, messaging, payment, and carrier constraints.
- `docs/HANDOFF.md` gives the next safe task to a future AI agent.

## Commercial boundary

The first commercial motion is a managed Indian advisory pilot: one store, one fulfilment path, one approved messaging channel, and operator review. Live providers and global-vendor capability are post-MVP until their complete request-to-storage-to-audit lifecycle is implemented and tested.
