# API Contracts

## Current

The current API is documented by `precourier/api/v1.py` and `precourier/api/schemas.py`. It uses bearer or `X-API-Key` authentication, rejects unknown request fields, accepts integer/decimal-string money through the existing boundary helpers, and exposes order lifecycle, outcomes, and hold sweep routes.

## Target public surface

```text
POST /v1/orders
GET  /v1/orders/{id}
GET  /v1/orders
POST /v1/orders/{id}/resolve
POST /v1/orders/{id}/release
POST /v1/orders/{id}/cancel
POST /v1/outcomes
POST /v1/holds/sweep
GET  /health
POST /v1/backtest
GET  /v1/reports/roi
GET  /v1/audit
GET  /v1/orders/{id}/messages
POST /v1/orders/{id}/messages/inbound
```

`POST /v1/backtest` accepts either the synthetic query-parameter form or a JSON
body with `csv_text` and optional `effectiveness_ppm`. A CSV response is always
labelled `provenance=merchant_historical_csv`; its observed ledger is historical
fact, while modeled uplift is explicitly assumption-labelled and is not causal
evidence.

## Idempotency

Every mutation must have an idempotency key or an immutable provider event ID. A replay with the same key and same canonical request returns the original result. The same key with different material input is a conflict. Outcome identity must not be based on kind alone.

## Release semantics

`release=true` is an order-specific instruction only when the request is authenticated and identifies a valid order. An internal decision timeout for that known order is fail-open and must set degraded evidence. Authentication failures, malformed requests, and unknown orders must not be interpreted as permission to ship.

## Versioning

Canonical endpoint and event names must be chosen before a live adapter ships. Marketing labels and simulator labels may not silently define API contracts. Provider payloads are retained separately from normalized business events.

## Provider boundaries

`precourier/adapters.py` defines the store, carrier, messaging, and payment
contracts. These are interfaces, not vendor integrations. A live adapter must
return provider identity, accept an idempotency key, distinguish rejection from
unknown provider state, and pass contract tests before it is enabled for a
merchant.
# Stage 1 decision fields

The decision engine accepts optional `external_signals` on an order. Each item is
`{"provider": "delhivery|shiprocket|gokwik|other", "score": 0..1, "label": "..."}`.
The baseline blends each valid signal conservatively into its labelled cause and
stores integer PPM probabilities for refusal, unreachable buyer, bad address, and
overall RTO. Provider calls use the fake adapters by default; Razorpay test mode
requires `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET`.
