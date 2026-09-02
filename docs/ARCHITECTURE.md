# Architecture

## Current flow

```text
HTTP request
  -> API authentication
  -> Tenant-scoped repositories
  -> OrderService
  -> Decider seam (currently NullDecider)
  -> state transitions / decision
  -> optional Actor seam
  -> release or hold
  -> outcome endpoint
  -> ledger + audit
```

## Target flow

```text
Store/OMS
  -> signed canonical order + idempotency
  -> synchronous decision service
       -> features -> risk -> diagnosis -> gate
  -> durable decision and action intent
  -> commit
  -> durable outbox executor
       -> messaging / carrier / payment / store adapter
  -> provider result and reconciliation
  -> fulfilment release/hold/cancel
  -> carrier outcome
  -> ledger + training label + report
```

## Non-negotiable boundaries

- The authenticated tenant, not a request field, owns every query.
- External side effects never run inside an open database transaction.
- Risk probability is calibrated/range-checked before money arithmetic.
- LLM output cannot set a rupee amount, state, release decision, or payment status.
- Static PIN data may validate consistency but cannot claim live COD serviceability.
- The backtest truth store is not imported by agent/runtime modules.
- Business state, fulfilment-gate state, provider state, payment state, and buyer-response state are separate concepts even if an MVP projects them into one response.
- A provider intent is committed before an adapter runs. `action_outbox` is the retry and reconciliation boundary; an action is never considered provider-complete merely because an HTTP call was attempted.

## Failure model

| Failure | Known order | Result |
|---|---|---|
| Rule/model/internal timeout | yes | release, mark degraded, audit, alert |
| Invalid authentication/request | no reliable dispatch identity | error; no inferred release instruction |
| Message provider rejected before acceptance | yes | record channel failure; do not claim buyer contact; release or bounded policy path |
| Provider accepted but receipt lost | yes | reconcile by provider idempotency key; never duplicate blindly |
| Payment callback unverified | yes | remain unpaid; never mark converted |
| Hold deadline reached | yes | auto-release and audit timeout |

## Data model direction

Preserve the existing append-only ledger and audit chain. Add durable feature, risk, diagnosis, action, message, payment, provider-event, and training-label records with explicit tenant ownership, idempotency identity, source, timestamps, and provenance.
