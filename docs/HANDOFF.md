# Current Handoff

**Date:** 2026-09-02
**Status:** MVP implementation in progress; deterministic rule, explicit simulated verification adapter, durable outbox, operator console, local provisioning, synthetic backtest, historical CSV projection, ROI, audit, and health slices implemented

## Read first

`PRD.md`, `docs/CURRENT-STATE.md`, `docs/MVP.md`, `docs/ARCHITECTURE.md`, `docs/API-CONTRACTS.md`, `docs/INDIA-READINESS.md`, and `docs/ROADMAP.md`.

## Evidence

The original backend suite passed 168 tests at inspection. The current suite passes 193 tests and covers foundation invariants plus deterministic intelligence, the default simulated address-repair flow, verification rejection, backtest reproducibility, ROI reporting, and audit/health surfaces. The landing-page source referenced by old `.agents` handoffs is absent. The PRD appendix preserves its product requirements and marks the frontend claims as historical, not runtime evidence.

## Safe next task

Implement the next complete boundary without exposing unsupported capabilities: add an operator-controlled outbox worker, then add one real store/carrier integration. Keep live messaging, carrier, payments, and Shopify work in `docs/POST-MVP.md` until provider contracts are verified. Provider contracts and HMAC signing primitives now exist in `precourier/adapters.py` and `precourier/webhooks.py`, but no vendor is enabled.

## Never assume

- `NullDecider` is not a production risk model.
- `NoChannelActor` is not a messaging integration.
- A schema table is not an implemented feature.
- Static PIN data is not carrier serviceability.
- A payment enum is not verified payment.
- `release=true` on an error means every unidentified parcel may ship.
- Synthetic or landing-page metrics are merchant results.

## Required handoff updates

Record changed files, tests, exact command output, PRD sections affected, unresolved contracts, security/PII impact, India readiness, and the next phase. Keep `PRD.md` additive-only. This task affects PRD sections 1, 2, 9, 14, and 16: fulfilment fail-open, legal state changes, immutable commercial facts, tenant/audit integrity, and the test matrix.
