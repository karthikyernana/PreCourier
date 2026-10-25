# Current State

**Inspection date:** 2026-09-02

## Repository reality

The repository is an uncommitted Python backend foundation. The deleted `/landing-page` or `/landing` frontend is not present, and the old `.agents` reports claiming a 91-test React application cannot be independently verified. Those reports are preserved as historical product input in the PRD appendix, not as implementation evidence.

## Implemented

- Merchant configuration and integer-paise economics: `precourier/config.py`, `precourier/ledger.py`.
- Domain vocabulary and legal order transition graph: `precourier/domain.py`.
- API-key authentication and tenant-scoped access: `precourier/tenancy.py`, `precourier/api/deps.py`.
- SQLite schema, repositories, idempotency, order lifecycle, outcomes, ledger persistence: `precourier/store/`.
- Holds, bounded release, adoption modes, and fail-open service path: `precourier/states.py`.
- Consent, quiet-hours, channel, budget, and cancellation guards: `precourier/guards.py`.
- Hash-chained audit events: `precourier/audit.py`.
- Indian phone and PIN/geography helpers: `precourier/pii.py`, `precourier/geo.py`.
- Isolated synthetic world and truth database: `precourier/worldgen/`.
- Current test suite: 209 passing tests after the current implementation slice.

## Partial or absent

The default decision path now uses the deterministic `RuleDecider`; it persists first-pass feature/risk/diagnosis artifacts, applies the integer economic gate, and has no outbound provider by default. The simulated address-repair channel is explicit test/verification infrastructure only. A durable `action_outbox` records provider intent, attempts, retries, and provider references before any adapter call. Provider contracts and HMAC webhook primitives are defined, but no vendor is enabled. A read-only operator console at `/console` visualizes authenticated recent orders in every state plus reasoning, gate, message, outcome, and audit data. Merchant configuration and API-key lifecycle are available through tenant-scoped control-plane endpoints. `tools/provision_local.py` is developer setup only and does not enable the simulator. A zero-dependency synthetic backtest, sensitivity grid, ten-seed aggregation, live ROI/audit reports, historical CSV projection, and deterministic address verification are present. Carrier/store/payment adapters, production messaging, storefront checkout, webhook delivery, bandit learning, model registry, LLM composition, and payment conversion remain absent or partial.

## Current API

- `GET /health`
- `POST /v1/orders`
- `GET /v1/orders`
- `GET /v1/orders/{order_id}`
- `POST /v1/orders/{order_id}/resolve`
- `POST /v1/orders/{order_id}/release`
- `POST /v1/orders/{order_id}/cancel`
- `POST /v1/outcomes`
- `POST /v1/holds/sweep`
- `POST /v1/orders/{order_id}/messages/inbound`
- `GET /v1/orders/{order_id}/messages`
- `POST /v1/backtest`
- `GET /v1/reports/roi`
- `GET /v1/audit`
- `GET /health`

The report endpoints now use synthetic provenance for backtests and persisted merchant provenance for live ROI. Historical CSV onboarding, a real carrier/channel, and a full merchant UI are still implementation targets.

## Known foundation risks

- External actor calls are coupled to the transaction boundary.
- Incentive and discount accounting are incomplete.
- Terminal/negative hold validation needs hardening.
- Some order commercial fields are omitted from immutability checks.
- Audit export does not yet provide a complete PII policy.
- Outcome retries, correction identity, and cross-tenant foreign-key ownership need explicit enforcement.

These are tracked in `docs/ROADMAP.md` and must not be hidden by new feature work.
# Stage 1 status

Pure Stage 1 decision primitives are available in `precourier/decision_engine.py`:
cause probabilities, external signal adjustment, six-action integer economics,
merchant holdout hashing, and provider adapters. Live persistence/API wiring and
the action-expanded backtest remain open limits.
