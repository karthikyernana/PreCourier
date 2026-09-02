# Test Matrix

## Existing evidence

The baseline suite contains 168 passing tests across API, audit, guards, ledger, orders, and tenancy. It proves the current foundation but not the missing product layers.

## MVP additions

| Area | Required assertion | Status |
|---|---|---|
| Features | first-pass features exclude post-contact data | Implemented, `tests/test_intelligence.py` |
| Risk | deterministic same-input score and persisted model metadata | Implemented, `tests/test_intelligence.py`, `tests/test_mvp_flow.py` |
| Diagnosis | one binding cause with evidence and runner-up | Implemented, `tests/test_intelligence.py`, `tests/test_mvp_flow.py` |
| Gate | derived threshold and harm term match hand arithmetic | Implemented, `tests/test_intelligence.py` |
| Messages | outbound/inbound rows, provider status, frequency cap, replay safety | Partial, simulated lifecycle covered; provider outbox and reconciliation planned |
| Verification | PIN unknown, city mismatch, locality dropped, aggressive rewrite, carrier rejection | Implemented for deterministic address path, `tests/test_verify.py`; LLM proposal and carrier verdict adapters planned |
| Lifecycle | verified patch releases; rejected patch preserves original; timeout releases | Implemented, `tests/test_mvp_flow.py` and existing API/order tests |
| Backtest | identical arms, same seed, byte-identical report | Implemented, `tests/test_backtest.py` |
| ROI | report totals reconcile to ledger and labels synthetic data | Implemented for live persisted outcomes and synthetic reports |
| PII | ordinary exports mask/remove operational PII | Partial, existing audit tests cover intake; full export boundary planned |
| Tenant security | no cross-tenant feature, message, outcome, or report access | Partial, existing tenancy tests plus scoped reasoning/report queries |
| Decision latency | deterministic rule path stays below configured budget in local check | Implemented, `tests/test_performance.py`; production p95 monitoring planned |

## Post-MVP additions

Provider contract tests, HMAC signing/replay, payment callback verification, bandit posterior movement, calibration promotion gate, model drift, Shopify privacy webhooks, WooCommerce fallback, carrier corrections, retention/deletion, regional-language extraction, SSO/RBAC, backup restore, and chaos-mode tests.

## Required command evidence

At each phase run the full Python suite, focused tests for changed behavior, compile checks, and the deterministic backtest command once it exists. Record failures rather than converting them into degraded success claims.
