# Roadmap

| Phase | Deliverable | Status at start of implementation | Exit evidence |
|---|---|---|---|
| 1 | Config, domain, ledger, tenancy | Implemented; invariant hardening complete | ledger, tenancy, immutability, and action-settlement tests |
| 2 | API lifecycle and auto-release | Partial; hold and fail-open invariants hardened | API, terminal/negative-hold, and timeout tests |
| 3 | Worldgen and truth isolation | Partial, generator exists | isolation/determinism tests |
| 4 | Rule risk, feature persistence, optional ML seam | Implemented for deterministic rule path; learned model planned | score artifacts, leakage, calibration tests |
| 5 | Diagnosis and evidence | Implemented for deterministic cause detectors | detector and confusion tests |
| 6 | Gate and policy caps | Implemented for rule path; full provider policy planned | threshold, harm, budget tests |
| 7 | Tools and outcome learning boundary | Partial: explicit simulated verification adapter, durable outbox, provider contracts, and address-repair/message path implemented; live tools planned | simulated action, outbox, outcome, idempotency tests |
| 8 | Three-arm backtest | Implemented for deterministic synthetic run | deterministic CLI and arm tests |
| 9 | Sensitivity and break-even | Partial: declared sensitivity grid and ten-seed aggregation implemented; break-even threshold report planned | ten-seed and sweep output |
| 10 | ROI report and exception list | Partial: live ROI/audit report, deterministic exceptions, and historical CSV projection implemented | reconciled report tests |
| 11 | Merchant console | Partial: authenticated read-only operations view and merchant control-plane API implemented; write controls/onboarding workflow planned | stranger follows order test |
| 12 | LLM and deterministic verification | Partial: deterministic address verifier implemented; LLM/proposal lifecycle planned | crafted rejection tests |
| 13 | Storefront, channel, carrier demo | Missing | browser/integration tests |
| 14 | Webhooks, bandit, compliance | Partial: HMAC primitives implemented; delivery/replay worker, bandit, and compliance operations planned | signature/replay/posterior tests |
| 15 | Payments and release packaging | Missing | signed payment reconciliation and release checklist |

## Execution order

1. Harden existing invariants.
2. Implement deterministic risk, diagnosis, and gate.
3. Implement simulated address repair with durable messages and verification.
4. Implement backtest and reporting.
5. Expose health, audit, ROI, and conversation APIs.
6. Add one real provider at a time under the post-MVP plan.

## Completion rule

Each phase requires source, focused tests, operational impact, security/PII impact, India-readiness impact, and a handoff. Documentation or schema names alone never advance status.
