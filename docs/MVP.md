# India-First MVP

## Objective

Build a truthful, sellable managed pilot for Indian D2C merchants. The MVP proves one complete path from COD order intake to a measured, auditable address repair before dispatch. The current implementation uses an in-app simulated channel; it does not claim that a customer was contacted on WhatsApp, SMS, Telegram, or a carrier network.

## In scope

- Multi-tenant API keys and merchant economics.
- Integer-paise ledger, derived gate arithmetic, and append-only audit.
- Deterministic observable feature construction.
- Rule risk score with versioned score persistence.
- Evidence-based diagnosis for address defect, unreachable contact, weak intent, repeat returner, not serviceable, and uncertain.
- One safe action: address repair.
- Persisted simulated message and inbound buyer reply in the explicit verification adapter; no simulated provider is selected by the production app factory.
- PIN/city/locality and carrier-validation boundary checks.
- Bounded gateway hold and release timeout.
- Delivered/RTO/cancelled outcome ingestion.
- Deterministic three-arm backtest, sensitivity, break-even, and exception output.
- ROI, audit, health, and backtest API reports.
- Small operational view or JSON report that allows an operator to follow one order end to end.

## Explicit exclusions

- Live WhatsApp, SMS, Telegram, carrier, store, or payment provider calls in the first vertical slice.
- Prepaid conversion and part payment.
- Automatic cancellation selected by the model.
- Public Shopify App Store or WooCommerce marketplace distribution.
- Global multi-country phone, currency, tax, or timezone support.
- Claims that synthetic data represents merchant performance.

## Product contract

```text
valid authenticated order + decision timeout/internal failure -> release known order
invalid request/auth failure/unknown order -> error, no inferred ship instruction
verified address patch -> append address revision, audit, release according to mode
unverified patch -> retain original address, log rejection, re-ask once or escalate
hold -> bounded lease with an independent timeout sweep
```

## Release gate

The MVP is not complete until:

1. The default service produces a real risk, diagnosis, gate decision, and persisted evidence.
2. A simulated address-repair conversation can release a held order only after verification.
3. All action and message attempts are durable and idempotent.
4. Outcome ingestion produces the expected ledger and training-label-ready record.
5. A deterministic backtest reproduces byte-identical output for a seed.
6. ROI output reconciles to ledger entries and includes synthetic/live provenance.
7. Tenant, PII, consent, timeout, provider-failure, and replay tests pass.
8. No advertised feature remains schema-only or a facade.
