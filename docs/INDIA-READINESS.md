# India Readiness

## Current fit

The current domain is appropriate for an India-first prototype: INR economics in paise, Indian phone normalization, PIN reference data, locality/city checks, and Asia/Kolkata-style quiet-hour policy. It is not yet a global order model.

## Small Indian merchant

**Initial fit:** managed advisory pilot. Requirements are one store, one dispatch operation, one shipping path, one approved communication channel, several hundred historical delivered/RTO outcomes, and an operator who reviews exceptions.

## Global vendor operating in India

**Conditional fit:** only when the vendor has an Indian entity or operating boundary, Indian warehouse/carrier/payment accounts, a defined India data boundary, and enterprise access/security requirements. Multi-store, multi-warehouse, tax, currency, SSO, retention, and procurement controls are post-MVP scope.

## Integration truth

- Shopify requires a platform-specific app/integration lifecycle, current API version, scopes, privacy compliance, webhook verification, and fulfilment semantics. Do not promise a public app until those are implemented.
- WooCommerce is a viable early connector but needs REST keys, HTTPS, webhook monitoring, custom-field handling, and polling fallback.
- Custom REST is the best first technical integration because the merchant can expose an explicit order-ready-for-fulfilment boundary.
- Shiprocket or another aggregator must be selected with the design partner. The adapter must prove pre-label hold, address change, cancellation, serviceability, shipment events, and outcome mapping.
- Static PIN reference data is an address consistency aid, not a carrier COD serviceability source.
- WhatsApp requires provider/business/template/consent/delivery/opt-out handling. SMS requires current telecom/DLT provider validation. Telegram requires an opted-in bot conversation and is not equivalent to phone outreach.
- Razorpay must remain merchant-owned. PreCourier creates and reconciles payment intents/links only after signed provider callbacks. It must not operate a wallet, PPI, pooled balance, or settlement system.

## Data requirements before production

Add country, currency, line items/SKUs, tax lines, shipping/COD charges, discount semantics, total payable, warehouse, merchant order version, store source, shipment/AWB/provider references, preferred language, alternate contact, delivery instructions, and original/transliterated address fields.

## Compliance and security gate

Before production claims, obtain current legal/provider review for DPDP obligations, notice and purpose, consent and withdrawal, retention/deletion/anonymisation, breach response, telecom messaging, payment boundaries, GST/tax treatment, and platform privacy requirements. Code-level consent booleans and hashed phones are not a complete compliance program.
