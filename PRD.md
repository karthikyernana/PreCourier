# PreCourier — Product Requirements Document

**Version 2.0 · 2026-08-28 · supersedes v1.0**

**What it is:** pre-dispatch RTO prevention infrastructure for Indian COD commerce. A multi-tenant decision API plus a merchant console that any store — a custom ceramics shop, a saree label, a supplements brand — connects to their existing order flow. It sits between *order placed* and *label printed*, finds the orders that will come back undelivered, diagnoses why, and repairs the cause before anything is packed.

**Not** a study, a notebook, or a data generator with a dashboard.

**Category:** AI Revenue Recovery (Razorpay Buildathon, Track 03). Ship date 2026-09-04.

---

## 0. Decision, and what changed from v1

Version 1.0 specified a measurement study. The synthetic world was the centre of the system and the merchant-facing surface was a viewer onto it. That is the wrong shape: nobody can integrate a synthetic world, and a judge asked to imagine the product has to do the imagining themselves.

**Decision: build a real multi-tenant service. The synthetic generator survives, demoted, with a genuine product job.**

The reframing that makes both things true at once:

> **The evaluation harness and the merchant's ROI backtest are the same code.**

A merchant with no history needs a projection before they will let software talk to their customers. That projection is produced by running the agent against orders with known outcomes and totalling the rupees. That is precisely what the Buildathon rubric demands. One engine, two audiences — the judges see measured recovery, the merchant sees a business case, and it is not a demo mode bolted on the side. It is the product's onboarding step.

| Concern | v1.0 | v2.0 |
|---|---|---|
| Primary artifact | Batch experiment | Decision API + merchant console |
| Synthetic data role | The product | Cold start, backtest projection, demo tenant seeding |
| Tenancy | Single, implicit | Multi-tenant with per-merchant cost model and API keys |
| Order intake | Generated in-process | REST API, CSV import, store-adapter interface |
| Outcome labels | Known by construction | Ingested from carrier and store webhooks; this is what makes the ML real |
| Insertion point | Implied | Explicit contract: label creation blocks on a release decision |
| Adoption | Assumed | Three modes — shadow, advisory, gateway |
| ML | "A calibrated risk model" | Named models per job, with a rejected-alternatives table |
| Storefront | None | Minimal live storefront so the integration is visible, not described |

---

## 1. Who integrates this, and how

The buyer of this product runs a direct-to-consumer store on Shopify, WooCommerce, or a custom stack, ships 200–20,000 COD orders a month, and loses a fifth of them to RTO. They are not going to hand customer messaging to an unproven system on day one. The product therefore ships as a three-rung ladder, and every rung is a real deployment.

| Mode | What the agent does | What it never does | Purpose |
|---|---|---|---|
| **Shadow** | Scores, diagnoses, decides, and logs everything. Produces the ROI report. | Contacts no buyer. Changes no order. | Trust. The merchant sees what it *would* have done and what it would have saved, on their own orders. |
| **Advisory** | Contacts buyers, repairs addresses, converts to prepaid, writes tags and notes back to the store. | Never blocks fulfilment. | The safe production mode. Merchant's existing flow is untouched. |
| **Gateway** | Holds the order out of label creation until it releases, cancels, or times out. | Never holds beyond `max_hold_minutes`. | Full value. Prevents the packing cost, not just the return. |

```
   ┌──────────┐      ┌──────────┐      ┌──────────┐
   │  SHADOW  │ ───▶ │ ADVISORY │ ───▶ │ GATEWAY  │
   └──────────┘      └──────────┘      └──────────┘
   week 1            week 2            week 4
   "show me"         "talk to them"    "hold the label"
   no risk           no fulfilment     full authority,
                     risk              bounded by policy
```

**Fulfilment latency is sacred.** In gateway mode the agent must produce a decision inside a time budget, and a hard timer auto-releases any order it has not resolved. A product that delays shipping is uninstalled in a week regardless of how much RTO it prevents.

| Guarantee | Target |
|---|---|
| Decision latency, no buyer contact needed | p95 under 5 seconds |
| Decision latency including one buyer round-trip | bounded by `reply_window_hours`, default 4 |
| Hard auto-release | `max_hold_minutes`, default 240 |
| Behaviour on any internal error | Release for dispatch, log, alert. Never hold on failure. |

---

## 2. The insertion point

Everything about this product's design follows from one choice: it acts at the moment the order is still cheap to change.

```
  STORE                        PRECOURIER                     CARRIER
    │                              │                              │
    │  order placed (COD)          │                              │
    ├─────────────────────────────▶│                              │
    │  POST /v1/orders             │ score → diagnose → decide    │
    │                              │                              │
    │                              │ ┌── act: message buyer ──┐   │
    │                              │ │   repair · verify ·    │   │
    │                              │ │   prepaid · cancel     │   │
    │                              │ └── buyer replies ───────┘   │
    │                              │                              │
    │  webhook: order.released     │                              │
    │◀─────────────────────────────┤                              │
    │  (+ patched address)         │                              │
    │                              │                              │
    │  create label ───────────────┼─────────────────────────────▶│
    │                              │                              │
    │                              │   outcome: delivered / RTO    │
    │                              │◀─────────────────────────────┤
    │                              │   POST /v1/outcomes           │
    │                              │   └─▶ becomes a training label
```

The return leg of that diagram is the part v1 was missing and the part that makes this a real system. **The product does not generate its own data. It ingests outcomes from the carrier, and those outcomes are the labels its models train on.** A merchant's first thousand delivered-or-returned orders turn a cold-start rule engine into a fitted model on their own population.

### Order state machine

```
  received ─▶ scoring ─▶ diagnosing ─▶ deciding ─┬─▶ no_action ──────┐
                                                 │                    │
                                                 ├─▶ acting ─▶ awaiting_buyer
                                                 │       │            │
                                                 │       ▼            ▼
                                                 │   repaired · converted ·
                                                 │   verified · escalated
                                                 │            │
                                                 └─▶ cancelled│
                                                              ▼
                                                   released_for_dispatch
                                                              │
                                              (also reached by auto_release_timeout)
                                                              ▼
                                                          shipped
                                                              │
                                                    ┌─────────┴─────────┐
                                                    ▼                   ▼
                                                delivered              rto
                                                    └── POST /v1/outcomes ──▶ training label
```

Every transition writes one audit event. `released_for_dispatch` is the contract event the store waits on.

---

## 3. The money, per merchant

The product's arithmetic is configured per tenant, and this is not a detail — it is why the product must be configured at all. The same order is worth repairing for one merchant and worth cancelling for another, because their freight and margins differ.

Worked example, a ₹1,200 order at a merchant with 30% gross margin:

| Outcome | Computation | Ledger |
|---|---|---|
| Delivered and paid | `0.30 × 1200 − 25 packaging − 70 freight` | **+₹265** |
| Returned to origin | `−(25 + 70 + 20 failed attempt + 70 return freight + 10 goodwill)` | **−₹195** |
| Cancelled before packing | nothing spent | **₹0** |
| Converted to prepaid, delivered | `265 − 24 gateway fee − 20 incentive` | **+₹221** |

The swing between delivering and returning is **₹460**. Cancelling forfeits ₹265 to avoid ₹195. Repairing an address costs about ₹3 and preserves the whole ₹265. That asymmetry is the product.

| Per-merchant config | Set during onboarding | Default |
|---|---|---|
| `gross_margin_rate` | required | 0.30 |
| `packaging_paise` | required | 2,500 |
| `forward_freight_paise` | required | 7,000 |
| `failed_attempt_fee_paise` | from carrier contract | 2,000 |
| `return_freight_paise` | from carrier contract | 7,000 |
| `goodwill_cost_paise` | optional | 1,000 |
| `gateway_fee_rate` | from payment provider | 0.02 |
| `max_incentive_paise` | merchant policy | 12,000 |
| `monthly_incentive_budget_paise` | merchant policy | required |
| `auto_cancel_enabled` | merchant policy | false |
| `value_review_threshold_paise` | above this, cancels need a human | 500,000 |

All money is integer paise. Every constant is a config row, not a literal in code.

---

## 4. Causes and repairs

Risk is not a cause, and one undifferentiated intervention wastes money on orders it cannot help. Each risky order is classified into one *binding* cause with its supporting evidence.

| Cause | Evidence | Action | Cost | Reasoning |
|---|---|---|---|---|
| **Address defect** | no floor or flat, no landmark, vague locality, pincode inconsistent with city, token count below threshold, fails carrier address validation | Repair conversation — ask for the one missing element, verify and patch | ~₹3 | Buyer wants the parcel; the courier cannot find them. Fixing text preserves full margin. |
| **Unreachable contact** | phone fails format or prefix checks, duplicate across accounts, carrier flagged previous attempt unreachable | Verify before packing — confirm the delivery-day number | ~₹2 | A courier who cannot call fails the attempt regardless of address quality. |
| **Weak intent** | impulse-hour order, first-time buyer, no post-order engagement, value far above category norm | Prepaid conversion — small discount to pay online now | ₹18–45, charged only on conversion | Payment is the strongest available proof of intent. |
| **Repeat returner** | prior RTOs on this phone or address, refusal ratio above threshold | Part-payment, or cancel with merchant approval | ~₹1 | Repair cannot fix a buyer who reliably refuses. The only cause where cancelling is rational. |
| **Not serviceable** | pincode not serviceable for COD, or serviceability withdrawn since checkout | Offer prepaid, or cancel with refund of nothing owed | ~₹1 | Deterministic, from carrier data. No model needed. |
| **Uncertain** | risk elevated, no cause dominates | No action, logged with reason | ₹0 | Acting without a diagnosis manufactures false positives. Abstention is a valid output. |

Abstention rate and the rupee consequence of abstaining are reported metrics. An agent that always acts is not deciding.

---

## 5. The gate

Deterministic arithmetic. No model output reaches it except a calibrated probability that has already been range-checked.

```
  act  ⟺  E[ledger | action]  −  E[ledger | no-op]  >  0

    E[ledger | no-op]  =  p·(−L) + (1−p)·M

    E[ledger | action] =  e·[outcome if it works] + (1−e)·[outcome if not]
                          − cost of acting
                          − (1−p)·harm_rate·M        ← charges the agent for
                                                        damaging good orders
```

Because it is arithmetic, thresholds are *derived*, never chosen:

```
  Cancel a ₹1,200 order only if  E[cancel] > E[no-op]:

        ₹0  >  p·(−195) + (1−p)·(265)
       460p >  265
          p >  0.576

  ⟹ the agent is arithmetically forbidden from cancelling below 57.6%.
     Nobody picked that number. Change the freight cost and it moves.
```

The same computation for a ₹3 repair message at believed effectiveness 0.55 gives roughly `3 ÷ (0.55 × 195) ≈ 2.8%`. The gate produces a **ladder**: cheap reversible actions fire often, destructive irreversible ones rarely.

The harm term is what stops the classic failure of this category. An agent rewarded for preventing returns will cancel good orders, because a cancelled order never appears as an RTO.

| Guardrail | Rule |
|---|---|
| One paid action per order per lifecycle | prevents message stacking |
| Incentive ceiling and monthly budget | gate refuses once exhausted |
| Quiet hours 21:00–08:00 local | no buyer contact |
| One outbound message per buyer per 24h | frequency cap |
| Consent flag required per channel | no flag, no message, logged |
| Idempotency key on every money action | replay is a no-op |
| Fail closed on channel or gateway error | hold, escalate once, no retry storm |
| Auto-release on timeout | fulfilment is never blocked indefinitely |
| Cancels above `value_review_threshold` | require merchant confirmation |
| Never-contact list | honoured before all other logic |

---

## 6. Why this is an agent

| Property | Implementation |
|---|---|
| Goal-directed autonomy | Given a queue and a budget, it decides per order whether to act and which of six actions to take. No fixed script. |
| Tool use | Eight tools with schemas and preconditions: `send_message`, `create_payment_link`, `validate_address`, `apply_address_patch`, `check_serviceability`, `hold_order`, `cancel_order`, `escalate_to_merchant`. |
| Multi-turn interaction | Buyer replies re-enter the loop as evidence; it re-diagnoses, patches, re-asks once, or gives up. |
| State and memory | Per-order state machine, per-buyer history, per-merchant learned effectiveness. |
| Planning under scarcity | A shared budget makes actions compete; spend goes where expected recovery per rupee is highest. |
| Learning from consequence | Realised effectiveness per action-cause pair updates from ingested outcomes. |
| Bounded authority | Gate plus policy caps; high-value cancels need a human. |
| Self-assessment | Reports its own costliest mistakes without being asked. |

The LLM's scope is narrow and stated plainly in the repository: **it reads and writes language and never touches money.**

| LLM does | LLM does not |
|---|---|
| Extract structured address fixes from free text | Produce a risk probability |
| Compose the buyer message in their language and register | Choose the action |
| Parse a free-text reply into a structured patch | Set any rupee amount |
| Draft the merchant-facing exception note | Approve a cancellation |

---

## 7. Machine learning

### Model per job

| Job | Model | Why this one |
|---|---|---|
| **P(RTO) — the risk score** | Gradient-boosted trees (LightGBM or XGBoost), with **isotonic regression calibration** on a held-out split | Tabular, mixed categorical and numeric, non-linear interactions, thousands not millions of rows. Best-in-class on exactly this shape. Feature importances and SHAP give per-order explanation, which the console requires. |
| **Mandatory baseline** | Logistic regression on the same features, plus a hand-written rule score | The GBM ships only if it beats an interpretable baseline on AUC *and* calibration. Otherwise the simpler model ships. |
| **Cold start, zero merchant data** | Rule score behind the identical interface | The agent must work on a merchant's first order, not their thousandth. |
| **Cause diagnosis** | Per-cause detectors — deterministic checks, reference lookups, and carrier validation — feeding a binding-constraint selector. Multinomial logistic only to break genuine ties. | The console must show *evidence*, not a label. Most of this is correctly not machine learning, and the repository says so. |
| **Address quality** | Token and pattern features, pincode↔locality reference match, character-n-gram similarity against known-good addresses in the same pincode | Cheap, explainable, no embedding infrastructure. Deliverability is a structural property of the text, not a semantic one. |
| **Free-text extraction and composition** | LLM, temperature 0, JSON schema output, deterministic verification layer, regex fallback | The one job where language understanding is genuinely required. |
| **Which action works for which cause** | **Beta-Bernoulli Thompson sampling per (action, cause) arm** — a contextual bandit | This is the agentic ML. The agent starts with priors, acts, observes ingested outcomes, and updates its own effectiveness estimates. Implementable in an afternoon and honest about uncertainty. |
| **Theoretically correct alternative** | Uplift / treatment-effect modelling (T-learner or X-learner) | The right question is not "will this order return" but "will this action change whether it returns". Named explicitly, with the bandit justified as the tractable form at this data scale. Upgrade path documented. |

### Explicitly rejected

| Rejected | Reason |
|---|---|
| Deep learning on tabular features | No accuracy gain at this scale, worse calibration, no per-order explanation, heavier to serve |
| Embedding-first address matching | Solves semantic similarity; deliverability needs structural completeness |
| A single multiclass cause classifier | Produces a label without evidence, and the console needs evidence |
| LLM-generated risk scores | Uncalibrated, unstable across runs, unauditable, and in the money path |
| Predicting RTO probability alone and thresholding it | Ignores treatment effect and margin; this is what the incumbents do and why blocking underperforms |

### Features

| Group | Examples |
|---|---|
| Address structure | token count, has floor or flat, has landmark, has locality in reference list, pincode valid, pincode↔city consistent, digit ratio, repeated characters, carrier validation verdict |
| Contact | phone format valid, prefix valid, duplicate-phone count, email present and plausible, contact verified previously |
| Buyer history | prior orders, prior RTOs, refusal ratio, days since last order, prior prepaid share, prior reply rate |
| Order | value, value relative to category median, item count, category, discount depth, first-time buyer, hour of day, day of week |
| Geography | pincode RTO base rate, tier, distance band, COD serviceability, monsoon or festival flag |
| Post-contact | reply received, reply latency, reply produced a valid patch |

Post-contact features are available only after an action, so they feed the second-pass decision, never the first. Leakage boundary is enforced by a test.

### Calibration is a first-class requirement

The gate multiplies `p` by rupees, so a miscalibrated model produces confidently wrong money decisions even when every line of code is correct.

| Reported always | Meaning |
|---|---|
| Brier score | overall probabilistic accuracy |
| Expected calibration error | do the 30% buckets return 30% of the time |
| Reliability curve | where the model is over- and under-confident |
| AUC / PR-AUC | ranking quality, against the baseline |

### Production ML operations

| Concern | Handling |
|---|---|
| Cold start | Rule score, priors from the synthetic population, replaced per-merchant once `min_labels_to_train` outcomes have arrived |
| Retraining | Scheduled on ingested outcomes, per merchant, with a held-out window |
| Drift | Population stability index on features and calibration drift monitored; breach raises an alert |
| Auto-fallback | Calibration degradation past a bound reverts to the rule score and notifies |
| Model registry | Version, training window, metrics, and feature list recorded; every stored score references its model version |
| Shadow evaluation | A new model runs alongside the live one before promotion |

---

## 8. The verification layer

Every LLM output is an untrusted proposal and must pass deterministic checks before it can change an order. This is the Buildathon's own stated thesis — verification capacity, not generation speed, is the bottleneck — applied literally.

| Check | Rejection reason |
|---|---|
| Parses as the declared JSON schema | `schema_invalid` |
| Extracted pincode exists in the reference table | `pincode_unknown` |
| Extracted pincode maps to the same city as the order | `pincode_city_mismatch` |
| Patched address retains original locality tokens | `locality_dropped` |
| Patch adds information rather than rewriting wholesale | `rewrite_too_aggressive` |
| Patched address passes carrier validation | `carrier_rejected` |
| Message contains no invented discount, date, or promise | `unauthorised_claim` |
| Template variables and length within channel limits | `template_violation` |

A rejection is never a silent fall-through: the original address is kept, the agent re-asks once or escalates, and the rejection is logged. Rejection rate and its rupee cost are reported, because a verification layer that never rejects anything is untested.

---

## 9. Platform architecture

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │ STOREFRONT (demo)   MERCHANT CONSOLE          PUBLIC REST API         │
 │ one-page shop       queue · order · chat ·    /v1/orders  /v1/outcomes│
 │ COD checkout        policy · money · audit    /v1/backtest /v1/reports│
 └───────────────┬───────────────┬───────────────────────┬───────────────┘
                 │               │                       │
                 ▼               ▼                       ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ TENANCY & AUTH — merchant_id scoping, pk_test/pk_live keys,           │
 │ HMAC-signed outbound webhooks, idempotency keys, rate limits          │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ AGENT PIPELINE — detect → diagnose → decide → act → measure           │
 │ one audit event per stage, one decision record per order              │
 └──┬─────────┬──────────┬──────────┬───────────┬────────────────────────┘
    ▼         ▼          ▼          ▼           ▼
 ┌──────┐ ┌────────┐ ┌──────┐ ┌────────┐ ┌──────────┐
 │ risk │ │diagnose│ │ gate │ │ tools  │ │ outcomes │
 │ +cal │ │+verify │ │      │ │        │ │ ingest   │
 └──────┘ └───┬────┘ └──────┘ └───┬────┘ └────┬─────┘
              ▼                   ▼           │
          ┌───────┐   ┌───────────────────┐   │ labels
          │  llm  │   │ ADAPTERS          │   │
          └───────┘   │ store · carrier · │   ▼
                      │ messaging · pay   │ ┌──────────┐
                      └───────────────────┘ │ training │
                                            │ + bandit │
                                            └──────────┘
 ┌───────────────────────────────────────────────────────────────────────┐
 │ STORE — Postgres-shaped schema on SQLite. Append-only ledger.         │
 │ Hash-chained audit log. Per-tenant isolation.                         │
 └───────────────────────────────────────────────────────────────────────┘
 ┌───────────────────────────────────────────────────────────────────────┐
 │ BACKTEST ENGINE — the same agent against orders with known outcomes.  │
 │ Serves the merchant's ROI report AND the submission's measured results│
 └───────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Responsibility | Third-party |
|---|---|---|
| `config` | Per-merchant cost model and policy; integer paise | none |
| `domain` | Order, Buyer, Features, Cause, Action, Decision, Outcome | none |
| `tenancy` | merchant_id scoping, API keys, isolation | none |
| `api` | Public REST surface, idempotency, validation | fastapi |
| `pipeline` | Five-stage orchestration, audit per stage | none |
| `risk` | Feature build, GBM + calibration, rule fallback, model registry | lightgbm (optional) |
| `diagnose` | Per-cause detectors, evidence, binding-constraint selector | none |
| `verify` | Deterministic checks on LLM proposals | none |
| `gate` | Expected-ledger arithmetic, caps, budget, quiet hours | none |
| `tools` | The eight tools, preconditions, idempotency | none |
| `bandit` | Beta-Bernoulli effectiveness learning per action-cause | none |
| `outcomes` | Ingest delivery results, produce training labels | none |
| `ledger` | Append-only money entries; balance by summation | none |
| `audit` | Hash-chained event log | none |
| `adapters.store` | Shopify / WooCommerce / generic REST / CSV | httpx, lazy |
| `adapters.carrier` | Serviceability, address validation, label hold, NDR feed | httpx, lazy |
| `adapters.messaging` | WhatsApp Cloud, SMS, Telegram, in-app simulated | httpx, lazy |
| `adapters.payments` | Razorpay Payment Links, HMAC webhook | httpx, lazy |
| `backtest` | Agent vs baselines on known outcomes; ROI and sensitivity | none |
| `console` | Merchant web UI | fastapi, jinja2 |
| `storefront` | One-page demo shop with COD checkout | fastapi, jinja2 |
| `worldgen` | Synthetic population for cold start, demo seeding, and the submission's controlled experiment | none |

**Hard requirement retained from v1: the backtest runs on the standard library alone.** `python -m precourier.backtest --seed 42` reproduces every published number with nothing installed. Network and ML libraries are imported at call time, never at module import, and a rule-score path exists so LightGBM is optional.

### Public API

| Endpoint | Purpose |
|---|---|
| `POST /v1/orders` | Submit an order for decision. Returns decision or `pending` with a poll URL. |
| `GET /v1/orders/{id}` | Current state, risk, cause, decision, reason, audit trail |
| `POST /v1/orders/{id}/release` | Manual override — release for dispatch |
| `POST /v1/orders/{id}/cancel` | Merchant-approved cancel |
| `POST /v1/outcomes` | Delivery result ingestion; becomes a training label |
| `POST /v1/backtest` | Historical CSV or order range → projection |
| `GET /v1/reports/roi` | Recovered rupees, spend, per-cause breakdown, exception list |
| `GET /v1/audit` | Hash-chained log, exportable |
| `GET /v1/health` | Adapter, model, and calibration status |

| Outbound webhook | Fired when |
|---|---|
| `decision.made` | Gate produces an action or a no-op |
| `action.taken` | A tool executed |
| `order.repaired` | Address patched and verified |
| `order.converted_prepaid` | Payment received |
| `order.cancelled` | Cancel executed |
| `order.released` | Safe to create the label — the contract event |
| `escalation.raised` | Human attention required |

Auth is `Bearer pk_test_…` / `pk_live_…`. Outbound webhooks are HMAC-SHA256 signed. All mutating endpoints accept an idempotency key.

### Data model

| Table | Mutability |
|---|---|
| `merchants`, `api_keys`, `merchant_config`, `users` | mutable |
| `buyers` — consent flags, never-contact, history counters | mutable |
| `orders` — value_paise, address, state, mode | state transitions only |
| `features`, `risk_scores`, `diagnoses`, `decisions`, `actions`, `messages` | immutable |
| `payments`, `outcomes`, `ledger_entries`, `audit_log` | append-only |
| `models` — version, metrics, feature list, training window | append-only |
| `bandit_state` — alpha, beta per merchant, action, cause | mutable, versioned |
| `ground_truth` — backtest only, isolated from the agent's read path | append-only |

| Design rule | Reason |
|---|---|
| Integer paise everywhere | Float rupees destroy an audit trail's credibility |
| Ledger append-only; corrections are compensating entries | A test asserts no row is mutated or deleted |
| Audit log hash-chained, `sha256(prev_hash + payload)` | Gaps and tampering are detectable |
| Ground truth unreachable from the agent's code path | Prevents the circularity that would void every metric |
| Every decision stores a reason string and its gate inputs | An unexplained decision is indistinguishable from a bug |
| Tenant isolation asserted by test | Cross-merchant leakage is the one unrecoverable bug |

---

## 10. Merchant console

| Module | Contents |
|---|---|
| **Onboarding** | Connect store, enter cost model, import history, run backtest, choose mode |
| **ROI report** | Projected or realised recovery, spend, net, per-cause breakdown, break-even assumption stated |
| **Live queue** | Pre-dispatch orders with risk, cause, evidence, decision, expected value, reason; hold, release, override |
| **Order detail** | Full timeline: features → score → cause with evidence → **the gate arithmetic rendered as arithmetic with real numbers** → action → buyer thread → outcome → ledger entry |
| **Conversations** | Buyer threads, extraction results, verification verdicts, human takeover |
| **Policy** | Budget, incentive ceiling, quiet hours, auto-cancel toggle, per-category rules, never-contact list, escalation routing |
| **Money** | Ledger, recovered rupees, spend, cost per return prevented, exportable |
| **Model health** | Calibration curve, drift indicators, active model version, fallback status |
| **Where I was wrong** | Auto-generated: the costliest decisions, false-positive cancels, rejected extractions, channel failures |
| **Developers** | API keys, test mode, webhook log with replay, docs |
| **Team** | Users and roles: owner, ops, viewer |

The gate arithmetic is rendered on screen with the actual numbers substituted, because a decision a merchant can verify by hand earns more trust than a confidence score.

---

## 11. Integrations

| Layer | Demo path (always works) | Live path |
|---|---|---|
| **Store intake** | Built-in storefront + CSV import | Generic REST; Shopify and WooCommerce as adapter implementations behind one interface |
| **Carrier** | Simulated carrier with a real pincode reference table | Shipping-aggregator adapters (Shiprocket, Delhivery, Bluedart, India Post); USPS adapter for non-India tenants. One interface: `serviceability`, `validate_address`, `hold_label`, `ndr_feed` |
| **Messaging** | In-app simulated channel; Telegram Bot API for a real phone in the demo | WhatsApp Cloud API with pre-approved utility templates; SMS over a DLT-registered sender |
| **Payments** | Simulated payment link | Razorpay test-mode Payment Links with HMAC-verified webhook driving real prepaid conversion |
| **LLM** | Deterministic regex extractor | Any OpenAI-compatible endpoint via `LLM_BASE_URL` |

Every integration degrades to a working simulated path. The backtest never touches a network and the demo never depends on a live key.

**Adapter interface, not vendor lock:** each layer is one small class. Adding a carrier or a channel means implementing three or four methods. This is stated in the repository because it is the difference between a product and a script.

---

## 12. Data strategy

The awkward truth of this build is that there is no real merchant data, and the spec is explicit about it rather than hiding it behind a demo.

| Stage | Data source | Product purpose |
|---|---|---|
| **Cold start** | `worldgen` synthetic population + rule score | The agent works on a merchant's first order. Priors for the bandit. |
| **Onboarding backtest** | Merchant's historical CSV, or synthetic if they have none | The ROI projection that earns permission to go live |
| **Live operation** | Real orders through the API | Decisions and actions |
| **Learning** | Outcomes ingested from carrier and store | Real labels; per-merchant retraining; bandit updates |
| **Submission evidence** | `worldgen` with hidden latents and isolated ground truth | The controlled three-arm experiment |

### Synthetic world design, for the controlled experiment

If ground truth were derived from the same features the agent scores, the agent would be grading its own homework.

```
   HIDDEN LATENTS                  ← never visible to the agent
   reliability · responsiveness · address_quality · purchase_intent
        │
        ├──────────────▶ TRUE OUTCOME      p_return = f(latents)
        │                                  true_cause = binding constraint
        │
        └──────────────▶ OBSERVABLE FEATURES  ← all the agent sees
                         emitted through a DIFFERENT noisy mapping
                                │
                                ▼
                         AGENT'S RISK MODEL
```

Two rules make it trustworthy. **Latents are shared, mappings are independent** — shared latents mean real signal exists, and independent functional forms mean the agent can be genuinely good and genuinely wrong. **Effectiveness lives in the world, not the agent** — the world holds the true probability each action repairs each cause, the agent holds deliberately mis-calibrated beliefs, and the report shows believed against realised.

| Swept parameter | Values |
|---|---|
| `repairable_share` — fraction of risk from fixable defects | 0.60 / 0.40 / 0.20 |
| `effectiveness_haircut` | 1.0 / 0.75 / 0.50 |
| `feature_noise` | 1× / 2× / 3× |
| `base_rto_rate` | ±50% |
| freight and margin | ±30% |
| `seed` | 10 seeds, mean and standard deviation |

`repairable_share` is the assumption the product's value depends on most, so it is declared and swept, and the report publishes the value at which the agent stops beating doing nothing.

---

## 13. Measurement and honesty

Three policies run against the same orders, the same outcomes, and the same ledger function. Only the decision layer differs.

| Arm | Policy |
|---|---|
| `ship_all` | Dispatch everything — what most merchants do today |
| `block_best` | Cancel above a risk threshold, **swept to its best threshold** |
| `agent` | Diagnose the cause, take one gated action |

The blocking arm is tuned to its optimum on purpose. Beating a badly-configured incumbent is a comparison that does not survive one question from a judge.

| Metric | Definition |
|---|---|
| Net ledger per order | Total balance ÷ orders. The headline. |
| Net vs `ship_all` | Whether any of this was worth building |
| Net vs `block_best` | Whether cause-matched repair beats blunt refusal |
| Returns prevented | Baseline RTOs that did not occur |
| False-positive cancels | Cancelled orders ground truth says would have delivered, and the forfeited margin in ₹ |
| Wasted spend | Paid actions on orders that would have delivered anyway |
| Precision / recall at the acting threshold | On act vs no-act, not on a tuned classifier |
| Brier, ECE, reliability curve | Calibration |
| Cost per return prevented | Spend ÷ returns prevented |
| Abstention rate | Share of risky orders left alone |
| LLM rejection rate | Share of extractions refused by verification |
| Believed vs realised effectiveness | Per action-cause pair |

### Numbers discipline

| Rule | Reason |
|---|---|
| No external market statistic in any deliverable without a citation and a date | An uncited figure invites the one question that cannot be answered |
| Every unit economic labelled illustrative and configurable, in one place | The structure is the contribution; constants are merchant inputs |
| Headline results are whatever the harness prints, including a negative one | A reported loss with a correct explanation beats an unverifiable win |
| No number in the video or README the harness cannot reproduce from a seed | Checkable in thirty seconds by anyone cloning the repository |
| Synthetic data disclosed as synthetic on every surface that displays it | Undisclosed synthetic results are the one unrecoverable credibility failure |

### Where this agent is wrong

A required, auto-generated report section: the ten decisions that cost the most money, each with the features seen, the cause inferred, the action taken, the ground truth, and the rupee consequence. Plus every false-positive cancel, every rejected extraction, every channel failure. Not a disclaimer — the deliverable that separates a measured system from a demo.

---

## 14. Compliance and security

| Requirement | Implementation |
|---|---|
| Consent before contact | Per-buyer, per-channel flag; absent flag logs `blocked_no_consent` |
| WhatsApp template pre-approval | Production sends use pre-approved utility templates; IDs in config, variables validated pre-send |
| DLT registration for SMS | Sender ID and template registration documented as a production precondition |
| Quiet hours | No contact 21:00–08:00 local |
| Frequency cap | One message per buyer per 24 hours |
| Immediate opt-out | Reply keyword clears consent and cancels pending actions |
| No dark patterns | The prepaid offer always preserves the COD option in the message body |
| PII minimisation | Phone hashed for analytics; plaintext only where operationally required; no PII in exported reports |
| Tenant isolation | `merchant_id` scoping enforced at the query layer and asserted by test |
| Secrets | Keys from environment only; never committed, never logged, never rendered |
| Webhook authenticity | HMAC verification inbound and signing outbound; replay window enforced |
| Retention | Actor, action, timestamp, and reason retained for every buyer contact |

---

## 15. Scope

| Priority | Item |
|---|---|
| **P0** | Per-merchant cost model, integer-paise ledger, append-only |
| **P0** | Tenancy, API keys, `merchant_id` scoping, isolation test |
| **P0** | `POST /v1/orders` and `POST /v1/outcomes`, idempotent |
| **P0** | Order state machine with auto-release timeout |
| **P0** | Three modes: shadow, advisory, gateway |
| **P0** | Rule risk score + GBM with isotonic calibration and a rule fallback |
| **P0** | Cause diagnosis with evidence |
| **P0** | The gate, with the harm term and all policy caps |
| **P0** | Hash-chained audit log |
| **P0** | Backtest engine: three arms, `block` swept, sensitivity, break-even |
| **P0** | Exception list — where the agent is wrong, in rupees |
| **P0** | Merchant console: queue, order detail with rendered arithmetic, ROI, audit |
| **P0** | Zero-dependency backtest reproduction path |
| **P1** | Demo storefront with COD checkout posting to the live API |
| **P1** | LLM extraction + verification layer + regex fallback |
| **P1** | Buyer conversation, one real channel (Telegram) end to end |
| **P1** | Carrier adapter interface + simulated carrier + pincode serviceability and validation |
| **P1** | Outbound webhooks with signing and a replay log |
| **P1** | Beta-Bernoulli bandit updating effectiveness from outcomes |
| **P1** | Compliance guardrails enforced in code |
| **P2** | Razorpay test-mode Payment Links + verified webhook |
| **P2** | Shopify adapter implementation (interface and stub in P1) |
| **P2** | WhatsApp Cloud API channel |
| **P2** | Model health page with drift indicators |
| **P2** | Multi-language message composition |
| **P2** | Chaos mode: channel and gateway failure during the demo |

### Non-goals

| Not building | Why |
|---|---|
| Cross-merchant identity or address network | Requires data volume no single merchant has. This is the incumbents' real moat and pretending otherwise would be dishonest. Named in the README as a limitation. |
| NDR or post-attempt recovery | The freight is already spent — a different product |
| Courier allocation or rate shopping | Adjacent, no shared mechanism |
| Payment fraud or chargebacks | Different failure mode, different data |
| A production Shopify app with OAuth review | Review timelines make it impossible; the adapter interface proves the design instead |
| Real customer data | No lawful source; synthetic origin disclosed everywhere |
| A marginally better classifier | The contribution is the decision layer |

---

## 16. Build order

| Phase | Deliverable | Done when |
|---|---|---|
| 1 | `config`, `domain`, `ledger`, tenancy | A hand-written order produces the correct ledger value for all four outcomes; two merchants with different cost models produce different thresholds |
| 2 | `api` skeleton, order state machine, auto-release | An order posted over HTTP reaches `released_for_dispatch` and the timeout path is tested |
| 3 | `worldgen` with hidden latents and isolated ground truth | A test proves the agent's modules cannot import ground truth; feature-outcome correlation is well above chance and well below 1 |
| 4 | `risk`: rule score, then GBM + isotonic, model registry | Brier, ECE, reliability curve printed; GBM beats the baseline or the baseline ships |
| 5 | `diagnose` + evidence | Confusion matrix of inferred against true cause, printed |
| 6 | `gate` + caps | Derived thresholds match hand computation; a test proves the harm term prevents cancelling a low-risk high-margin order |
| 7 | `tools` + `outcomes` ingestion | Actions change the world through world-owned effectiveness; outcomes become labels |
| 8 | `backtest`: three arms, `block` swept | All arms on one seed, reproducing exactly on re-run |
| 9 | Sensitivity + break-even + 10 seeds | Net per order at each `repairable_share` and haircut, with standard deviations |
| 10 | `report` + exception list | Generates end to end including the ten costliest mistakes |
| 11 | `console`: queue, order detail with rendered arithmetic, ROI, audit | A stranger follows one order from intake to ledger entry |
| 12 | `llm` + `verify` + fallback | Works with and without a key; every rejection reason fires on a crafted input |
| 13 | Storefront + Telegram + simulated carrier | A real order placed in a browser triggers a real message to a real phone and a verified patch |
| 14 | Webhooks out, bandit, compliance enforcement | Signed deliveries logged and replayable; bandit posteriors move with outcomes |
| 15 | Razorpay test mode; README, architecture diagram, video | Every number in the video traces to a seed |

### Test matrix

| Test | Asserts |
|---|---|
| Ledger arithmetic | All four outcomes at several values and two cost models |
| Ledger immutability | No update or delete; balance equals sum of entries |
| Audit chain | Chain validates; a tampered row is detected |
| Tenant isolation | No query returns another merchant's rows |
| Ground-truth isolation | Agent modules cannot reach ground truth |
| Anti-circularity | Feature-outcome correlation within declared bounds |
| Feature leakage | Post-contact features never enter the first-pass score |
| Gate: cancel threshold | Matches the closed-form derivation per merchant |
| Gate: harm term | A low-risk high-margin order is never cancelled |
| Gate: budget | Actions stop when the budget is exhausted |
| Gate: quiet hours, consent, never-contact | No message sent when any rule is violated |
| Idempotency | Replaying a request creates no second ledger entry |
| Auto-release | An unresolved order releases at `max_hold_minutes` |
| Fail closed | Channel error holds and escalates once, with no retry storm |
| Fail open on internal error | An exception releases the order rather than blocking it |
| Verification layer | Each rejection reason fires on a crafted bad input |
| Calibration gate | A model failing an ECE bound cannot be promoted |
| Determinism | Same seed, byte-identical backtest results |
| Arm comparability | All arms see identical orders and identical ground truth |
| Webhook signing | Signature verifies; a tampered body fails |

---

## 17. Demo script, five minutes

| Time | Beat |
|---|---|
| 0:00–0:30 | The leak, in rupees. One ₹1,200 order: +₹265 delivered, −₹195 returned, ₹0 cancelled. The ₹460 swing. |
| 0:30–1:10 | Onboard a merchant, "Kolam Ceramics". Enter their freight and margin. Import 2,000 historical orders. Backtest returns the ROI projection and states its own break-even assumption. |
| 1:10–1:30 | Switch shadow → gateway. |
| 1:30–3:00 | Place a real COD order in the storefront with a deliberately incomplete address. The order is held. Console shows risk, cause `address_defect`, and the evidence. A message arrives on a real phone. Reply with the flat number. Extraction, verification, patch, `order.released` webhook fires. |
| 3:00–3:50 | A second order: repeat returner, risk above 57.6%. The gate shows the arithmetic on screen and chooses prepaid conversion over cancelling. Razorpay test payment link, real webhook, converted. |
| 3:50–4:30 | The report: three arms, agent against best-tuned blocking, sensitivity grid, break-even. Then "Where I was wrong" — the ten costliest mistakes, unprompted. |
| 4:30–5:00 | Audit log for one order, hash chain verified. The limitation stated aloud: no cross-merchant network, synthetic population, constants are merchant inputs. |

---

## 18. Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Platform surface consumes the measurement work | Loses the rubric | Backtest is P0 and doubles as onboarding; console is thin and reuses it |
| Circularity leaks into the world | Every metric meaningless | Module isolation, isolation test, correlation bounds test |
| Risk model uncalibrated | Correct code, wrong money | Calibration is P0, gated at promotion |
| `repairable_share` optimistic | Value evaporates in reality | Declared, swept, break-even published |
| Agent beliefs equal world truth | Results look excellent and mean nothing | Deliberate mis-calibration; believed vs realised reported |
| Single lucky seed | Does not survive one question | Ten seeds with standard deviations |
| Strawman blocking baseline | Comparison collapses | Swept to its best threshold |
| Gateway mode delays fulfilment | Uninstalled in a week | Hard auto-release, fail-open on error, p95 latency target |
| Live integrations eat the schedule | P0 work cut | Simulated paths always work; Razorpay and Shopify are P2 |
| Dependency install fails for a judge | Claims uncheckable | Backtest on the standard library alone; LightGBM optional behind a rule fallback |
| Uncited market statistics | An unanswerable question in judging | Numbers discipline, §13 |
| Synthetic data mistaken for real | Unrecoverable credibility failure | Disclosed on every surface and stated aloud in the video |

---

## 19. Definition of done

A merchant who has never seen this software can create an account, enter their own freight and margin, import a CSV of past orders, receive a projection that names its own break-even assumption, switch to gateway mode, and watch a real order placed in a browser get held, diagnosed, repaired through a message to a real phone, and released for dispatch — with every step readable in an audit log whose hash chain verifies.

A stranger can clone the repository, run one command with nothing installed, and reproduce every number in the video. The report states the net rupee effect per order against both doing nothing and best-tuned blocking, across ten seeds. The break-even assumption is named. The ten costliest mistakes the agent made are printed without being asked for.

A negative result meeting all of the above is a valid submission. An unverifiable positive one is not.
