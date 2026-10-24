"""Per-merchant cost model and policy.

PRD section 3: "Every constant is a config row, not a literal in code." That is
the reason this file exists and the reason it is boring. The same order is worth
repairing for one merchant and worth cancelling for another because their
freight and margins differ, so the arithmetic that decides is parameterised per
tenant and the thresholds fall out of it.

All money is integer paise. All rates are integer basis points. Nothing here is
a float.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional

from .domain import ActionKind, Channel, Mode
from .money import Bps, Paise, Ppm, to_bps, to_paise


@dataclasses.dataclass(frozen=True)
class ActionEconomics:
    """What one action costs, and how often it damages an order that was fine.

    `harm_bps` is the probability that taking this action on an order which
    would have delivered anyway causes it *not* to. It is what stops the classic
    failure of this product category (PRD section 5): an agent rewarded for
    preventing returns will cancel good orders, because a cancelled order never
    appears as an RTO.

    It ships as a documented prior rather than a measured quantity. The derived
    repair threshold is sensitive to it -- at 100 bps the ~2.9% figure in PRD
    section 5 follows; at 0 it would be ~1.2% -- so it is reported alongside
    every threshold the console renders, and it is the next quantity the bandit
    should learn once enough outcomes have been ingested.
    """

    cost_paise: Paise = 0          # incurred whenever the action is taken
    harm_bps: Bps = 0              # damage rate on orders that would have delivered
    reversible: bool = True
    contacts_buyer: bool = False


#: Costs from PRD section 4. Repair ~Rs 3, verify ~Rs 2, part-payment and the
#: serviceability nudge ~Rs 1. Prepaid conversion's message is cheap; its real
#: cost is the gateway fee plus the incentive, and both are charged only on
#: conversion, so they are computed per order rather than fixed here.
DEFAULT_ACTION_ECONOMICS: Dict[ActionKind, ActionEconomics] = {
    ActionKind.NO_ACTION: ActionEconomics(0, 0, True, False),
    ActionKind.ADDRESS_REPAIR: ActionEconomics(300, 100, True, True),
    ActionKind.CONTACT_VERIFY: ActionEconomics(200, 100, True, True),
    ActionKind.PREPAID_CONVERSION: ActionEconomics(200, 0, True, True),
    ActionKind.PART_PAYMENT: ActionEconomics(100, 200, True, True),
    ActionKind.CANCEL: ActionEconomics(100, 0, False, True),
    ActionKind.ESCALATE: ActionEconomics(0, 0, True, False),
}


class ConfigError(ValueError):
    """A cost model that would produce nonsense money. Raised at onboarding."""


@dataclasses.dataclass(frozen=True)
class MerchantConfig:
    """One tenant's arithmetic and policy.

    Defaults reproduce the worked example in PRD section 3: a Rs 1,200 order at
    30% margin is +Rs 265 delivered, -Rs 195 returned, Rs 0 cancelled before
    packing, +Rs 221 converted to prepaid and delivered.

    Every default is illustrative and configurable. The structure is the
    contribution; the constants are merchant inputs.
    """

    merchant_id: str
    monthly_incentive_budget_paise: Paise      # required: no sane default exists

    # ---- cost model (PRD section 3) --------------------------------------
    gross_margin_bps: Bps = 3000               # 0.30
    packaging_paise: Paise = 2500              # Rs 25
    forward_freight_paise: Paise = 7000        # Rs 70
    failed_attempt_fee_paise: Paise = 2000     # Rs 20, from the carrier contract
    return_freight_paise: Paise = 7000         # Rs 70, from the carrier contract
    goodwill_cost_paise: Paise = 1000          # Rs 10, optional
    gateway_fee_bps: Bps = 200                 # 0.02, from the payment provider

    # ---- merchant policy -------------------------------------------------
    max_incentive_paise: Paise = 12000         # Rs 120 ceiling on any one offer
    auto_cancel_enabled: bool = False          # off by default, deliberately
    value_review_threshold_paise: Paise = 500000   # above this a cancel needs a human

    # ---- adoption and latency (PRD section 1) ----------------------------
    mode: Mode = Mode.SHADOW
    max_hold_minutes: int = 240                # hard auto-release; fulfilment is sacred
    reply_window_hours: int = 4
    #: What happens when a gateway payment offer is not paid before its hold
    #: deadline. ``ship_cod`` preserves fulfilment; ``cancel`` is for merchants
    #: who do not want an unpaid offer to enter the carrier network.
    payment_fallback: str = "ship_cod"
    decision_latency_budget_ms: int = 5000     # p95 target with no buyer contact

    # ---- guardrails (PRD section 5) --------------------------------------
    quiet_hours_start: int = 21                # local hour, inclusive
    quiet_hours_end: int = 8                   # local hour, exclusive
    tz_offset_minutes: int = 330               # Asia/Kolkata, +05:30, no DST
    max_paid_actions_per_order: int = 1        # prevents message stacking
    max_messages_per_buyer_24h: int = 1        # frequency cap

    # ---- evaluation -------------------------------------------------------
    #: Deterministic control-group share for measuring the counterfactual.
    holdout_rate_percent: int = 10

    #: Where this merchant's buyers are contacted, and therefore which consent
    #: flag is checked before any message goes out.
    #:
    #: Per-merchant rather than global, because consent is per channel: a buyer who
    #: agreed to WhatsApp has not agreed to SMS, and a deployment that checked one
    #: flag for all channels would be contacting people on a channel they never
    #: opted into. `SIMULATED` is the honest default -- a fresh merchant has no
    #: channel wired, and defaulting to a real one would imply a delivery path that
    #: does not exist.
    default_channel: Channel = Channel.SIMULATED

    # ---- geography -------------------------------------------------------
    #: Where this merchant ships from. Backs the origin-to-destination distance
    #: feature; empty disables it rather than defaulting to somebody else's
    #: warehouse, because a wrong origin produces a confidently wrong feature.
    origin_pincode: str = ""

    # ---- model operations (PRD section 7) --------------------------------
    min_labels_to_train: int = 1000
    max_ece_to_promote_ppm: Ppm = 50_000       # 5%: a worse-calibrated model cannot ship

    # ---- action economics ------------------------------------------------
    action_economics: Dict[ActionKind, ActionEconomics] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_ACTION_ECONOMICS)
    )

    # ------------------------------------------------------------------
    def economics(self, action: ActionKind) -> ActionEconomics:
        """What this merchant pays for `action`, falling back to the platform default.

        The fallback is `DEFAULT_ACTION_ECONOMICS`, never a bare
        `ActionEconomics()`. A bare one is free and harmless -- cost 0, harm 0 --
        and a free action clears every threshold the gate derives, because the
        gate's question is "does the expected gain exceed the cost". So a missing
        entry would not merely be unpriced; it would be the one action the agent
        always takes. Only an action this build has never heard of gets the zero,
        and `ActionKind` is a closed enum.
        """
        if action in self.action_economics:
            return self.action_economics[action]
        return DEFAULT_ACTION_ECONOMICS.get(action, ActionEconomics())

    def validate(self) -> "MerchantConfig":
        """Reject a cost model that would make the gate produce nonsense.

        Called at onboarding rather than at decision time: a merchant should
        learn their freight is negative while typing it in, not when an order
        silently fails to be scored.
        """
        if not self.merchant_id:
            raise ConfigError("merchant_id is required")
        if not 0 <= self.gross_margin_bps <= 10_000:
            raise ConfigError("gross_margin_bps must be within 0..10000, got %d"
                              % self.gross_margin_bps)
        if not 0 <= self.gateway_fee_bps <= 10_000:
            raise ConfigError("gateway_fee_bps must be within 0..10000, got %d"
                              % self.gateway_fee_bps)
        for name in ("packaging_paise", "forward_freight_paise",
                     "failed_attempt_fee_paise", "return_freight_paise",
                     "goodwill_cost_paise", "max_incentive_paise",
                     "monthly_incentive_budget_paise",
                     "value_review_threshold_paise"):
            value = getattr(self, name)
            if value < 0:
                raise ConfigError("%s must not be negative, got %d" % (name, value))
        if self.max_hold_minutes <= 0:
            raise ConfigError("max_hold_minutes must be positive; fulfilment must "
                              "always be released eventually")
        if self.payment_fallback not in ("ship_cod", "cancel"):
            raise ConfigError(
                "payment_fallback must be ship_cod or cancel, got %r"
                % self.payment_fallback)
        if not 0 <= self.quiet_hours_start <= 23 or not 0 <= self.quiet_hours_end <= 23:
            raise ConfigError("quiet hours must be hours of a day")
        if self.max_paid_actions_per_order < 1:
            raise ConfigError("max_paid_actions_per_order must be at least 1")
        if not 0 <= self.holdout_rate_percent <= 100:
            raise ConfigError(
                "holdout_rate_percent must be within 0..100, got %d"
                % self.holdout_rate_percent)
        return self

    def replace(self, **changes: object) -> "MerchantConfig":
        """A modified copy. Config is frozen so a decision cannot be re-explained
        against a config that has since been edited in place."""
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    @classmethod
    def from_decimal_inputs(
        cls,
        merchant_id: str,
        monthly_incentive_budget: str,
        gross_margin_rate: str = "0.30",
        packaging: str = "25",
        forward_freight: str = "70",
        failed_attempt_fee: str = "20",
        return_freight: str = "70",
        goodwill_cost: str = "10",
        gateway_fee_rate: str = "0.02",
        max_incentive: str = "120",
        value_review_threshold: str = "5000",
        **kwargs: object
    ) -> "MerchantConfig":
        """Build from the decimal strings an onboarding form collects.

        The console and the API accept rupees as typed; this is the one place
        they become integers. Floats are refused by `money.to_paise`, so a
        mis-wired form fails loudly instead of rounding somebody's freight.
        """
        return cls(
            merchant_id=merchant_id,
            monthly_incentive_budget_paise=to_paise(monthly_incentive_budget),
            gross_margin_bps=to_bps(gross_margin_rate),
            packaging_paise=to_paise(packaging),
            forward_freight_paise=to_paise(forward_freight),
            failed_attempt_fee_paise=to_paise(failed_attempt_fee),
            return_freight_paise=to_paise(return_freight),
            goodwill_cost_paise=to_paise(goodwill_cost),
            gateway_fee_bps=to_bps(gateway_fee_rate),
            max_incentive_paise=to_paise(max_incentive),
            value_review_threshold_paise=to_paise(value_review_threshold),
            **kwargs                                   # type: ignore[arg-type]
        ).validate()


#: The demo tenant of the PRD's five-minute script. Named here rather than in a
#: fixture so the console, the storefront and the backtest all seed the same one.
#:
#: These are *decimal onboarding inputs* -- rupees as a merchant would type them,
#: not paise. Feed them to `from_decimal_inputs`, or just call `kolam_config()`.
KOLAM_CERAMICS = dict(
    merchant_id="kolam_ceramics",
    monthly_incentive_budget="25000",
    gross_margin_rate="0.30",
    packaging="25",
    forward_freight="70",
)

#: A studio pottery seller shipping from Chennai. 600001 is a real pincode in the
#: reference table (Mannady, Chennai, metro), so the distance feature has a true
#: origin to measure from rather than a placeholder.
KOLAM_ORIGIN_PINCODE = "600001"


def kolam_config(**overrides: object) -> MerchantConfig:
    """The demo tenant's config, built and validated.

    A function rather than a module-level constant so a caller cannot mutate the
    shared instance, and so `KOLAM_CERAMICS`'s decimal strings are never mistaken
    for paise -- which is exactly the mistake a bare dict of rupee strings
    invites.
    """
    return MerchantConfig.from_decimal_inputs(
        origin_pincode=KOLAM_ORIGIN_PINCODE, **{**KOLAM_CERAMICS, **overrides})
