"""Request and response shapes for the v1 API.

Two conventions here are load-bearing.

**Money arrives as a decimal string or as integer paise, never as a JSON number.**
`"value": "1200.00"` and `"value_paise": 120000` are both accepted; `"value":
1200.00` is refused with a 422. That looks pedantic until you notice that a JSON
number becomes a float, a float becomes 1199.9999999999998, and the difference
lands in a merchant's margin report. `money.to_paise` refuses floats at the bottom
of the stack; this refuses them at the top, so the error names the field.

**Every response says what to do with the parcel.** `release` is the field the
merchant's integration branches on, and it is present on success, on a handled
error, and on an internal failure. A response this system cannot phrase as an
instruction about a parcel is not a useful response.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..money import Paise, to_paise


class _Strict(BaseModel):
    """Refuses unknown fields.

    A misspelled `pincode` silently ignored is a defective address delivered on
    purpose. Better to fail the request and name the field.
    """

    model_config = ConfigDict(extra="forbid")


# ==========================================================================
# requests
# ==========================================================================

class AddressIn(_Strict):
    raw: str = ""
    line1: str = ""
    line2: str = ""
    landmark: str = ""
    locality: str = ""
    city: str = ""
    state: str = ""
    pincode: str = ""

    @model_validator(mode="after")
    def _something_to_deliver_to(self) -> "AddressIn":
        if not any((self.raw, self.line1, self.line2, self.landmark,
                    self.locality, self.city, self.pincode)):
            raise ValueError(
                "an address needs at least one of raw, line1, line2, landmark, "
                "locality, city or pincode; an empty address is not a defect this "
                "system can diagnose, it is a missing field")
        return self


class OrderIn(_Strict):
    """A COD order awaiting a release decision."""

    address: AddressIn
    phone: str = ""
    email: str = ""
    value: Optional[str] = Field(
        default=None,
        description="Order value in rupees as a decimal string, e.g. \"1200.00\".")
    value_paise: Optional[int] = Field(
        default=None, ge=0,
        description="Order value in integer paise. Use this or `value`, not both.")
    external_id: str = ""
    placed_at: Optional[str] = None
    payment_method: str = "cod"
    item_count: int = Field(default=1, ge=1)
    category: str = ""
    discount_bps: int = Field(default=0, ge=0, le=10_000)
    mode: Optional[str] = Field(
        default=None,
        description="Override the merchant's mode for this order: shadow, "
                    "advisory or gateway. Used to canary a slice of traffic.")
    consent: Dict[str, bool] = Field(
        default_factory=dict,
        description="Channel name to consent flag, as captured at checkout. "
                    "Without a flag for the merchant's channel, no message is "
                    "sent and the refusal is recorded.")

    @model_validator(mode="after")
    def _exactly_one_money_field(self) -> "OrderIn":
        if (self.value is None) == (self.value_paise is None):
            raise ValueError(
                "supply exactly one of `value` (rupees as a decimal string) or "
                "`value_paise` (integer paise)")
        if self.value is not None and to_paise(self.value) < 0:
            raise ValueError("order value must be non-negative")
        return self

    def paise(self) -> Paise:
        if self.value_paise is not None:
            return int(self.value_paise)
        return to_paise(self.value or "0")


class OutcomeIn(_Strict):
    """What the carrier reported. One per order, ever."""

    kind: str = Field(
        description="delivered, rto, cancelled or prepaid_delivered.")
    order_id: Optional[str] = None
    external_id: Optional[str] = Field(
        default=None,
        description="The merchant's own order reference, if the order id is not "
                    "to hand. One of `order_id` or `external_id` is required.")
    carrier_reason: str = ""
    attempts: int = Field(default=0, ge=0)
    reported_at: Optional[str] = None

    @model_validator(mode="after")
    def _identifies_an_order(self) -> "OutcomeIn":
        has_order_id = bool(self.order_id and self.order_id.strip())
        has_external_id = bool(self.external_id and self.external_id.strip())
        if has_order_id == has_external_id:
            raise ValueError(
                "supply exactly one of `order_id` or `external_id`")
        return self


class ResolutionIn(_Strict):
    """How a held order came back."""

    resolution: str = Field(
        description="address_corrected, address_confirmed, contact_confirmed, "
                    "converted_to_prepaid, cancelled_by_buyer, no_reply, "
                    "merchant_released or merchant_cancelled.")
    address: Optional[AddressIn] = None
    detail: str = ""


class InboundMessageIn(_Strict):
    """A buyer reply on the simulated channel.

    The optional address is the structured result of deterministic extraction. A
    future LLM may propose it, but the service still verifies it before patching.
    """

    body: str = Field(min_length=1, max_length=4000)
    channel: str
    address: Optional[AddressIn] = None
    detail: str = ""


class HistoricalBacktestIn(_Strict):
    """A provider-neutral historical CSV projection request."""

    csv_text: str = Field(min_length=1, max_length=20_000_000)
    effectiveness_ppm: int = Field(default=550_000, ge=0, le=1_000_000)


class MessageOut(BaseModel):
    id: str
    order_id: str
    direction: str
    channel: str
    body: str
    status: str
    created_at: str


class ReleaseIn(_Strict):
    reason: str = Field(min_length=1,
                        description="Recorded verbatim in the audit trail.")


class ConfigUpdateIn(_Strict):
    gross_margin_rate: Optional[str] = None
    packaging: Optional[str] = None
    forward_freight: Optional[str] = None
    failed_attempt_fee: Optional[str] = None
    return_freight: Optional[str] = None
    goodwill_cost: Optional[str] = None
    gateway_fee_rate: Optional[str] = None
    max_incentive: Optional[str] = None
    monthly_incentive_budget: Optional[str] = None
    auto_cancel_enabled: Optional[bool] = None
    value_review_threshold: Optional[str] = None
    mode: Optional[str] = None
    max_hold_minutes: Optional[int] = Field(default=None, gt=0)
    reply_window_hours: Optional[int] = Field(default=None, gt=0)
    payment_fallback: Optional[str] = None
    decision_latency_budget_ms: Optional[int] = Field(default=None, gt=0)
    quiet_hours_start: Optional[int] = Field(default=None, ge=0, le=23)
    quiet_hours_end: Optional[int] = Field(default=None, ge=0, le=23)
    tz_offset_minutes: Optional[int] = None
    max_paid_actions_per_order: Optional[int] = Field(default=None, ge=1)
    max_messages_per_buyer_24h: Optional[int] = Field(default=None, ge=0)
    holdout_rate_percent: Optional[int] = Field(default=None, ge=0, le=100)
    default_channel: Optional[str] = None
    origin_pincode: Optional[str] = None


class ApiKeyCreateIn(_Strict):
    mode: str = "test"
    label: str = ""


class ConfigOut(BaseModel):
    merchant_id: str
    revision: int
    mode: str
    default_channel: str
    gross_margin_bps: int
    packaging_paise: int
    forward_freight_paise: int
    failed_attempt_fee_paise: int
    return_freight_paise: int
    goodwill_cost_paise: int
    gateway_fee_bps: int
    max_incentive_paise: int
    monthly_incentive_budget_paise: int
    max_hold_minutes: int
    reply_window_hours: int
    payment_fallback: str
    decision_latency_budget_ms: int
    max_paid_actions_per_order: int
    max_messages_per_buyer_24h: int
    holdout_rate_percent: int


class ApiKeyOut(BaseModel):
    id: str
    key_prefix: str
    mode: str
    label: str
    created_at: str
    revoked_at: Optional[str] = None
    plaintext: Optional[str] = None


# ==========================================================================
# responses
# ==========================================================================

class DecisionOut(BaseModel):
    """The answer to "may I print this label", and the arithmetic behind it."""

    order_id: str
    state: str
    mode: str

    #: The field to branch on. True means print the label now.
    release: bool
    held_until: Optional[str] = None

    action: str
    recommended: str
    suppressed: bool = False
    blocked_reason: str = ""

    p_rto_ppm: int = 0
    threshold_ppm: int = 0
    cause: str = "uncertain"
    reason: str = ""
    decision_id: Optional[str] = None
    action_id: Optional[str] = None

    #: Stage 1's engine vocabulary is retained alongside the lifecycle vocabulary
    #: above. ``action`` says what the service did; this says what the rupee argmax
    #: selected, including a shadow or holdout counterfactual.
    chosen_action: str = "no_action"
    runner_up: str = ""
    expected_saving_paise: int = 0
    top_drivers: List[str] = Field(default_factory=list)
    holdout: bool = False
    explanation: Dict[str, Any] = Field(default_factory=dict)
    payment_link_id: Optional[str] = None
    payment_link_url: Optional[str] = None
    payment_link_amount_paise: Optional[int] = None

    #: True when the release happened because this system failed rather than
    #: because it decided. Surfaced, not hidden: an order that shipped without a
    #: decision must not be counted as one the agent chose to leave alone.
    degraded: bool = False


class OutcomeOut(BaseModel):
    order_id: str
    kind: str
    reported_at: str
    net_paise: int
    net_rupees: str
    lines: List[Dict[str, Any]]
    release: bool = True


class OrderOut(BaseModel):
    """An order and its history, for the console and for support."""

    order_id: str
    external_id: str = ""
    state: str
    mode: str
    value_paise: int
    payment_method: str
    placed_at: str
    held_until: Optional[str] = None
    payment_link_id: Optional[str] = None
    payment_link_url: Optional[str] = None
    payment_link_amount_paise: Optional[int] = None
    address: Dict[str, str]
    address_revisions: List[Dict[str, Any]] = Field(default_factory=list)
    transitions: List[Dict[str, Any]] = Field(default_factory=list)
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    features: List[Dict[str, Any]] = Field(default_factory=list)
    risk_scores: List[Dict[str, Any]] = Field(default_factory=list)
    diagnoses: List[Dict[str, Any]] = Field(default_factory=list)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    outcome: Optional[Dict[str, Any]] = None


class ErrorOut(BaseModel):
    """Every error body, including the fail-open one.

    `release` is present on errors too, and that is the whole point. A gateway-mode
    integration blocked on a 500 with no instruction has to guess, and the guess
    that loses money is "wait". So the error says what to do.
    """

    error: str
    detail: str = ""
    release: bool = True
    order_id: Optional[str] = None


class HealthOut(BaseModel):
    status: str
    schema_version: int
    mode: str
    synthetic_data: bool
    build: str
