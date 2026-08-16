"""Domain vocabulary: the nouns the whole system agrees on.

This module imports only `precourier.money`. Nothing here knows about storage,
HTTP, models, or the synthetic world -- which is what lets the state machine,
the gate and the backtest all speak the same language without importing each
other.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Dict, FrozenSet, List, Optional, Tuple

from .money import Paise, Ppm


# ==========================================================================
# enums
# ==========================================================================

class Mode(str, enum.Enum):
    """The adoption ladder of PRD section 1. Every rung is a real deployment."""

    SHADOW = "shadow"        # decides and logs; contacts nobody, changes nothing
    ADVISORY = "advisory"    # contacts buyers and writes back; never blocks fulfilment
    GATEWAY = "gateway"      # holds the label until it releases, cancels, or times out


class OrderState(str, enum.Enum):
    """PRD section 2. Every transition writes exactly one audit event."""

    RECEIVED = "received"
    SCORING = "scoring"
    DIAGNOSING = "diagnosing"
    DECIDING = "deciding"
    NO_ACTION = "no_action"
    ACTING = "acting"
    AWAITING_BUYER = "awaiting_buyer"
    REPAIRED = "repaired"
    CONVERTED = "converted"
    VERIFIED = "verified"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"
    RELEASED_FOR_DISPATCH = "released_for_dispatch"   # the contract event
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    RTO = "rto"


class Cause(str, enum.Enum):
    """The binding constraint on delivery. PRD section 4.

    Risk is not a cause. One undifferentiated intervention wastes money on
    orders it cannot help, so each risky order is classified into exactly one
    binding cause with its supporting evidence.
    """

    ADDRESS_DEFECT = "address_defect"
    UNREACHABLE_CONTACT = "unreachable_contact"
    WEAK_INTENT = "weak_intent"
    REPEAT_RETURNER = "repeat_returner"
    NOT_SERVICEABLE = "not_serviceable"
    UNCERTAIN = "uncertain"          # risk elevated, no cause dominates -> abstain


class ActionKind(str, enum.Enum):
    """What the agent may do. `NO_ACTION` is a first-class output, not a failure."""

    NO_ACTION = "no_action"
    ADDRESS_REPAIR = "address_repair"
    CONTACT_VERIFY = "contact_verify"
    PREPAID_CONVERSION = "prepaid_conversion"
    PART_PAYMENT = "part_payment"
    CANCEL = "cancel"
    ESCALATE = "escalate"


#: Actions that spend money and are therefore subject to the one-paid-action
#: cap, the monthly budget, quiet hours and the frequency cap.
PAID_ACTIONS = frozenset({
    ActionKind.ADDRESS_REPAIR,
    ActionKind.CONTACT_VERIFY,
    ActionKind.PREPAID_CONVERSION,
    ActionKind.PART_PAYMENT,
})

#: Actions that contact the buyer, and so need consent and quiet-hours checks.
CONTACT_ACTIONS = frozenset({
    ActionKind.ADDRESS_REPAIR,
    ActionKind.CONTACT_VERIFY,
    ActionKind.PREPAID_CONVERSION,
    ActionKind.PART_PAYMENT,
})

#: Actions that cannot be undone once taken.
IRREVERSIBLE_ACTIONS = frozenset({ActionKind.CANCEL})


# --------------------------------------------------------------------------
# the order lifecycle
#
# Declared as data rather than as branches in a service, for two reasons. It is
# checkable -- `tests/test_orders.py` asserts every state is reachable from
# RECEIVED, and that every state in OPEN_STATES can reach RELEASED_FOR_DISPATCH.
# That second invariant is the fulfilment-is-sacred guarantee expressed as a
# graph property: no configuration of this table can strand an order in the hold
# queue. It applies to the open states only -- SHIPPED is already past release.
# And it is renderable: the console draws the merchant's actual lifecycle from
# this map instead of from a diagram that drifts.
#
# RELEASED_FOR_DISPATCH is the contract event of PRD section 1 -- the moment
# PreCourier stops having an opinion and the label may be printed.
# --------------------------------------------------------------------------
TRANSITIONS: Dict[OrderState, FrozenSet[OrderState]] = {
    OrderState.RECEIVED: frozenset({
        OrderState.SCORING,
        # Shadow and advisory modes may release immediately; gateway holds.
        OrderState.RELEASED_FOR_DISPATCH,
        OrderState.CANCELLED,
    }),
    OrderState.SCORING: frozenset({
        OrderState.DIAGNOSING, OrderState.NO_ACTION,
        OrderState.RELEASED_FOR_DISPATCH,
    }),
    OrderState.DIAGNOSING: frozenset({
        OrderState.DECIDING, OrderState.NO_ACTION,
        OrderState.RELEASED_FOR_DISPATCH,
    }),
    OrderState.DECIDING: frozenset({
        OrderState.ACTING, OrderState.NO_ACTION, OrderState.ESCALATED,
        OrderState.CANCELLED, OrderState.RELEASED_FOR_DISPATCH,
    }),
    OrderState.ACTING: frozenset({
        OrderState.AWAITING_BUYER, OrderState.REPAIRED, OrderState.CONVERTED,
        OrderState.VERIFIED, OrderState.ESCALATED, OrderState.CANCELLED,
        OrderState.RELEASED_FOR_DISPATCH,
    }),
    OrderState.AWAITING_BUYER: frozenset({
        OrderState.REPAIRED, OrderState.CONVERTED, OrderState.VERIFIED,
        OrderState.ESCALATED, OrderState.CANCELLED,
        # The buyer never replied and the hold expired: fulfilment wins.
        OrderState.RELEASED_FOR_DISPATCH,
    }),
    OrderState.REPAIRED: frozenset({OrderState.RELEASED_FOR_DISPATCH}),
    OrderState.CONVERTED: frozenset({OrderState.RELEASED_FOR_DISPATCH}),
    OrderState.VERIFIED: frozenset({OrderState.RELEASED_FOR_DISPATCH}),
    OrderState.NO_ACTION: frozenset({OrderState.RELEASED_FOR_DISPATCH}),
    OrderState.ESCALATED: frozenset({
        # A human decides. Both doors stay open, and the hold timeout still
        # forces a release if nobody looks.
        OrderState.CANCELLED, OrderState.RELEASED_FOR_DISPATCH,
    }),
    OrderState.RELEASED_FOR_DISPATCH: frozenset({
        OrderState.SHIPPED,
        # Some merchants report a terminal outcome without a shipped event.
        OrderState.DELIVERED, OrderState.RTO,
    }),
    OrderState.SHIPPED: frozenset({OrderState.DELIVERED, OrderState.RTO}),

    # Terminal.
    OrderState.CANCELLED: frozenset(),
    OrderState.DELIVERED: frozenset(),
    OrderState.RTO: frozenset(),
}

#: States after which nothing further happens to the order.
TERMINAL_STATES = frozenset(
    state for state, nxt in TRANSITIONS.items() if not nxt)

#: States in which the order is still inside PreCourier's decision window. In
#: gateway mode these are exactly the states where a label must not be printed.
OPEN_STATES = frozenset({
    OrderState.RECEIVED, OrderState.SCORING, OrderState.DIAGNOSING,
    OrderState.DECIDING, OrderState.ACTING, OrderState.AWAITING_BUYER,
    OrderState.ESCALATED,
})

#: States in which the order is waiting on something outside our control, so the
#: auto-release timer of PRD section 1 applies.
HOLDING_STATES = frozenset({
    OrderState.AWAITING_BUYER, OrderState.ESCALATED,
})


def can_transition(current: OrderState, target: OrderState) -> bool:
    return target in TRANSITIONS.get(current, frozenset())


def path_to(current: OrderState,
            target: OrderState) -> Optional[List[OrderState]]:
    """The shortest legal sequence of states from `current` to `target`, exclusive.

    Exists because real integrations skip events. A merchant reports `delivered`
    for an order we last saw at `received`, and there are only three things we can
    do with that: refuse it and lose the training label, jump the order straight to
    `delivered` and write an audit trail that claims a release that never
    happened, or walk the legal edges in between and say that we inferred them.

    The third is the only honest one, so this returns the walk. `None` means no
    legal route exists at all -- which for a terminal order is the correct answer
    and the caller should refuse.

    Breadth-first, so the route taken is the one with the fewest invented steps.
    """
    if current is target:
        return []
    frontier: List[Tuple[OrderState, List[OrderState]]] = [(current, [])]
    seen = {current}
    while frontier:
        state, route = frontier.pop(0)
        for nxt in sorted(TRANSITIONS.get(state, frozenset()),
                          key=lambda s: s.value):
            if nxt in seen:
                continue
            if nxt is target:
                return route + [nxt]
            seen.add(nxt)
            frontier.append((nxt, route + [nxt]))
    return None


class IllegalTransition(ValueError):
    """A state change the lifecycle does not permit.

    Raised, never coerced. A system that quietly repairs an illegal transition
    produces an audit trail that disagrees with what happened, and the audit trail
    is the product.
    """

    def __init__(self, current: OrderState, target: OrderState) -> None:
        allowed = sorted(s.value for s in TRANSITIONS.get(current, frozenset()))
        super().__init__(
            "cannot move an order from %s to %s; legal next states are %s"
            % (current.value, target.value,
               ", ".join(allowed) if allowed else "(none: terminal)"))
        self.current = current
        self.target = target


class OutcomeKind(str, enum.Enum):
    """What the carrier eventually reported. These become training labels."""

    DELIVERED = "delivered"
    RTO = "rto"
    CANCELLED = "cancelled"
    PREPAID_DELIVERED = "prepaid_delivered"


#: The order state each terminal outcome implies. Used when a merchant reports an
#: outcome for an order whose intermediate states we never saw -- a real
#: integration will skip events, and refusing the outcome would lose the label.
OUTCOME_STATE = {
    OutcomeKind.DELIVERED: OrderState.DELIVERED,
    OutcomeKind.PREPAID_DELIVERED: OrderState.DELIVERED,
    OutcomeKind.RTO: OrderState.RTO,
    OutcomeKind.CANCELLED: OrderState.CANCELLED,
}


class Channel(str, enum.Enum):
    SIMULATED = "simulated"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    SMS = "sms"


class LedgerReason(str, enum.Enum):
    """Why a ledger entry exists. Sums by reason produce the ROI report."""

    GROSS_MARGIN = "gross_margin"
    PACKAGING = "packaging"
    FORWARD_FREIGHT = "forward_freight"
    FAILED_ATTEMPT_FEE = "failed_attempt_fee"
    RETURN_FREIGHT = "return_freight"
    GOODWILL = "goodwill"
    GATEWAY_FEE = "gateway_fee"
    INCENTIVE = "incentive"
    ACTION_COST = "action_cost"
    CORRECTION = "correction"

    #: A zero-amount marker recording that an outcome was posted at all.
    #:
    #: Exists because `cancelled` costs nothing: its breakdown has no lines, so
    #: without a marker row a cancellation could be posted twice and the ledger
    #: would have no way to tell. It also makes cancelled orders visible in a
    #: per-order ledger view, contributing exactly Rs 0 -- which is the honest
    #: representation of a prevented RTO, and better than an absent row that
    #: looks like an order nobody processed.
    OUTCOME_POSTED = "outcome_posted"


# ==========================================================================
# value objects
# ==========================================================================

def _address_tokens(text: str) -> List[str]:
    """Split address text on commas and whitespace.

    A free function so `Address.as_text` and `Address.tokens` cannot disagree about
    what a token is -- the first uses it to de-duplicate `raw` against the
    structured fields, the second to build features, and a divergence between the
    two would mean the model scored text no human ever saw.
    """
    return [t for t in text.replace(",", " ").split() if t]


@dataclasses.dataclass(frozen=True)
class Address:
    """A delivery address as the store supplied it, plus any verified patch.

    Deliverability is a *structural* property of this text, not a semantic one
    (PRD section 7) -- which is why there is no embedding anywhere near it.
    """

    raw: str = ""
    line1: str = ""
    line2: str = ""
    landmark: str = ""
    locality: str = ""
    city: str = ""
    state: str = ""
    pincode: str = ""

    def as_text(self) -> str:
        """Single-line rendering, used for token features and carrier validation."""
        parts = [self.line1, self.line2, self.landmark, self.locality,
                 self.city, self.state, self.pincode]
        structured = ", ".join(p.strip() for p in parts if p and p.strip())
        raw = self.raw.strip()
        if not structured or not raw:
            return structured or raw

        # `raw` is either the whole address (one free-text box) or just the street
        # portion (free text plus a pincode dropdown -- the commonest Indian
        # checkout form). Appending whatever `raw` contains that the structured
        # fields do not is right for both shapes: it recovers the house number in
        # the second case without repeating the city in the first.
        #
        # Returning `structured` alone, which this used to do whenever any
        # structured field was set, silently deleted the house number and left
        # "Bengaluru, 560001" -- which the defect detector would then diagnose as a
        # missing house number and try to repair. Harming an address that would
        # have delivered is the expensive error, so the union is the safe direction.
        known = {token.lower() for token in _address_tokens(structured)}
        extra = [t for t in _address_tokens(raw) if t.lower() not in known]
        if not extra:
            return structured
        return "%s, %s" % (" ".join(extra), structured)

    def tokens(self) -> List[str]:
        return _address_tokens(self.as_text())


@dataclasses.dataclass(frozen=True)
class Buyer:
    """Per-buyer state and consent. Consent is checked before all other logic."""

    id: str
    merchant_id: str
    phone_hash: str                       # sha256, used for analytics and joins
    phone: str = ""                       # plaintext, only where operationally required
    email: str = ""
    consent: Dict[str, bool] = dataclasses.field(default_factory=dict)
    never_contact: bool = False
    prior_orders: int = 0
    prior_rtos: int = 0
    prior_prepaid: int = 0
    prior_messages: int = 0
    prior_replies: int = 0
    last_order_at: Optional[str] = None
    last_contacted_at: Optional[str] = None

    def consents_to(self, channel: Channel) -> bool:
        """No flag, no message -- and the refusal is logged, never silent."""
        if self.never_contact:
            return False
        return bool(self.consent.get(channel.value, False))

    @property
    def refusal_ratio(self) -> float:
        if self.prior_orders <= 0:
            return 0.0
        return self.prior_rtos / self.prior_orders


@dataclasses.dataclass(frozen=True)
class Order:
    """A COD order awaiting a release decision. Value is always integer paise."""

    id: str
    merchant_id: str
    buyer_id: str
    value_paise: Paise
    address: Address
    external_id: str = ""
    state: OrderState = OrderState.RECEIVED
    mode: Mode = Mode.SHADOW
    payment_method: str = "cod"
    item_count: int = 1
    category: str = ""
    discount_bps: int = 0
    placed_at: str = ""
    held_until: Optional[str] = None       # gateway mode: the auto-release deadline
    is_first_time_buyer: bool = True

    @property
    def is_cod(self) -> bool:
        return self.payment_method == "cod"


@dataclasses.dataclass(frozen=True)
class Evidence:
    """One observed fact supporting a cause. The console shows these, not a label."""

    code: str
    detail: str = ""
    weight: int = 1

    def __str__(self) -> str:
        return "%s: %s" % (self.code, self.detail) if self.detail else self.code


@dataclasses.dataclass(frozen=True)
class RiskScore:
    """A calibrated P(RTO), quantised to ppm, tagged with the model that made it.

    Every stored score references its model version so a decision can be
    re-explained months later against the model that actually produced it.
    """

    p_rto_ppm: Ppm
    model_version: str
    features_id: str = ""
    raw_ppm: Optional[Ppm] = None          # pre-calibration, for the drift report


@dataclasses.dataclass(frozen=True)
class Diagnosis:
    """The binding cause plus the evidence that selected it."""

    cause: Cause
    evidence: Tuple[Evidence, ...] = ()
    runner_up: Optional[Cause] = None
    detail: str = ""

    @property
    def abstained(self) -> bool:
        return self.cause is Cause.UNCERTAIN


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What the carrier reported. Ingested, never generated by this system."""

    order_id: str
    merchant_id: str
    kind: OutcomeKind
    reported_at: str = ""
    carrier_reason: str = ""
    attempts: int = 0
