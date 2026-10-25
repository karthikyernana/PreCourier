"""The order lifecycle as a service. Decide, act, hold -- and always release.

This module is where PRD section 1's hardest promise is kept: *fulfilment latency
is sacred*. Everything else here is subordinate to that. An order enters, and
whatever happens next -- the model is missing, the decider raises, the buyer never
replies, a guardrail refuses, the database throws -- the order leaves. There is no
path through this file that strands one.

Three mechanisms enforce it, and they are independent on purpose. Any one of them
failing leaves the other two.

1. **The graph.** `domain.TRANSITIONS` gives every open state an edge to
   `RELEASED_FOR_DISPATCH`, and `tests/test_orders.py` asserts that as a property
   of the graph rather than as a property of this code.
2. **The timer.** `sweep` releases anything past `held_until`, and `OrderRepo.hold`
   caps every deadline at `max_hold_minutes` on the way in, so no caller can
   extend a hold past the merchant's stated tolerance even by accident.
3. **Fail-open.** Every internal exception is caught and converted into a release.
   Not a retry, not a 500 -- a release. A system that blocks dispatch when it is
   broken is worse than no system, because the merchant cannot ship and cannot see
   why.

The seams are deliberate. `Decider` is what M4's gate plugs into and `Actor` is what
M7's Telegram channel plugs into; both have honest null implementations here, which
is why the state machine and the API are complete and testable before either of
those exists. The null implementations do not pretend: `NullDecider` says it has no
model, and `NoChannelActor` reports a block rather than a delivery.

What each mode means, in one line each, because this is the file where it is
decided:

* **shadow** -- reason fully, record the counterfactual, touch nothing, release.
* **advisory** -- act, then release without waiting. Never blocks fulfilment.
* **gateway** -- act, and hold the label until resolved, cancelled, or timed out.

Payment actions have a separate gateway branch. ``awaiting_payment`` means a
provider link was created and the label is held. A verified paid event walks the
order through ``converted`` and then releases it. An expiry/cancellation walks
through ``payment_expired`` and applies ``payment_fallback`` (ship COD or cancel).
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from . import guards
from .audit import AuditKind
from .domain import (ActionKind, Address, Buyer, CONTACT_ACTIONS, Cause,
                     Channel, Evidence, HOLDING_STATES, Mode, Order,
                     OrderState, PAID_ACTIONS, RiskScore, TERMINAL_STATES,
                     path_to)
from .ledger import loss_on_rto, margin_on_delivery, swing
from .money import Micro, Paise, Ppm
from .runtime import parse_ts
from .store.repo import (ActionRepo, BuyerRepo, DecisionRepo, LedgerRepo,
                           MessageRepo, NotFound, OrderRepo, OutboxRepo,
                           PaymentRepo)
from .tenancy import Tenant

#: Where the detail of a fail-open goes. The audit log records that one happened and
#: why; this carries the traceback, which belongs in an operator's log and nowhere a
#: merchant's integration can read it.
log = logging.getLogger("precourier.states")


#: States in which the merchant may print the label. `CANCELLED` is deliberately
#: absent and `SHIPPED` onwards are deliberately present -- an order past release
#: was already cleared, and a disposition read back for one must not claim we are
#: still holding it.
PRINTABLE_STATES = frozenset({
    OrderState.RELEASED_FOR_DISPATCH, OrderState.SHIPPED,
    OrderState.DELIVERED, OrderState.RTO,
})



# ==========================================================================
# what the decision layer says
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class Verdict:
    """The decision layer's answer, before mode policy or guardrails touch it.

    Carries the arithmetic, not just the conclusion. `threshold_ppm`,
    `expected_delta_micro` and `gate_inputs` exist so the console can render the
    working -- PRD section 10 asks for a decision a merchant can check by hand, and
    that is impossible if only the verb survives.

    `Cause.UNCERTAIN` with `NO_ACTION` is the honest default and the shape of every
    verdict this build produces until M4 supplies a real gate.
    """

    action: ActionKind = ActionKind.NO_ACTION
    p_rto_ppm: Ppm = 0
    cause: Cause = Cause.UNCERTAIN
    threshold_ppm: Ppm = 0
    expected_delta_micro: Micro = 0
    incentive_paise: Paise = 0
    reason: str = ""
    model_version: str = ""
    evidence: Tuple[Evidence, ...] = ()
    gate_inputs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    risk_score_id: Optional[str] = None
    diagnosis_id: Optional[str] = None
    hold_minutes: Optional[int] = None
    #: Pure reasoning artifacts. The service persists them after the decider returns;
    #: the decider itself never mutates the order or database.
    feature_payload: Optional[Dict[str, Any]] = None
    risk_score: Any = None
    diagnosis: Any = None
    #: Stage 1's immutable rupee explanation. The service persists it with the
    #: decision row and carries it through to every API disposition.
    explanation: Dict[str, Any] = dataclasses.field(default_factory=dict)
    holdout: bool = False


class Decider:
    """The seam M4's gate fills. Subclass or duck-type.

    A class rather than a `Protocol` so the null implementation and the real gate
    share a documented contract in one place: `decide` must not raise for ordinary
    reasons, and if it does raise the caller releases the order. It is also
    forbidden from writing to the order -- it observes and answers.
    """

    def decide(self, tenant: Tenant, order: Order, buyer: Buyer) -> Verdict:
        raise NotImplementedError


class NullDecider(Decider):
    """No model, no gate, no opinion. Says so.

    This is not a stub that fakes a score. A build with no trained model has
    genuinely no basis for an intervention, and the correct behaviour is to reason
    about nothing, record that fact, and release. Every order gets a decision row
    saying exactly this, so a merchant running the shadow deployment on day one
    sees "no model yet" rather than an empty console that looks broken.
    """

    reason = ("no risk model is deployed for this merchant, so no intervention "
              "can be justified")

    def decide(self, tenant: Tenant, order: Order, buyer: Buyer) -> Verdict:
        return Verdict(action=ActionKind.NO_ACTION, reason=self.reason,
                       model_version="none",
                       gate_inputs={"model": "none"})


class StaticDecider(Decider):
    """Returns a fixed verdict. For tests and for the guided demo walkthrough."""

    def __init__(self, verdict: Verdict) -> None:
        self._verdict = verdict

    def decide(self, tenant: Tenant, order: Order, buyer: Buyer) -> Verdict:
        return self._verdict


class RuleDecider(Decider):
    """The India-first cold-start decider.

    It deliberately uses only observable order, buyer, and postal-reference data.
    The rule model is a safe production fallback until a merchant has enough real
    outcomes to train and calibrate a learned model.
    """

    def __init__(self, *, reference: Any = None,
                 model_version: str = "rule-v1") -> None:
        self._reference = reference
        self._model_version = model_version

    def decide(self, tenant: Tenant, order: Order, buyer: Buyer) -> Verdict:
        from .geo import REFERENCE
        from .intelligence import (build_features, diagnose, evaluate_gate,
                                   rule_risk_score)
        from .guards import month_start

        features = build_features(
            order, buyer, config=tenant.config,
            reference=self._reference if self._reference is not None else REFERENCE)
        score = rule_risk_score(features, model_version=self._model_version)
        diagnosis = diagnose(features)
        gate = evaluate_gate(
            tenant.config, order, score, diagnosis,
            spent_this_month_paise=LedgerRepo(tenant).spend_since(
                month_start(tenant)))
        if gate.action in (ActionKind.PREPAID_CONVERSION,
                           ActionKind.PART_PAYMENT, ActionKind.CANCEL):
            gate = dataclasses.replace(
                gate, action=ActionKind.NO_ACTION,
                reason=("%s is not enabled in the India-first MVP; no action "
                        "was taken" % gate.action.value),
                hold_minutes=None,
                gate_inputs={**gate.gate_inputs,
                             "unsupported_mvp_action": gate.action.value})
        return dataclasses.replace(
            gate.to_verdict(),
            feature_payload=features.first_pass_mapping(), risk_score=score,
            diagnosis=diagnosis)


class ProfitDecisionDecider(Decider):
    """Live Stage 1 decider: intelligence features feed the rupee argmax engine."""

    def __init__(self, *, reference: Any = None, model_version: str = "rule-cause-v1") -> None:
        self._reference = reference
        self._model_version = model_version

    def decide(self, tenant: Tenant, order: Order, buyer: Buyer) -> Verdict:
        from .geo import REFERENCE
        from .intelligence import build_features, diagnose
        from .decision_engine import (RuleRiskModel, choose_action, in_holdout,
                                      specs_from_config)
        features = build_features(order, buyer, config=tenant.config,
                                  reference=self._reference or REFERENCE)
        probabilities = RuleRiskModel().predict(order, features)
        actions = specs_from_config(tenant.config)
        eligible_actions = {
            spec.name for spec in actions
            if spec.name != "cancel"
            or (tenant.config.auto_cancel_enabled
                and order.value_paise < tenant.config.value_review_threshold_paise)
        }
        decision = choose_action(order, probabilities,
                                 actions=actions,
                                 config=tenant.config,
                                 eligible_actions=eligible_actions)
        holdout = in_holdout(order.id, tenant.merchant_id,
                             tenant.config.holdout_rate_percent)
        cause_map = {"bad_address": Cause.ADDRESS_DEFECT,
                     "unreachable_buyer": Cause.UNREACHABLE_CONTACT,
                     "refusal": Cause.WEAK_INTENT}
        diagnosis = diagnose(features)
        action_map = {"confirm_intent": ActionKind.CONTACT_VERIFY,
                      "fix_address": ActionKind.ADDRESS_REPAIR,
                      "partial_cod": ActionKind.PART_PAYMENT,
                      "prepaid_nudge": ActionKind.PREPAID_CONVERSION,
                      "cancel": ActionKind.CANCEL,
                      "ship_as_is": ActionKind.NO_ACTION}
        chosen = action_map[decision.chosen_action]
        if chosen is ActionKind.CANCEL and not tenant.config.auto_cancel_enabled:
            chosen = ActionKind.ESCALATE
        drivers = decision.top_drivers
        return Verdict(action=chosen, p_rto_ppm=probabilities.rto_ppm,
                       cause=cause_map.get(drivers[0], diagnosis.cause),
                       expected_delta_micro=decision.expected_saving_paise * 1_000_000,
                       reason="%s; saving=%d paise; runner_up=%s" %
                              (decision.chosen_action, decision.expected_saving_paise,
                               decision.runner_up), model_version=self._model_version,
                       evidence=diagnosis.evidence,
                       gate_inputs={"cause_probabilities": probabilities.as_dict(),
                                     "chosen": decision.chosen_action,
                                     "runner_up": decision.runner_up,
                                     "expected_saving_paise": decision.expected_saving_paise,
                                     "margin_paise": margin_on_delivery(
                                         tenant.config, order.value_paise),
                                     "loss_paise": loss_on_rto(
                                         tenant.config, order.value_paise),
                                     "swing_paise": swing(
                                         tenant.config, order.value_paise),
                                     "top_drivers": list(drivers),
                                     "evaluations": [
                                         {"action": item.action,
                                          "expected_net_paise": item.expected_net_paise,
                                          "eligible": item.eligible}
                                         for item in decision.evaluations]},
                        feature_payload=features.first_pass_mapping(),
                        risk_score=RiskScore(
                            p_rto_ppm=probabilities.rto_ppm,
                            # ``rule-v1`` is the persisted score vocabulary used
                            # by the pre-Stage-1 console; the richer cause/action
                            # explanation is stored beside it and is versioned by
                            # the verdict below.
                            model_version="rule-v1",
                            raw_ppm=probabilities.rto_ppm,
                            cause_probabilities=probabilities.as_dict()),
                        diagnosis=diagnosis,
                        explanation={
                            "chosen_action": decision.chosen_action,
                            "runner_up": decision.runner_up,
                            "expected_saving_paise": decision.expected_saving_paise,
                            "top_drivers": list(drivers),
                            "holdout": holdout,
                            "cause_probabilities": probabilities.as_dict(),
                            "evaluations": [
                                {"action": item.action,
                                 "expected_net_paise": item.expected_net_paise,
                                 "eligible": item.eligible}
                                for item in decision.evaluations]},
                        holdout=holdout)


# ==========================================================================
# what actually reaches the buyer
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class Dispatch:
    """The result of trying to reach a buyer."""

    sent: bool
    channel: Channel
    detail: str = ""
    message_id: str = ""
    blocked_reason: str = ""
    body: str = ""
    template_id: str = ""
    provider_ref: str = ""


class Actor:
    """The seam M7's messaging channel fills.

    Performs the side effect only. It does not write the `actions` row, charge the
    ledger, or move the order -- `OrderService` does all three, so a new channel is
    an adapter rather than a second copy of the accounting.
    """

    #: Whether this actor can reach anyone at all. Asked *before* the guard pass so
    #: an unconfigured deployment blocks on `no_channel` at the same place every
    #: other refusal is recorded, instead of discovering it halfway through a send.
    available = True

    def perform(self, tenant: Tenant, order: Order, buyer: Buyer,
                verdict: Verdict) -> Dispatch:
        raise NotImplementedError


class NoChannelActor(Actor):
    """Sends nothing, and reports that it sent nothing.

    The default wiring for a deployment with no messaging channel configured. It
    returns a *block*, which means the order releases and the `actions` row records
    `no_channel` -- rather than reporting a success and leaving an order waiting
    for a reply to a message that was never sent. That second behaviour is the one
    worth naming, because it is the plausible shortcut here and it would put orders
    into a four-hour hold on the strength of a lie.
    """

    available = False

    def perform(self, tenant: Tenant, order: Order, buyer: Buyer,
                verdict: Verdict) -> Dispatch:
        return Dispatch(sent=False, channel=Channel.SIMULATED,
                        blocked_reason=guards.NO_CHANNEL,
                        detail="no messaging channel is configured")


class SimulatedActor(Actor):
    """A durable in-app channel for the India-first MVP and local demos."""

    available = True

    def perform(self, tenant: Tenant, order: Order, buyer: Buyer,
                verdict: Verdict) -> Dispatch:
        message = _simulated_message(order, verdict)
        return Dispatch(
            # This adapter never calls a provider. Never label its record as
            # WhatsApp/SMS merely because that is the merchant's future channel.
            sent=True, channel=Channel.SIMULATED,
            detail="simulated message queued", message_id="",
            body=message, template_id="address_repair_v1")


def _simulated_message(order: Order, verdict: Verdict) -> str:
    """Keep the demo channel useful without claiming a live provider send."""
    if verdict.action is ActionKind.ADDRESS_REPAIR:
        return ("PreCourier simulated message for order %s: please reply with "
                "your flat/house number, locality and landmark so we can verify "
                "your delivery address. COD remains available." % order.id)
    if verdict.action is ActionKind.CONTACT_VERIFY:
        return ("PreCourier simulated message for order %s: please confirm the "
                "mobile number the courier should use for delivery." % order.id)
    return "PreCourier simulated message for order %s: please confirm delivery." % order.id


# ==========================================================================
# how a hold ends
# ==========================================================================

class Resolution(str, enum.Enum):
    """How a held order came back. The inbound half of the buyer conversation."""

    ADDRESS_CORRECTED = "address_corrected"      # buyer supplied a new address
    ADDRESS_CONFIRMED = "address_confirmed"      # buyer said the old one is right
    CONTACT_CONFIRMED = "contact_confirmed"
    CONVERTED_TO_PREPAID = "converted_to_prepaid"
    PART_PAID = "part_paid"                      # token amount taken, rest on delivery
    OFFER_DECLINED = "offer_declined"            # replied, and said no
    CANCELLED_BY_BUYER = "cancelled_by_buyer"
    NO_REPLY = "no_reply"
    MERCHANT_RELEASED = "merchant_released"
    MERCHANT_CANCELLED = "merchant_cancelled"


#: Where each resolution leaves the order. `REPAIRED`, `CONVERTED` and `VERIFIED`
#: are distinct states rather than one `resolved`, because the backtest needs to
#: know which repair worked -- an address fix and a prepaid conversion have
#: different costs and different effects, and collapsing them would make the
#: per-cause effectiveness table unreportable.
#:
#: `OFFER_DECLINED` and `NO_REPLY` share a state and must not share a resolution.
#: Both release the parcel, so the state machine cannot tell them apart -- but they
#: are opposite facts about the buyer, and the difference is the most useful thing
#: the backtest learns about a payment offer. Silence means the channel or the
#: timing was wrong; a refusal means the *offer* was wrong, and only one of those is
#: fixed by raising the incentive.
RESOLUTION_STATE: Dict[Resolution, OrderState] = {
    Resolution.ADDRESS_CORRECTED: OrderState.REPAIRED,
    Resolution.ADDRESS_CONFIRMED: OrderState.REPAIRED,
    Resolution.CONTACT_CONFIRMED: OrderState.VERIFIED,
    Resolution.CONVERTED_TO_PREPAID: OrderState.CONVERTED,
    Resolution.PART_PAID: OrderState.CONVERTED,
    Resolution.OFFER_DECLINED: OrderState.RELEASED_FOR_DISPATCH,
    Resolution.CANCELLED_BY_BUYER: OrderState.CANCELLED,
    Resolution.NO_REPLY: OrderState.RELEASED_FOR_DISPATCH,
    Resolution.MERCHANT_RELEASED: OrderState.RELEASED_FOR_DISPATCH,
    Resolution.MERCHANT_CANCELLED: OrderState.CANCELLED,
}

#: `RESOLUTION_STATE[resolution]` is indexed directly below, so a resolution added
#: without a mapping would raise `KeyError` on the one inbound message that used
#: it -- after the buyer had already replied, which is the worst possible moment to
#: lose a reply. Checked at import instead.
_unmapped = sorted(r.value for r in Resolution if r not in RESOLUTION_STATE)
if _unmapped:
    raise RuntimeError("RESOLUTION_STATE has no entry for: %s"
                       % ", ".join(_unmapped))
del _unmapped


# ==========================================================================
# what the caller must do
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class Disposition:
    """The answer to "may I print this label", plus everything behind it.

    This object *is* the contract, not the database row. If persistence fails after
    the decision is made, the caller still receives `release=True` and the label is
    printed -- see `OrderService.accept`. The row is our record of what we told
    them; their copy is what governs the parcel.
    """

    order_id: str
    state: OrderState
    mode: Mode
    release: bool
    held_until: Optional[str] = None

    action: ActionKind = ActionKind.NO_ACTION       # what was actually attempted
    recommended: ActionKind = ActionKind.NO_ACTION   # what the gate chose
    suppressed: bool = False                         # the mode forbade acting
    blocked_reason: str = ""                         # a guardrail refused

    p_rto_ppm: Ppm = 0
    threshold_ppm: Ppm = 0
    cause: Cause = Cause.UNCERTAIN
    reason: str = ""
    decision_id: Optional[str] = None
    action_id: Optional[str] = None
    degraded: bool = False        # released because we failed, not because we chose
    explanation: Dict[str, Any] = dataclasses.field(default_factory=dict)
    payment_link_id: Optional[str] = None
    payment_link_url: Optional[str] = None
    payment_link_amount_paise: Optional[Paise] = None

    @property
    def held(self) -> bool:
        return self.held_until is not None and not self.release


@dataclasses.dataclass(frozen=True)
class _Plan:
    """Mode policy applied to a verdict: what we will actually attempt."""

    action: ActionKind
    suppressed: bool = False
    note: str = ""


# ==========================================================================
# the service
# ==========================================================================

class OrderService:
    """Accepts orders, runs the pipeline, and guarantees a release.

    Constructed per request from an authenticated `Tenant`. Cheap: it holds
    repositories, not connections.
    """

    def __init__(self, tenant: Tenant, *, decider: Optional[Decider] = None,
                 actor: Optional[Actor] = None, arm: str = "live",
                 payment_provider: Any = None) -> None:
        self._t = tenant
        self._decider = decider or NullDecider()
        self._actor = actor or NoChannelActor()
        self._arm = arm
        if payment_provider is None:
            # Import lazily: adapters.py exposes the provider contract while
            # states.py owns the lifecycle that consumes it.
            from .adapters import default_payment_link_provider
            payment_provider = default_payment_link_provider()
        self._payment_provider = payment_provider
        self.orders = OrderRepo(tenant)
        self.buyers = BuyerRepo(tenant)
        self.actions = ActionRepo(tenant)
        self.decisions = DecisionRepo(tenant)
        self.ledger = LedgerRepo(tenant)

    # ------------------------------------------------------------- accepting

    def accept(self, *, phone: str = "", email: str = "", address: Address,
               value_paise: Paise, external_id: str = "",
               placed_at: Optional[str] = None, payment_method: str = "cod",
               item_count: int = 1, category: str = "", discount_bps: int = 0,
               mode: Optional[Mode] = None,
               consent: Optional[Dict[Channel, bool]] = None) -> Disposition:
        """Take an order and answer whether its label may be printed.

        The order is created and committed *before* the pipeline runs, in its own
        transaction. That ordering is the fail-open's foundation: if the pipeline
        then explodes, there is a durable order to release and a durable id to hand
        back. Creating and reasoning in one transaction would roll the order away
        along with the failure, leaving nothing to release and nothing to explain.
        """
        buyer = self.buyers.upsert(phone=phone, email=email, consent=consent)

        order = self.orders.create(
            buyer_id=buyer.id, value_paise=value_paise, address=address,
            external_id=external_id, placed_at=placed_at,
            payment_method=payment_method, item_count=item_count,
            category=category, discount_bps=discount_bps, mode=mode)

        return self.process(order.id)

    def process(self, order_id: str) -> Disposition:
        """Run the pipeline on an existing order, fail-open on anything.

        Separate from `accept` so a shadow-mode backfill, a retry, and the
        storefront all reach the same code path -- and so the fail-open has exactly
        one implementation.

        The existence check is done here rather than by catching `NotFound` around
        the pipeline. A `NotFound` raised *inside* the pipeline -- a missing buyer
        row for an order that does exist -- is precisely the internal breakage the
        fail-open is for, and distinguishing the two by exception type would send
        that case to the caller as a 404 while an order sat unreleased.
        """
        self.orders.get(order_id)     # raises NotFound: no order, nothing to gate
        try:
            return self._pipeline(order_id)
        except Exception as exc:            # noqa: BLE001 -- deliberate catch-all
            return self._fail_open(order_id, exc)

    # ------------------------------------------------------------- the walk

    def _pipeline(self, order_id: str) -> Disposition:
        """Score, diagnose, decide, then execute any external action after commit.

        Every intermediate state is a real transition with its own audit entry. It
        costs six appends per order and it is worth them: the merchant timeline
        reads "scored at 14:02:01, diagnosed at 14:02:01, decided, held" instead of
        "received, then held", and the second version cannot be argued with.
        """
        with self._t.db.tx():
            order = self.orders.get(order_id)
            if order.state in TERMINAL_STATES:
                return self._settled(order, reason="already terminal")
            if order.state is OrderState.RELEASED_FOR_DISPATCH:
                return self._settled(order, reason="already released")

            buyer = self.buyers.get(order.buyer_id)
            verdict = self._decider.decide(self._t, order, buyer)

            if verdict.feature_payload is not None and verdict.risk_score is not None \
                    and verdict.diagnosis is not None:
                reasoning = self._persist_reasoning(order, verdict)
                verdict = dataclasses.replace(
                    verdict, risk_score_id=reasoning[1], diagnosis_id=reasoning[2])

            order = self.orders.transition(
                order.id, OrderState.SCORING,
                reason="scoring with model %s" % (verdict.model_version or "none"))
            order = self.orders.transition(
                order.id, OrderState.DIAGNOSING,
                reason="cause %s from %d piece(s) of evidence"
                       % (verdict.cause.value, len(verdict.evidence)))
            order = self.orders.transition(
                order.id, OrderState.DECIDING, reason=verdict.reason)

            plan = self._apply_mode(order, verdict)
            decision = self._record_decision(order, verdict, plan)

            if plan.action is ActionKind.NO_ACTION:
                order = self.orders.transition(
                    order.id, OrderState.NO_ACTION,
                    reason=plan.note or verdict.reason)
                order = self.orders.release(
                    order.id, reason="no intervention was justified")
                return self._disposition(order, verdict, plan, decision,
                                         release=True)

        # Provider calls must never run while the decision transaction is open. The
        # durable decision and its reasoning are now committed, so a provider retry
        # cannot erase the evidence of what we attempted.
        if plan.action is ActionKind.CANCEL:
            return self._cancel_path(order, buyer, verdict, plan, decision)

        if plan.action is ActionKind.ESCALATE:
            return self._escalate_path(order, verdict, plan, decision)

        if (order.mode is Mode.GATEWAY and plan.action in (
                ActionKind.PART_PAYMENT, ActionKind.PREPAID_CONVERSION)):
            return self._payment_path(order, buyer, verdict, plan, decision)

        return self._contact_path(order, buyer, verdict, plan, decision)

    # ------------------------------------------------------------ mode policy

    def _apply_mode(self, order: Order, verdict: Verdict) -> _Plan:
        """Translate the gate's choice into what this mode permits.

        Not a filter on top of the decision -- a translation of it, which is why the
        original survives in `_Plan.suppressed` and in the decision row. A merchant
        evaluating shadow mode is evaluating the counterfactual, so throwing it away
        would remove the only reason to run shadow mode at all.
        """
        action = verdict.action
        mode = order.mode

        if verdict.holdout:
            if action is ActionKind.NO_ACTION:
                return _Plan(ActionKind.NO_ACTION,
                             note="holdout order: ship_as_is")
            return _Plan(
                ActionKind.NO_ACTION, suppressed=True,
                note="holdout order: would have taken %s, took nothing"
                     % action.value)

        if mode is Mode.SHADOW:
            if action is ActionKind.NO_ACTION:
                return _Plan(ActionKind.NO_ACTION, note=verdict.reason)
            return _Plan(
                ActionKind.NO_ACTION, suppressed=True,
                note="shadow mode: would have taken %s, took nothing"
                     % action.value)

        if mode is Mode.ADVISORY and action is ActionKind.CANCEL:
            # Cancelling *is* blocking fulfilment, and advisory mode promises never
            # to. So the recommendation goes to a human rather than being executed
            # or silently dropped -- dropping it would hide the one case where the
            # arithmetic says the parcel should not be packed at all.
            return _Plan(
                ActionKind.ESCALATE, suppressed=True,
                note="advisory mode cannot cancel; escalated for a human to decide")

        return _Plan(action, note=verdict.reason)

    # --------------------------------------------------------------- branches

    def _cancel_path(self, order: Order, buyer: Buyer, verdict: Verdict,
                     plan: _Plan, decision: Any) -> Disposition:
        """Cancel, or escalate if a guardrail says a human must sign it off."""
        guard = self._guard(order, buyer, ActionKind.CANCEL, verdict)
        if not guard.allowed:
            record = self.actions.record(
                order.id, ActionKind.CANCEL, status="blocked",
                blocked_reason=guard.code, decision_id=decision.id,
                arm=self._arm)
            escalated = self.orders.transition(
                order.id, OrderState.ESCALATED,
                reason="cancel refused (%s); a human decides" % guard.reason())
            return self._hold_or_release(
                escalated, verdict, plan, decision,
                hold_minutes=self._t.config.max_hold_minutes,
                blocked_reason=guard.code, action_id=record.id)

        record = self.actions.record(order.id, ActionKind.CANCEL,
                                     status="succeeded", decision_id=decision.id,
                                     arm=self._arm)
        cancelled = self.orders.transition(
            order.id, OrderState.CANCELLED,
            reason=verdict.reason or "cancelled before packing")
        # No ledger post here. The cancellation is not an outcome until it is
        # reported as one, and `OutcomeRepo.record` is the single place a terminal
        # outcome becomes money -- a second posting site is how double-counting
        # starts.
        return self._disposition(cancelled, verdict, plan, decision,
                                 release=False, action_id=record.id)

    def _escalate_path(self, order: Order, verdict: Verdict, plan: _Plan,
                       decision: Any) -> Disposition:
        escalated = self.orders.transition(
            order.id, OrderState.ESCALATED, reason=plan.note or verdict.reason)
        return self._hold_or_release(
            escalated, verdict, plan, decision,
            hold_minutes=self._t.config.max_hold_minutes)

    def _payment_path(self, order: Order, buyer: Buyer, verdict: Verdict,
                      plan: _Plan, decision: Any) -> Disposition:
        """Create a payment link after the action intent is durable.

        The link amount is the immutable order value. ``partial_cod`` asks the
        provider to accept a partial payment against that amount; ``prepaid_nudge``
        asks for the full amount. The amount is therefore not a second economics
        catalogue: the order value and the configured ledger determine it.
        """
        guard = self._guard(order, buyer, plan.action, verdict)
        if not guard.allowed:
            record = self.actions.record(
                order.id, plan.action, status="blocked",
                blocked_reason=guard.code, decision_id=decision.id,
                incentive_paise=verdict.incentive_paise, arm=self._arm)
            released = self.orders.release(
                order.id, reason="payment action refused by a guardrail: %s"
                % guard.reason())
            return self._disposition(released, verdict, plan, decision,
                                     release=True, action_id=record.id,
                                     blocked_reason=guard.code)

        acting = self.orders.transition(
            order.id, OrderState.ACTING,
            reason="attempting %s" % plan.action.value)
        record = self.actions.record(
            order.id, plan.action, status="pending", decision_id=decision.id,
            incentive_paise=verdict.incentive_paise, arm=self._arm)
        amount_paise = int(order.value_paise)
        hold_minutes = self._payment_hold_minutes(verdict)
        # ``Clock.plus_minutes`` intentionally returns the storage timestamp. The
        # provider contract uses Razorpay's Unix seconds, so convert only at this
        # boundary without introducing a second time source.
        expire_by = int(parse_ts(
            self._t.runtime.clock.plus_minutes(hold_minutes)).timestamp())
        callback_url = os.environ.get(
            "RAZORPAY_CALLBACK_URL",
            "https://precourier.invalid/v1/webhooks/razorpay")
        notify = {"sms": True, "email": True}
        idempotency_key = "payment:%s" % record.id
        outbox = OutboxRepo(self._t).enqueue(
            order.id, record.id, plan.action,
            {"action": plan.action.value, "amount_paise": amount_paise,
             "expire_by": expire_by, "notify": notify,
             "callback_url": callback_url,
             "accept_partial": plan.action is ActionKind.PART_PAYMENT},
            idempotency_key=idempotency_key)
        OutboxRepo(self._t).mark_processing(outbox.id)

        try:
            response = self._payment_provider.create(
                acting, amount_paise=amount_paise,
                idempotency_key=idempotency_key, expire_by=expire_by,
                notify=notify, callback_url=callback_url,
                accept_partial=plan.action is ActionKind.PART_PAYMENT)
            link_id = str(response.get("id", "")).strip()
            link_url = str(response.get("short_url") or
                            response.get("url") or "").strip()
            if not link_id or not link_url:
                raise ValueError("payment provider returned no link id and URL")
        except Exception:
            OutboxRepo(self._t).fail(outbox.id, "payment_provider_error")
            self.actions.settle(record.id, "failed",
                                blocked_reason="payment_provider_error")
            raise

        provider_name = str(getattr(self._payment_provider, "provider_name",
                                   self._payment_provider.__class__.__name__))
        try:
            with self._t.db.tx():
                self.orders.attach_payment_link(
                    order.id, link_id=link_id, url=link_url,
                    amount_paise=amount_paise)
                PaymentRepo(self._t).record_link(
                    order.id, provider=provider_name, provider_ref=link_id,
                    amount_paise=amount_paise, idempotency_key=idempotency_key)
                OutboxRepo(self._t).succeed(outbox.id, provider_ref=link_id)
                self.actions.settle(record.id, "succeeded")
                self.buyers.record_contact(buyer.id, at=self._t.runtime.stamp())
                waiting = self.orders.transition(
                    acting.id, OrderState.AWAITING_PAYMENT,
                    reason="%s created payment link over %s" %
                           (plan.action.value, provider_name))
                held = self.orders.hold(
                    waiting.id, minutes=hold_minutes,
                    reason=plan.note or verdict.reason)
        except Exception:
            # The provider may already have accepted the link, but the local
            # reconciliation record did not commit. Leave the durable outbox
            # retryable and fail open rather than claiming a successful hold.
            OutboxRepo(self._t).fail(outbox.id, "payment_persistence_error")
            self.actions.settle(record.id, "failed",
                                blocked_reason="payment_persistence_error")
            raise
        return self._disposition(held, verdict, plan, decision, release=False,
                                 action_id=record.id)

    def _payment_hold_minutes(self, verdict: Verdict) -> int:
        wanted = verdict.hold_minutes
        if wanted is None:
            wanted = self._t.config.reply_window_hours * 60
        return min(int(wanted), self._t.config.max_hold_minutes)

    def _contact_path(self, order: Order, buyer: Buyer, verdict: Verdict,
                      plan: _Plan, decision: Any) -> Disposition:
        """Try to reach the buyer. A refusal or a failure releases the order."""
        guard = self._guard(order, buyer, plan.action, verdict)
        if not guard.allowed:
            record = self.actions.record(
                order.id, plan.action, status="blocked",
                blocked_reason=guard.code, decision_id=decision.id,
                incentive_paise=verdict.incentive_paise, arm=self._arm)
            released = self.orders.release(
                order.id, reason="action refused by a guardrail: %s"
                                 % guard.reason())
            return self._disposition(released, verdict, plan, decision,
                                     release=True, action_id=record.id,
                                     blocked_reason=guard.code)

        acting = self.orders.transition(
            order.id, OrderState.ACTING,
            reason="attempting %s" % plan.action.value)

        # The row and the ledger line go in before the side effect. If the send
        # then fails we have already charged for it, which is the correct
        # direction of error: a message that may or may not have left cost money
        # either way, and the alternative is an unrecorded send.
        record = self.actions.record(
            order.id, plan.action, status="pending", decision_id=decision.id,
            incentive_paise=verdict.incentive_paise, arm=self._arm)
        outbox = OutboxRepo(self._t).enqueue(
            order.id, record.id, plan.action,
            {"action": plan.action.value, "cause": verdict.cause.value,
             "p_rto_ppm": verdict.p_rto_ppm,
             "model_version": verdict.model_version},
            idempotency_key="action:%s" % record.id)
        OutboxRepo(self._t).mark_processing(outbox.id)

        try:
            # This is intentionally outside the transaction that created `record`.
            # A real provider may accept a message and then time out locally; the
            # durable action remains available for reconciliation in that case.
            dispatch = self._actor.perform(self._t, acting, buyer, verdict)
        except Exception:
            OutboxRepo(self._t).fail(outbox.id, "actor_error")
            self.actions.settle(record.id, "failed", blocked_reason="actor_error")
            raise

        if not dispatch.sent:
            OutboxRepo(self._t).fail(outbox.id, dispatch.blocked_reason or "not_sent")
            self.actions.settle(record.id, "blocked",
                                blocked_reason=dispatch.blocked_reason
                                               or "not_sent")
            released = self.orders.release(
                acting.id, reason="could not reach the buyer: %s"
                                  % (dispatch.detail or dispatch.blocked_reason))
            return self._disposition(released, verdict, plan, decision,
                                      release=True, action_id=record.id,
                                      blocked_reason=dispatch.blocked_reason)

        if dispatch.channel is None:
            OutboxRepo(self._t).fail(outbox.id, "provider_channel_missing")
            self.actions.settle(record.id, "failed",
                                blocked_reason="provider_channel_missing")
            raise ValueError("a successful provider dispatch must name its channel")
        message_id = dispatch.message_id or self._t.runtime.new_id("message")
        MessageRepo(self._t).record_outbound(
            order.id, buyer.id, action_id=record.id, channel=dispatch.channel,
            body=dispatch.body or dispatch.detail, template_id=dispatch.template_id,
            provider_ref=dispatch.provider_ref, status="sent", message_id=message_id)
        OutboxRepo(self._t).succeed(outbox.id,
                                    provider_ref=dispatch.provider_ref)
        self.actions.settle(record.id, "succeeded")
        self.buyers.record_contact(buyer.id, at=self._t.runtime.stamp())
        waiting = self.orders.transition(
            acting.id, OrderState.AWAITING_BUYER,
            reason="%s sent over %s" % (plan.action.value,
                                        dispatch.channel.value))
        hold_minutes = verdict.hold_minutes
        if hold_minutes is None:
            hold_minutes = self._t.config.reply_window_hours * 60
        return self._hold_or_release(waiting, verdict, plan, decision,
                                     hold_minutes=hold_minutes,
                                     action_id=record.id)

    def _hold_or_release(self, order: Order, verdict: Verdict, plan: _Plan,
                         decision: Any, *, hold_minutes: int,
                         action_id: Optional[str] = None,
                         blocked_reason: str = "") -> Disposition:
        """Gateway holds; every other mode releases now.

        The hold is the only place this system delays a parcel, and it is bounded
        twice -- by `hold_minutes` here and by `max_hold_minutes` inside
        `OrderRepo.hold`, which caps whatever it is given.
        """
        if order.mode is not Mode.GATEWAY:
            released = self.orders.release(
                order.id,
                reason="%s mode releases without waiting" % order.mode.value)
            return self._disposition(released, verdict, plan, decision,
                                     release=True, action_id=action_id,
                                     blocked_reason=blocked_reason)

        held = self.orders.hold(order.id, minutes=hold_minutes,
                                reason=plan.note or verdict.reason)
        return self._disposition(held, verdict, plan, decision, release=False,
                                 action_id=action_id,
                                 blocked_reason=blocked_reason)

    # -------------------------------------------------------------- resolving

    def resolve(self, order_id: str, resolution: Resolution, *,
                address: Optional[Address] = None,
                detail: str = "") -> Disposition:
        """Close a hold: the buyer replied, or a human decided.

        The address supplied with `ADDRESS_CORRECTED` is written as a
        `buyer_confirmed` revision, which is the strongest evidence a repair can
        carry and the one repair source that risks no harm -- the buyer is telling
        us their own address. Our own normalisation is `patched`, and the console
        shows the difference, because "the buyer told us" and "we guessed" are not
        the same claim.
        """
        with self._t.db.tx():
            order = self.orders.get(order_id)
            if order.state in TERMINAL_STATES:
                return self._settled(order, reason="already %s"
                                                   % order.state.value)

            if order.state is OrderState.AWAITING_PAYMENT:
                raise ValueError(
                    "payment holds resolve only through a verified provider event "
                    "or the payment hold deadline")

            if order.state not in HOLDING_STATES \
                    and order.state is not OrderState.RELEASED_FOR_DISPATCH:
                raise ValueError(
                    "resolution is only valid for an order on hold; order %s is "
                    "in state %s" % (order.id, order.state.value))

            if resolution is Resolution.ADDRESS_CORRECTED:
                if address is None:
                    raise ValueError(
                        "ADDRESS_CORRECTED needs the corrected address; "
                        "recording the resolution without it would claim a repair "
                        "that left the address untouched")
                self.orders.patch_address(
                    order.id, address, source="buyer_confirmed",
                    evidence={"resolution": resolution.value,
                              "buyer_said": detail} if detail else
                             {"resolution": resolution.value})

            if resolution not in (Resolution.MERCHANT_RELEASED,
                                  Resolution.MERCHANT_CANCELLED,
                                  Resolution.NO_REPLY):
                self.buyers.record_reply(order.buyer_id,
                                         at=self._t.runtime.stamp())

            target = RESOLUTION_STATE[resolution]
            if order.state is OrderState.RELEASED_FOR_DISPATCH:
                # Advisory mode released before the buyer answered. The reply is
                # still worth keeping -- the label may not be printed yet, and the
                # address correction helps this buyer's next order either way --
                # but the state does not go backwards.
                return self._settled(
                    order,
                    reason="%s arrived after release" % resolution.value)

            order = self._advance(order, target,
                                  reason="%s%s" % (resolution.value,
                                                   ": " + detail if detail else ""))
            if target is not OrderState.RELEASED_FOR_DISPATCH \
                    and target not in TERMINAL_STATES:
                order = self.orders.release(
                    order.id, reason="resolved as %s" % resolution.value)

        return self._settled(order, reason=resolution.value)

    def release(self, order_id: str, *, reason: str,
                actor: str = "merchant") -> Disposition:
        """Release now, whatever state the order is in. The manual override."""
        with self._t.db.tx():
            order = self.orders.get(order_id)
            if order.state in TERMINAL_STATES:
                return self._settled(order, reason="already %s"
                                                   % order.state.value)
            order = self._advance(order, OrderState.RELEASED_FOR_DISPATCH,
                                  reason=reason, actor=actor)
        return self._settled(order, reason=reason)

    def cancel(self, order_id: str, *, reason: str,
               actor: str = "merchant") -> Disposition:
        with self._t.db.tx():
            order = self.orders.get(order_id)
            if order.state in TERMINAL_STATES:
                return self._settled(order, reason="already %s"
                                                   % order.state.value)
            if order.state is OrderState.RELEASED_FOR_DISPATCH:
                raise ValueError(
                    "order %s is already released for dispatch; a released order "
                    "is the carrier's, and cancelling it here would tell the "
                    "merchant a parcel was stopped that is already moving"
                    % order_id)
            order = self._advance(order, OrderState.CANCELLED, reason=reason,
                                  actor=actor)
        return self._settled(order, reason=reason)

    # ------------------------------------------------------------- the sweep

    def sweep(self, *, limit: int = 500) -> List[Disposition]:
        """Release every order whose hold has expired. The timer of PRD section 1.

        Each order is released in its own transaction and its own try. One order
        that cannot be released must not stop the sweep -- otherwise a single stuck
        row holds back every parcel behind it, which is the exact failure the timer
        exists to prevent.
        """
        now = self._t.runtime.stamp()
        released: List[Disposition] = []
        for order in self.orders.expired_holds(now, limit=limit):
            try:
                with self._t.db.tx():
                    fresh = self.orders.get(order.id)
                    if fresh.state in TERMINAL_STATES \
                            or fresh.state is OrderState.RELEASED_FOR_DISPATCH:
                        continue
                    if fresh.state is OrderState.AWAITING_PAYMENT:
                        moved = self._expire_payment_state(
                            fresh,
                            reason="payment hold expired after %d minute(s)"
                                   % self._t.config.max_hold_minutes,
                            actor="timeout")
                    else:
                        moved = self._advance(
                            fresh, OrderState.RELEASED_FOR_DISPATCH,
                            reason="hold expired after %d minute(s); fulfilment "
                                   "latency is sacred"
                                   % self._t.config.max_hold_minutes,
                            actor="timeout")
                released.append(self._settled(moved, reason="hold expired"))
            except Exception as exc:            # noqa: BLE001
                released.append(self._fail_open(order.id, exc))
        return released

    def pending_holds(self) -> List[Order]:
        return self.orders.in_states(sorted(HOLDING_STATES,
                                            key=lambda s: s.value))

    # ----------------------------------------------------------- payment events

    def handle_payment_event(self, order_id: str, *, event_id: str,
                             event: str, payload: Dict[str, Any],
                             provider_ref: str = "", amount_paise: int = 0,
                             signature: str = "") -> Tuple[Disposition, bool]:
        """Apply one verified provider event exactly once.

        The event claim and the lifecycle transition share one transaction. A
        crash before commit leaves the event retryable; a committed duplicate is
        acknowledged without moving an order backwards.
        """
        with self._t.db.tx():
            order = self.orders.get(order_id)
            claimed = PaymentRepo(self._t).claim_event(
                event_id=event_id, event=event, order_id=order_id,
                payload=payload, signature=signature)
            if not claimed:
                return self._settled(order, reason="duplicate payment event"), True

            if event == "payment_link.paid":
                if order.state is OrderState.AWAITING_PAYMENT:
                    PaymentRepo(self._t).record_status(
                        order.id, provider="razorpay", provider_ref=provider_ref,
                        amount_paise=amount_paise,
                        status="paid", idempotency_key="payment-event:%s"
                        % event_id)
                    converted = self.orders.transition(
                        order.id, OrderState.CONVERTED,
                        reason="payment_link.paid verified")
                    order = self.orders.release(
                        converted.id,
                        reason="payment link paid; converted order released",
                        actor="provider")
                return self._settled(order, reason="payment_link.paid"), False

            if event in ("payment_link.expired", "payment_link.cancelled"):
                PaymentRepo(self._t).record_status(
                    order.id, provider="razorpay", provider_ref=provider_ref,
                    amount_paise=amount_paise,
                    status="expired" if event.endswith("expired") else "cancelled",
                    idempotency_key="payment-event:%s" % event_id)
                if order.state is OrderState.AWAITING_PAYMENT:
                    order = self._expire_payment_state(
                        order, reason="%s; applying %s fallback" %
                        (event, self._t.config.payment_fallback),
                        actor="provider")
                return self._settled(order, reason=event), False

            # Unknown-but-signed events are recorded for idempotency and safely
            # ignored. Razorpay retries must not turn an unsupported event into a
            # dispatch hold or a payment state transition.
            return self._settled(order, reason="ignored payment event %s" % event), False

    def _expire_payment_state(self, order: Order, *, reason: str,
                              actor: str) -> Order:
        """Walk expiry through the explicit state before applying fallback."""
        expired = self._advance(order, OrderState.PAYMENT_EXPIRED,
                                reason=reason, actor=actor)
        if self._t.config.payment_fallback == "cancel":
            return self._advance(
                expired, OrderState.CANCELLED,
                reason="payment expired; merchant fallback cancelled order",
                actor=actor)
        return self._advance(
            expired, OrderState.RELEASED_FOR_DISPATCH,
            reason="payment expired; merchant fallback ships COD",
            actor=actor)

    # ------------------------------------------------------------- fail-open

    def _fail_open(self, order_id: str, exc: BaseException) -> Disposition:
        """Release the order because we broke. The last line of the promise.

        Runs in a fresh transaction: the caller's has already rolled back, which is
        why the release attempt is not poisoned by whatever failed.

        If even this fails, the disposition still says `release=True`. That is not
        optimism -- in gateway mode the merchant's integration is gating on our
        answer, and an answer of "I cannot tell you" means an unshipped parcel. So
        we answer, mark it `degraded`, and accept that our own record of the release
        may be missing. Losing our audit row is a problem for us; losing the parcel
        is a problem for the merchant's customer.

        **The exception text goes to the audit log and to the process log, never
        into the `Disposition`.** An exception message is not written for an
        audience: it can carry a file path, a SQL fragment, a hostname or a
        connection string, and this one travels straight out of an API endpoint with
        a 200 status. The caller needs to know that no decision was made -- which is
        what `degraded` says. It does not need our stack.
        """
        detail = "%s: %s" % (type(exc).__name__, exc)
        log.error("failing open on order %s: %s", order_id, detail,
                  exc_info=exc)
        state = OrderState.RELEASED_FOR_DISPATCH
        mode = self._t.mode
        try:
            with self._t.db.tx():
                order = self.orders.get(order_id)
                # Per-order canaries are persisted on the order. The fail-open
                # response must describe that order, not the tenant default.
                mode = order.mode
                if order.state in TERMINAL_STATES:
                    state = order.state
                elif order.state is not OrderState.RELEASED_FOR_DISPATCH:
                    order = self._advance(
                        order, OrderState.RELEASED_FOR_DISPATCH,
                        reason="internal error, releasing rather than holding: %s"
                               % detail,
                        actor="fail_open")
                    state = order.state
                else:
                    state = order.state
                self._t.audit.append(
                    self._t.merchant_id, AuditKind.DECISION_FAILED_OPEN,
                    {"order_id": order_id, "error": detail,
                     "released": state is OrderState.RELEASED_FOR_DISPATCH},
                    order_id=order_id, actor="fail_open")
        except Exception:                       # noqa: BLE001
            # Nothing left to try. The answer below is still a release.
            log.exception("fail-open persistence also failed for order %s",
                          order_id)

        return Disposition(
            order_id=order_id, state=state, mode=mode,
            release=state in PRINTABLE_STATES, degraded=True,
            reason="released without a decision: this system errored while "
                   "reasoning about the order, and fulfilment takes priority "
                   "over our opinion")

    # ------------------------------------------------------------- internals

    def _guard(self, order: Order, buyer: Buyer, action: ActionKind,
               verdict: Verdict) -> guards.GuardResult:
        """Assemble the counters the guard layer needs and run the pass."""
        messages = 0
        if action in CONTACT_ACTIONS:
            messages = self.buyers.messages_since(
                buyer.id, guards.hours_ago(self._t, 24))
        spent = 0
        if action in PAID_ACTIONS:
            spent = self.ledger.spend_since(guards.month_start(self._t),
                                            arm=self._arm)
        has_channel = self._actor.available
        if action in (ActionKind.PART_PAYMENT,
                      ActionKind.PREPAID_CONVERSION):
            # A payment link is a payment-provider side effect, not a message
            # actor send. The API exposes the URL and the merchant may deliver it
            # through its own checkout/channel, so a missing optional messaging
            # adapter must not suppress link creation.
            has_channel = self._payment_provider is not None
        return guards.check(
            self._t, order, buyer, action,
            incentive_paise=verdict.incentive_paise,
            paid_actions_taken=self.actions.paid_count(order.id),
            messages_last_24h=messages, spent_this_month=spent,
            has_channel=has_channel)

    def _advance(self, order: Order, target: OrderState, *, reason: str,
                 actor: str = "agent") -> Order:
        """Move to `target`, walking intermediate states when the edge is missing.

        Used by every out-of-band path -- the sweep, a manual release, the fail-open
        -- because those arrive at arbitrary states. `path_to` supplies the shortest
        legal route, so a release from `RECEIVED` records the steps it passed
        through instead of teleporting.
        """
        if order.state is target:
            return order
        route = path_to(order.state, target)
        if not route:
            raise ValueError("no legal path from %s to %s for order %s"
                             % (order.state.value, target.value, order.id))
        current = order
        for index, step in enumerate(route):
            last = index == len(route) - 1
            current = self.orders.transition(
                order.id, step, actor=actor,
                reason=reason if last
                       else "passing through %s en route to %s"
                            % (step.value, target.value))
        return current

    def _record_decision(self, order: Order, verdict: Verdict,
                         plan: _Plan) -> Any:
        inputs = dict(verdict.gate_inputs)
        inputs.setdefault("p_rto_ppm", verdict.p_rto_ppm)
        inputs.setdefault("cause", verdict.cause.value)
        inputs.setdefault("mode", order.mode.value)
        if plan.suppressed:
            return self.decisions.suppressed(
                order.id, verdict.action, reason=plan.note,
                threshold_ppm=verdict.threshold_ppm,
                expected_delta_micro=verdict.expected_delta_micro,
                gate_inputs=inputs, risk_score_id=verdict.risk_score_id,
                diagnosis_id=verdict.diagnosis_id,
                explanation=verdict.explanation)
        return self.decisions.record(
            order.id, plan.action,
            expected_delta_micro=verdict.expected_delta_micro,
            threshold_ppm=verdict.threshold_ppm, gate_inputs=inputs,
            reason=verdict.reason, risk_score_id=verdict.risk_score_id,
            diagnosis_id=verdict.diagnosis_id,
            explanation=verdict.explanation)

    def _persist_reasoning(self, order: Order, verdict: Verdict) -> Tuple[str, str, str]:
        """Persist the immutable reasoning artifacts inside the pipeline transaction."""
        from .store.repo import ReasoningRepo
        return ReasoningRepo(self._t).record(
            order.id, payload=verdict.feature_payload or {},
            risk_score=verdict.risk_score, diagnosis=verdict.diagnosis)

    def _disposition(self, order: Order, verdict: Verdict, plan: _Plan,
                     decision: Any, *, release: bool,
                     action_id: Optional[str] = None,
                     blocked_reason: str = "") -> Disposition:
        return Disposition(
            order_id=order.id, state=order.state, mode=order.mode,
            release=release, held_until=order.held_until,
            action=plan.action, recommended=verdict.action,
            suppressed=plan.suppressed, blocked_reason=blocked_reason,
            p_rto_ppm=verdict.p_rto_ppm, threshold_ppm=verdict.threshold_ppm,
            cause=verdict.cause, reason=plan.note or verdict.reason,
            decision_id=getattr(decision, "id", None), action_id=action_id,
            explanation=dict(verdict.explanation),
            payment_link_id=order.payment_link_id or None,
            payment_link_url=order.payment_link_url or None,
            payment_link_amount_paise=order.payment_link_amount_paise)

    def _settled(self, order: Order, *, reason: str) -> Disposition:
        """A disposition for an order we did not re-decide this call.

        `release` is read from the state via `PRINTABLE_STATES` rather than
        recomputed. An order still in an open state is not printable regardless of
        mode: shadow and advisory reach `RELEASED_FOR_DISPATCH` within the same
        call, so an open state here means something is genuinely still pending.
        """
        latest = self.decisions.latest(order.id)
        return Disposition(
            order_id=order.id, state=order.state, mode=order.mode,
            release=order.state in PRINTABLE_STATES,
            held_until=order.held_until,
            action=latest.action if latest else ActionKind.NO_ACTION,
            recommended=latest.action if latest else ActionKind.NO_ACTION,
            threshold_ppm=latest.threshold_ppm if latest else 0,
            reason=reason,
            decision_id=latest.id if latest else None,
            explanation=dict(latest.explanation) if latest else {},
            payment_link_id=order.payment_link_id or None,
            payment_link_url=order.payment_link_url or None,
            payment_link_amount_paise=order.payment_link_amount_paise)
