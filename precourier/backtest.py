"""Zero-dependency controlled evaluation of the PreCourier decision layer.

The backtest is the same economic question a merchant asks during onboarding:
does one cause-matched intervention beat shipping everything or blunt blocking?
Only this module may join observable generated orders to the isolated truth store.
Runtime intelligence modules never import this module or ``worldgen._truth``.

Run locally with::

    python -m precourier.backtest --seed 42 --orders 500

The default output is stable JSON. It contains no wall-clock timestamp or random
identifier, so a report can be compared byte-for-byte between runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import ledger
from .config import MerchantConfig, kolam_config
from .domain import ActionKind, Buyer, Mode, Order, OrderState, OutcomeKind
from .geo import REFERENCE
from .intelligence import build_features, diagnose, evaluate_gate, rule_risk_score
from .money import PPM, Paise, format_paise
from .runtime import SimClock, Runtime, CountingIds
from .worldgen import WorldSpec
from .worldgen.experiment import Outcome as WorldOutcome
from .worldgen.experiment import generate, respond


REPORT_VERSION = "backtest-v1"


@dataclasses.dataclass(frozen=True)
class ArmResult:
    arm: str
    orders: int
    net_paise: int
    net_per_order_paise: int
    returns_prevented: int
    false_positive_cancels: int
    false_positive_cancel_cost_paise: int
    action_spend_paise: int
    wasted_spend_paise: int
    abstentions: int
    acted: int
    delivered: int
    returned: int
    cancelled: int
    decisions: Tuple[Dict[str, Any], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "orders": self.orders,
            "net_paise": self.net_paise,
            "net_rupees": format_paise(self.net_paise, sign=True),
            "net_per_order_paise": self.net_per_order_paise,
            "returns_prevented": self.returns_prevented,
            "false_positive_cancels": self.false_positive_cancels,
            "false_positive_cancel_cost_paise": self.false_positive_cancel_cost_paise,
            "action_spend_paise": self.action_spend_paise,
            "wasted_spend_paise": self.wasted_spend_paise,
            "abstention_rate_ppm": (self.abstentions * PPM // self.orders
                                     if self.orders else 0),
            "acted_rate_ppm": (self.acted * PPM // self.orders
                               if self.orders else 0),
            "abstentions": self.abstentions,
            "acted": self.acted,
            "delivered": self.delivered,
            "returned": self.returned,
            "cancelled": self.cancelled,
            "decisions": list(self.decisions),
        }


def _buyer(raw: Any) -> Buyer:
    phone = str(raw.phone)
    return Buyer(
        id=str(raw.id), merchant_id="backtest",
        phone_hash=hashlib.sha256(phone.encode("utf-8")).hexdigest(),
        phone=phone, email=str(raw.email),
        consent={str(k): bool(v) for k, v in raw.consent.items()},
        never_contact=bool(raw.never_contact),
        prior_orders=int(raw.prior_orders), prior_rtos=int(raw.prior_rtos),
        prior_prepaid=int(raw.prior_prepaid))


def _order(raw: Any) -> Order:
    return Order(
        id=str(raw.id), merchant_id="backtest", buyer_id=str(raw.buyer.id),
        value_paise=int(raw.value_paise),
        address=raw.address, external_id=str(raw.id),
        state=OrderState.RECEIVED,
        mode=Mode.SHADOW, payment_method=str(raw.payment_method),
        item_count=int(raw.item_count), category=str(raw.category),
        is_first_time_buyer=bool(raw.is_first_time_buyer),
        placed_at=str(raw.placed_at))


def _kind(outcome: WorldOutcome) -> OutcomeKind:
    if outcome.cancelled:
        return OutcomeKind.CANCELLED
    if outcome.converted and outcome.delivered:
        return OutcomeKind.PREPAID_DELIVERED
    return OutcomeKind.DELIVERED if outcome.delivered else OutcomeKind.RTO


def _outcome_value(cfg: MerchantConfig, value: Paise, action: ActionKind,
                   outcome: WorldOutcome) -> Tuple[int, int]:
    kind = _kind(outcome)
    breakdown = ledger.breakdown(
        cfg, value, kind, incentive_paise=outcome.incentive_paid_paise)
    action_cost = cfg.economics(action).cost_paise if action is not ActionKind.NO_ACTION else 0
    return breakdown.net_paise - action_cost, action_cost


def _truth_baseline(cfg: MerchantConfig, raw: Any, truth: Any) -> WorldOutcome:
    return respond(
        WorldSpec(seed=0), truth, ActionKind.NO_ACTION,
        value_paise=int(raw.value_paise))


def _best_block_threshold(spec: WorldSpec, orders: Sequence[Any], truths: Mapping[str, Any],
                          cfg: MerchantConfig) -> Tuple[int, ArmResult]:
    candidates = sorted({0, PPM, *(int(t.p_return_ppm) for t in truths.values())})
    best: Optional[Tuple[int, ArmResult]] = None
    for threshold in candidates:
        result = _run_arm(
            "block_best", spec, orders, truths, cfg,
            action_for=lambda raw, truth, _threshold=threshold:
                (ActionKind.CANCEL if int(truth.p_return_ppm) >= _threshold
                 else ActionKind.NO_ACTION, None))
        if best is None or result.net_paise > best[1].net_paise:
            best = (threshold, result)
    assert best is not None
    return best


def _run_arm(name: str, spec: WorldSpec, orders: Sequence[Any],
             truths: Mapping[str, Any], cfg: MerchantConfig,
             action_for: Any) -> ArmResult:
    net = prevented = false_cancels = false_cancel_cost = spend = wasted = 0
    abstentions = acted = delivered = returned = cancelled = 0
    decisions: List[Dict[str, Any]] = []
    for raw in orders:
        truth = truths[str(raw.id)]
        action, decision = action_for(raw, truth)
        incentive = (int(decision.get("incentive_paise", 0))
                     if decision is not None else 0)
        world_outcome = respond(
            spec, truth, action, incentive_paise=incentive,
            value_paise=int(raw.value_paise))
        value, action_cost = _outcome_value(cfg, int(raw.value_paise), action,
                                            world_outcome)
        net += value
        spend += action_cost + int(world_outcome.incentive_paid_paise)
        if action is ActionKind.NO_ACTION:
            abstentions += 1
        else:
            acted += 1
            if not truth.returns_untouched:
                wasted += action_cost + int(world_outcome.incentive_paid_paise)
        if truth.returns_untouched and not world_outcome.returned \
                and not world_outcome.cancelled:
            prevented += 1
        if action is ActionKind.CANCEL and not truth.returns_untouched:
            false_cancels += 1
            false_cancel_cost += ledger.margin_on_delivery(cfg, int(raw.value_paise))
        delivered += int(world_outcome.delivered)
        returned += int(world_outcome.returned)
        cancelled += int(world_outcome.cancelled)
        if decision is not None:
            decision = dict(decision)
            decision.update({
                "outcome": _kind(world_outcome).value,
                "returned": bool(world_outcome.returned),
                "would_have_returned": bool(truth.returns_untouched),
                "mistake_paise": max(0, -value) if value < 0 else 0,
            })
            decisions.append(decision)
    return ArmResult(
        arm=name, orders=len(orders), net_paise=net,
        net_per_order_paise=net // len(orders) if orders else 0,
        returns_prevented=prevented, false_positive_cancels=false_cancels,
        false_positive_cancel_cost_paise=false_cancel_cost,
        action_spend_paise=spend, wasted_spend_paise=wasted,
        abstentions=abstentions, acted=acted, delivered=delivered,
        returned=returned, cancelled=cancelled,
        decisions=tuple(decisions))


def _agent_action(cfg: MerchantConfig, raw: Any, truth: Any) -> Tuple[ActionKind, Optional[Dict[str, Any]]]:
    order = _order(raw)
    buyer = _buyer(raw.buyer)
    features = build_features(
        order, buyer, config=cfg, reference=REFERENCE,
        cod_serviceable=getattr(raw, "cod_serviceable", None))
    score = rule_risk_score(features)
    diagnosis = diagnose(features)
    gate = evaluate_gate(cfg, order, score, diagnosis)
    decision = {
        "order_id": str(raw.id),
        "p_rto_ppm": score.p_rto_ppm,
        "cause": diagnosis.cause.value,
        "action": gate.action.value,
        "threshold_ppm": gate.threshold_ppm,
        "expected_delta_micro": gate.expected_delta_micro,
        "incentive_paise": gate.incentive_paise,
    }
    return gate.action, decision


def run(seed: int = 42, *, n_orders: int = 500,
        include_sensitivity: bool = False,
        include_ten_seeds: bool = False) -> Dict[str, Any]:
    """Run one reproducible world and return a JSON-safe report."""
    spec = WorldSpec(seed=seed, n_orders=n_orders,
                     n_buyers=max(1, min(n_orders, n_orders // 2)))
    cfg = MerchantConfig(
        merchant_id="backtest", monthly_incentive_budget_paise=25_00_000,
        origin_pincode=spec.origin_pincode, auto_cancel_enabled=True,
        value_review_threshold_paise=10_000_000_000)
    with tempfile.TemporaryDirectory(prefix="precourier-backtest-") as directory:
        truth_path = os.path.join(directory, "truth.db")
        with generate(spec, truth_path) as world:
            truths = world.truths()
            orders = tuple(world.orders)
            ship_all = _run_arm(
                "ship_all", spec, orders, truths, cfg,
                action_for=lambda raw, truth: (ActionKind.NO_ACTION, None))
            threshold, block_best = _best_block_threshold(spec, orders, truths, cfg)
            agent = _run_arm(
                "agent", spec, orders, truths, cfg,
                action_for=lambda raw, truth: _agent_action(cfg, raw, truth))
            report: Dict[str, Any] = {
                "report_version": REPORT_VERSION,
                "provenance": "synthetic_world",
                "synthetic_data": True,
                "seed": seed,
                "spec": spec.as_dict(),
                "world": world.summary(),
                "block_best_threshold_ppm": threshold,
                "arms": {
                    "ship_all": ship_all.as_dict(),
                    "block_best": block_best.as_dict(),
                    "agent": agent.as_dict(),
                },
                "comparison": {
                    "agent_net_vs_ship_all_paise": agent.net_paise - ship_all.net_paise,
                    "agent_net_vs_block_best_paise": agent.net_paise - block_best.net_paise,
                    "agent_cost_per_return_prevented_paise": (
                        agent.action_spend_paise // agent.returns_prevented
                        if agent.returns_prevented else None),
                },
                }
            report["where_i_was_wrong"] = sorted(
                [item for item in agent.decisions if item["mistake_paise"] > 0],
                key=lambda item: (-item["mistake_paise"], item["order_id"]))[:10]
    if include_sensitivity:
        report["sensitivity"] = sensitivity(seed=seed, n_orders=n_orders)
    if include_ten_seeds:
        report["ten_seed_summary"] = multi_seed(n_orders=n_orders)
    return report


def multi_seed(*, n_orders: int = 500,
               seeds: Sequence[int] = (20260904, 11, 1729, 4093, 57721,
                                       8675309, 271828, 161803, 66260,
                                       299792)) -> Dict[str, Any]:
    """Aggregate fixed seeds without changing the underlying worlds."""
    reports = [run(seed, n_orders=n_orders) for seed in seeds]
    values = [int(report["arms"]["agent"]["net_per_order_paise"])
              for report in reports]
    mean = sum(values) // len(values) if values else 0
    variance = (sum((value - mean) ** 2 for value in values) // len(values)
                if values else 0)
    return {
        "seeds": list(seeds),
        "orders_per_seed": n_orders,
        "agent_net_per_order_paise": values,
        "mean_net_per_order_paise": mean,
        "stddev_net_per_order_paise": math.sqrt(variance),
        "runs": reports,
    }


def sensitivity(*, seed: int = 42, n_orders: int = 500) -> List[Dict[str, Any]]:
    """Run the declared repairable-share and effectiveness sweeps."""
    rows: List[Dict[str, Any]] = []
    for repairable_share in (0.60, 0.40, 0.20):
        for haircut in (1.0, 0.75, 0.50):
            spec = WorldSpec(seed=seed, n_orders=n_orders,
                             n_buyers=max(1, min(n_orders, n_orders // 2)),
                             repairable_share=repairable_share,
                             effectiveness_haircut=haircut)
            cfg = MerchantConfig(
                merchant_id="backtest", monthly_incentive_budget_paise=25_00_000,
                origin_pincode=spec.origin_pincode, auto_cancel_enabled=True,
                value_review_threshold_paise=10_000_000_000)
            with tempfile.TemporaryDirectory(prefix="precourier-sensitivity-") as directory:
                with generate(spec, os.path.join(directory, "truth.db")) as world:
                    truths = world.truths()
                    agent = _run_arm(
                        "agent", spec, tuple(world.orders), truths, cfg,
                        action_for=lambda raw, truth: _agent_action(cfg, raw, truth))
                    rows.append({
                        "repairable_share": repairable_share,
                        "effectiveness_haircut": haircut,
                        "agent_net_per_order_paise": agent.net_per_order_paise,
                        "returns_prevented": agent.returns_prevented,
                    })
    return rows


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic PreCourier backtest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orders", type=int, default=500)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--ten-seeds", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.orders < 1:
        parser.error("--orders must be positive")
    print(json.dumps(run(args.seed, n_orders=args.orders,
                         include_sensitivity=args.sensitivity,
                         include_ten_seeds=args.ten_seeds),
                     sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = ["REPORT_VERSION", "multi_seed", "run", "sensitivity"]
