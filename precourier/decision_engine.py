"""Stage 1 cause probabilities and brute-force rupee decision engine.

The implementation is intentionally deterministic.  It is the ``RiskModel``
baseline seam; a trained model can implement the same protocol later.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Dict, Iterable, Mapping, Protocol, Sequence

from .domain import Cause, Order
from .intelligence import Features
from .money import PPM, Paise


CAUSES = ("refusal", "unreachable_buyer", "bad_address")


@dataclasses.dataclass(frozen=True)
class CauseProbabilities:
    refusal_ppm: int
    unreachable_buyer_ppm: int
    bad_address_ppm: int
    rto_ppm: int

    def as_dict(self) -> Dict[str, int]:
        return {"refusal": self.refusal_ppm, "unreachable_buyer": self.unreachable_buyer_ppm,
                "bad_address": self.bad_address_ppm, "rto": self.rto_ppm}


class RiskModel(Protocol):
    def predict(self, order: Order, features: Features) -> CauseProbabilities: ...


def _clip(value: int) -> int:
    return max(0, min(PPM, int(value)))


class RuleRiskModel:
    """Interpretable baseline: structural rules plus optional provider signals.

    External scores are blended at 20% per signal (capped at 60% total), and a
    label maps to its cause. This fixed conservative blend is documented so it
    can be calibrated against outcomes without changing the API.
    """
    def predict(self, order: Order, features: Features) -> CauseProbabilities:
        refusal = 35_000 + min(500_000, features.refusal_ratio_ppm // 2)
        refusal += 100_000 if features.is_first_time_buyer else 0
        unreachable = 25_000 + (250_000 if not features.phone_format_valid else 0)
        unreachable += min(200_000, features.duplicate_phone_count * 40_000)
        bad = 25_000 + (180_000 if not features.has_house_number else 0)
        bad += 120_000 if not features.pincode_city_consistent else 0
        bad += 100_000 if not features.pincode_known else 0
        adjustments = {cause: 0 for cause in CAUSES}
        for signal in order.external_signals:
            try:
                score = _clip(round(float(signal.get("score", 0)) * PPM))
            except (TypeError, ValueError):
                continue
            label = str(signal.get("label", "")).casefold()
            cause = "bad_address" if "address" in label else "unreachable_buyer" if "reach" in label or "phone" in label else "refusal"
            adjustments[cause] += min(200_000, score // 5)
        refusal, unreachable, bad = (_clip(refusal + adjustments["refusal"]),
                                     _clip(unreachable + adjustments["unreachable_buyer"]),
                                     _clip(bad + adjustments["bad_address"]))
        # Causes are competing failure explanations; union is bounded and stable.
        rto = _clip(refusal + unreachable + bad - (refusal * unreachable // PPM)
                    - (refusal * bad // PPM) - (unreachable * bad // PPM))
        return CauseProbabilities(refusal, unreachable, bad, rto)


@dataclasses.dataclass(frozen=True)
class ActionSpec:
    name: str
    direct_cost_paise: int
    reductions_ppm: Mapping[str, int]
    abandonment_ppm: int = 0
    delay_cost_paise: int = 0


DEFAULT_ACTIONS = (
    ActionSpec("ship_as_is", 0, {}),
    ActionSpec("confirm_intent", 200, {"refusal": 300_000}, 50_000, 50),
    ActionSpec("fix_address", 300, {"bad_address": 650_000}, 30_000, 50),
    ActionSpec("partial_cod", 100, {"refusal": 450_000}, 100_000, 100),
    ActionSpec("prepaid_nudge", 200, {"refusal": 500_000}, 80_000, 100),
    ActionSpec("cancel", 100, {"refusal": PPM, "unreachable_buyer": PPM, "bad_address": PPM}),
)


@dataclasses.dataclass(frozen=True)
class ActionEvaluation:
    action: str
    expected_net_paise: int
    eligible: bool = True


@dataclasses.dataclass(frozen=True)
class Decision:
    chosen_action: str
    evaluations: tuple[ActionEvaluation, ...]
    runner_up: str
    expected_saving_paise: int
    top_drivers: tuple[str, ...]
    probabilities: CauseProbabilities


def expected_net_paise(order_value_paise: int, probabilities: CauseProbabilities,
                       action: ActionSpec, *, margin_bps: int = 3000,
                       rto_loss_paise: int = 19500) -> int:
    delivered = order_value_paise * margin_bps // 10_000
    p = probabilities.rto_ppm
    reductions = sum(probabilities.as_dict().get(c, 0) * reduction // PPM
                     for c, reduction in action.reductions_ppm.items())
    remaining = max(0, p - reductions)
    abandon = max(0, PPM - p) * action.abandonment_ppm // PPM
    return ((PPM - remaining) * delivered - remaining * rto_loss_paise) // PPM - action.direct_cost_paise - action.delay_cost_paise - abandon * delivered // PPM


def choose_action(order: Order, probabilities: CauseProbabilities, *, actions: Sequence[ActionSpec] = DEFAULT_ACTIONS,
                  margin_bps: int = 3000, rto_loss_paise: int = 19500) -> Decision:
    values = tuple(ActionEvaluation(a.name, expected_net_paise(order.value_paise, probabilities, a,
                                                               margin_bps=margin_bps, rto_loss_paise=rto_loss_paise)) for a in actions)
    ranked = sorted(values, key=lambda x: (x.expected_net_paise, x.action), reverse=True)
    baseline = next(x.expected_net_paise for x in values if x.action == "ship_as_is")
    chosen = ranked[0]
    return Decision(chosen.action, values, ranked[1].action if len(ranked) > 1 else chosen.action,
                    max(0, chosen.expected_net_paise - baseline),
                    tuple(sorted(("refusal", "unreachable_buyer", "bad_address"),
                                 key=lambda c: probabilities.as_dict()[c], reverse=True)[:2]), probabilities)


def in_holdout(order_id: str, merchant_id: str, rate_percent: int = 10) -> bool:
    bucket = int(hashlib.sha256((merchant_id + ":" + order_id).encode()).hexdigest()[:8], 16) % 100
    return bucket < rate_percent
