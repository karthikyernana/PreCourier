"""The ledger: what an order is worth, and the thresholds that fall out of it.

This module is the arithmetic PRD sections 3 and 5 describe, and nothing else.
It has no opinion about risk, causes, or actions -- it answers only "given this
merchant's cost model, what does each outcome do to the balance, and at what
probability does each action stop being a waste of money".

Two properties matter more than the code:

*   **Thresholds are derived, never chosen.** Nobody picked 57.6%. It is
    `(M + c) / (L + M)` for a particular merchant and order value. Change the
    freight and it moves. `derive_*` returns the inputs alongside the answer so
    the console can render the division a merchant can check by hand.

*   **Nothing is a float.** Expected values accumulate as micro-paise and are
    converted once. See `precourier.money`.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

from .config import MerchantConfig
from .domain import ActionKind, LedgerReason, OutcomeKind
from .money import (PPM, Paise, Ppm, apply_bps, apply_ppm, div_half_up,
                    format_paise, format_ppm)


# ==========================================================================
# what each outcome is worth
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class LedgerLine:
    """One signed money movement, with the reason it happened."""

    reason: LedgerReason
    amount_paise: Paise
    note: str = ""


@dataclasses.dataclass(frozen=True)
class LedgerBreakdown:
    """An outcome's value, itemised. The console renders these lines verbatim."""

    outcome: OutcomeKind
    value_paise: Paise
    lines: Tuple[LedgerLine, ...]

    @property
    def net_paise(self) -> Paise:
        """Balance by summation. There is no stored total to disagree with."""
        return sum(line.amount_paise for line in self.lines)

    def render(self) -> str:
        """The arithmetic, as arithmetic -- PRD section 10.

        A decision a merchant can verify by hand earns more trust than a
        confidence score, so this shows the working rather than the answer.
        """
        if not self.lines:
            return "%s: nothing spent, nothing earned = %s" % (
                self.outcome.value, format_paise(0))
        parts = []
        for line in self.lines:
            sign = "+" if line.amount_paise >= 0 else "-"
            parts.append("%s %s (%s)" % (
                sign, format_paise(abs(line.amount_paise)), line.reason.value))
        body = " ".join(parts).lstrip("+ ")
        return "%s: %s = %s" % (
            self.outcome.value, body, format_paise(self.net_paise, sign=True))


def breakdown(cfg: MerchantConfig, value_paise: Paise, outcome: OutcomeKind,
              incentive_paise: Paise = 0) -> LedgerBreakdown:
    """Itemise what `outcome` does to the balance for an order of `value_paise`."""
    lines: List[LedgerLine] = []

    if outcome is OutcomeKind.DELIVERED:
        lines.append(LedgerLine(LedgerReason.GROSS_MARGIN,
                                apply_bps(value_paise, cfg.gross_margin_bps),
                                "%s of order value" % format_ppm(
                                    cfg.gross_margin_bps * 100, 0)))
        lines.append(LedgerLine(LedgerReason.PACKAGING, -cfg.packaging_paise))
        lines.append(LedgerLine(LedgerReason.FORWARD_FREIGHT,
                                -cfg.forward_freight_paise))

    elif outcome is OutcomeKind.PREPAID_DELIVERED:
        lines.append(LedgerLine(LedgerReason.GROSS_MARGIN,
                                apply_bps(value_paise, cfg.gross_margin_bps)))
        lines.append(LedgerLine(LedgerReason.PACKAGING, -cfg.packaging_paise))
        lines.append(LedgerLine(LedgerReason.FORWARD_FREIGHT,
                                -cfg.forward_freight_paise))
        lines.append(LedgerLine(LedgerReason.GATEWAY_FEE,
                                -apply_bps(value_paise, cfg.gateway_fee_bps)))
        if incentive_paise:
            lines.append(LedgerLine(LedgerReason.INCENTIVE, -incentive_paise,
                                    "discount to pay online now"))

    elif outcome is OutcomeKind.RTO:
        # No gross margin: the buyer never paid. Every cost is still spent.
        lines.append(LedgerLine(LedgerReason.PACKAGING, -cfg.packaging_paise))
        lines.append(LedgerLine(LedgerReason.FORWARD_FREIGHT,
                                -cfg.forward_freight_paise))
        lines.append(LedgerLine(LedgerReason.FAILED_ATTEMPT_FEE,
                                -cfg.failed_attempt_fee_paise))
        lines.append(LedgerLine(LedgerReason.RETURN_FREIGHT,
                                -cfg.return_freight_paise))
        if cfg.goodwill_cost_paise:
            lines.append(LedgerLine(LedgerReason.GOODWILL,
                                    -cfg.goodwill_cost_paise))

    elif outcome is OutcomeKind.CANCELLED:
        pass   # cancelled before packing: nothing was spent

    else:
        raise ValueError("unknown outcome %r" % (outcome,))

    return LedgerBreakdown(outcome, value_paise, tuple(lines))


def margin_on_delivery(cfg: MerchantConfig, value_paise: Paise) -> Paise:
    """`M` -- what a delivered, paid COD order is worth. Positive."""
    return breakdown(cfg, value_paise, OutcomeKind.DELIVERED).net_paise


def loss_on_rto(cfg: MerchantConfig, value_paise: Paise) -> Paise:
    """`L` -- the *magnitude* of an RTO's cost. Positive by convention.

    Returned positive because every formula in PRD section 5 is written with
    `-L`, and a sign that flips between the spec and the code is a bug waiting
    to happen.
    """
    return -breakdown(cfg, value_paise, OutcomeKind.RTO).net_paise


def margin_on_prepaid(cfg: MerchantConfig, value_paise: Paise,
                      incentive_paise: Paise = 0) -> Paise:
    """`P` -- a converted order that delivered. Less than `M` by fee + incentive."""
    return breakdown(cfg, value_paise, OutcomeKind.PREPAID_DELIVERED,
                     incentive_paise).net_paise


def swing(cfg: MerchantConfig, value_paise: Paise) -> Paise:
    """`L + M` -- the gap between delivering and returning. The whole product."""
    return margin_on_delivery(cfg, value_paise) + loss_on_rto(cfg, value_paise)


# ==========================================================================
# derived thresholds
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class DerivedThreshold:
    """A risk threshold, plus the division that produced it.

    `explanation` is rendered in the merchant console and in the exception
    report. An unexplained threshold is indistinguishable from a number
    somebody picked, which is precisely what this product claims not to do.
    """

    action: ActionKind
    threshold_ppm: Ppm
    numerator_paise: Paise
    denominator_paise: Paise
    explanation: str

    @property
    def feasible(self) -> bool:
        """False when no probability could justify the action for this order."""
        return self.threshold_ppm <= PPM


def _threshold(action: ActionKind, numerator: Paise, denominator: Paise,
               explanation: str) -> DerivedThreshold:
    if denominator <= 0:
        # Nothing to gain: the action can never pay for itself at any risk.
        return DerivedThreshold(action, PPM + 1, numerator, denominator,
                                explanation + " -- no achievable gain")
    ppm = div_half_up(numerator * PPM, denominator)
    if ppm < 0:
        ppm = 0
    return DerivedThreshold(action, ppm, numerator, denominator, explanation)


def derive_cancel_threshold(cfg: MerchantConfig,
                            value_paise: Paise) -> DerivedThreshold:
    """Cancel only if doing nothing is worse than forfeiting the margin.

        E[cancel] > E[no-op]
        -c > p.(-L) + (1-p).M
        p.(L+M) > M + c
        p > (M + c) / (L + M)

    At the PRD's worked example -- Rs 1,200, 30% margin, Rs 70 freight -- this
    is 57.6% before the notification cost and 57.8% after it. The agent is
    arithmetically forbidden from cancelling below that.
    """
    margin = margin_on_delivery(cfg, value_paise)
    total_swing = swing(cfg, value_paise)
    cost = cfg.economics(ActionKind.CANCEL).cost_paise
    return _threshold(
        ActionKind.CANCEL, margin + cost, total_swing,
        "p > (M %s + cost %s) / (L+M %s)" % (
            format_paise(margin), format_paise(cost), format_paise(total_swing)))


def derive_repair_threshold(cfg: MerchantConfig, value_paise: Paise,
                            action: ActionKind,
                            effectiveness_ppm: Ppm) -> DerivedThreshold:
    """A repair converts a would-be RTO into a delivery, and sometimes spoils a
    good order.

        D = (L+M).[ p.e - (1-p).h ] - c  >  0
        p > (c + (L+M).h) / ((L+M).(e + h))

    The `h` term is what charges the agent for damaging orders that would have
    delivered anyway. Without it the threshold would be ~1.2%; with h = 1% it is
    ~2.95%, matching PRD section 5.
    """
    total_swing = swing(cfg, value_paise)
    econ = cfg.economics(action)
    harm_ppm = econ.harm_bps * 100          # bps -> ppm
    harm_cost = apply_ppm(total_swing, harm_ppm)
    denominator = apply_ppm(total_swing, effectiveness_ppm + harm_ppm)
    return _threshold(
        action, econ.cost_paise + harm_cost, denominator,
        "p > (cost %s + harm %s) / ((L+M %s) x (e %s + h %s))" % (
            format_paise(econ.cost_paise), format_paise(harm_cost),
            format_paise(total_swing), format_ppm(effectiveness_ppm),
            format_ppm(harm_ppm)))


def derive_prepaid_threshold(cfg: MerchantConfig, value_paise: Paise,
                             effectiveness_ppm: Ppm,
                             incentive_paise: Paise) -> DerivedThreshold:
    """Prepaid conversion's cost on a good order is certain, not probabilistic.

    Converting an order that would have delivered anyway still gives up the
    gateway fee and the incentive, so the penalty term is `e.(M - P)` with no
    harm rate involved -- a sharper model than a guessed `h`.

        D = p.e.(P+L) - (1-p).e.(M-P) - c  >  0
        p > (c + e.(M-P)) / (e.(L+M))
    """
    margin = margin_on_delivery(cfg, value_paise)
    prepaid = margin_on_prepaid(cfg, value_paise, incentive_paise)
    total_swing = swing(cfg, value_paise)
    econ = cfg.economics(ActionKind.PREPAID_CONVERSION)
    giveaway = apply_ppm(margin - prepaid, effectiveness_ppm)
    denominator = apply_ppm(total_swing, effectiveness_ppm)
    return _threshold(
        ActionKind.PREPAID_CONVERSION, econ.cost_paise + giveaway, denominator,
        "p > (cost %s + e x (M-P) %s) / (e %s x (L+M) %s)" % (
            format_paise(econ.cost_paise), format_paise(giveaway),
            format_ppm(effectiveness_ppm), format_paise(total_swing)))


def derive_ladder(cfg: MerchantConfig, value_paise: Paise,
                  effectiveness_ppm: Ppm = 550_000,
                  incentive_paise: Optional[Paise] = None
                  ) -> List[DerivedThreshold]:
    """Every action's threshold for this order, cheapest-firing first.

    PRD section 5: "The gate produces a *ladder*: cheap reversible actions fire
    often, destructive irreversible ones rarely." This returns that ladder, and
    a test asserts the ordering holds -- if a cancel ever became cheaper to
    justify than a repair, the cost model would be broken.
    """
    if incentive_paise is None:
        incentive_paise = min(cfg.max_incentive_paise, 2000)
    ladder = [
        derive_repair_threshold(cfg, value_paise, ActionKind.ADDRESS_REPAIR,
                                effectiveness_ppm),
        derive_repair_threshold(cfg, value_paise, ActionKind.CONTACT_VERIFY,
                                effectiveness_ppm),
        derive_repair_threshold(cfg, value_paise, ActionKind.PART_PAYMENT,
                                effectiveness_ppm),
        derive_prepaid_threshold(cfg, value_paise, effectiveness_ppm,
                                 incentive_paise),
        derive_cancel_threshold(cfg, value_paise),
    ]
    ladder.sort(key=lambda t: t.threshold_ppm)
    return ladder
