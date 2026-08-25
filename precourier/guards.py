"""Guardrails: the checks that run *after* the gate decides and *before* we act.

Deliberately separate from the gate. The gate answers an economic question -- is
this action worth its cost -- and it can answer yes about an action we are not
permitted to take. Consent, quiet hours, the frequency cap and the budget are not
inputs to that arithmetic; they are vetoes over its output. Folding them into the
gate would let a large enough expected gain outvote a buyer who asked never to be
contacted, which is not a trade-off anyone gets to make.

So every guard returns a *block*, never a score. A blocked action is recorded with
its reason (PRD section 5 requires the refusal to be logged, not silent), the order
continues to fulfilment, and the merchant can produce the evidence later.

The order of the checks matters and is asserted by test: consent first. A buyer who
has withdrawn consent must not have their message count, their order value or the
month's remaining budget consulted at all, because the answer cannot change and
looking is itself a small processing of their data.

Nothing here reads the clock directly -- the `Runtime` on the tenant supplies it,
so a quiet-hours test can be at 03:00 in Kolkata without waiting for 03:00.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import List, Optional, Tuple

from .domain import (ActionKind, Buyer, CONTACT_ACTIONS, Order, PAID_ACTIONS)
from .money import Paise
from .runtime import UTC, parse_ts
from .tenancy import Tenant


# ==========================================================================
# the verdict of the guard layer
# ==========================================================================

#: Stable codes, because these end up in the `actions.blocked_reason` column and
#: in compliance exports. Renaming one would make old rows unattributable, so they
#: are treated like audit kinds: additions are cheap, renames are forbidden.
NO_CONSENT = "no_consent"
NEVER_CONTACT = "never_contact"
QUIET_HOURS = "quiet_hours"
FREQUENCY_CAP = "frequency_cap"
ACTION_CAP = "action_cap"
BUDGET_EXHAUSTED = "budget_exhausted"
INCENTIVE_CEILING = "incentive_ceiling"
CANCEL_DISABLED = "cancel_disabled"
CANCEL_NEEDS_REVIEW = "cancel_needs_review"
NO_CHANNEL = "no_channel"


@dataclasses.dataclass(frozen=True)
class Block:
    """One reason an action may not be taken, in words a merchant can act on."""

    code: str
    detail: str = ""

    def __str__(self) -> str:
        return "%s: %s" % (self.code, self.detail) if self.detail else self.code


@dataclasses.dataclass(frozen=True)
class GuardResult:
    """Whether the action may proceed, and every reason it may not.

    All blocks are collected rather than short-circuiting after the first, with
    one exception -- consent, which stops the pass immediately. The difference is
    the point: for the checks that are merely policy, a merchant fixing one and
    finding a second waiting is a worse experience than being told both at once.
    For consent there is nothing to fix and nothing further to compute.
    """

    blocks: Tuple[Block, ...] = ()

    @property
    def allowed(self) -> bool:
        return not self.blocks

    @property
    def code(self) -> str:
        """The single code stored on the action row. First block wins."""
        return self.blocks[0].code if self.blocks else ""

    def reason(self) -> str:
        return "; ".join(str(b) for b in self.blocks)


ALLOWED = GuardResult()


# ==========================================================================
# individual checks
# ==========================================================================

def local_hour(tenant: Tenant, *, at: Optional[str] = None) -> int:
    """The hour of the day at the buyer's end, per the merchant's stated offset.

    A fixed offset, not a timezone database: `tz_offset_minutes` defaults to +05:30
    for Asia/Kolkata, which has no DST and has not changed since 1945. A merchant
    shipping somewhere that does observe DST sets their own offset and accepts the
    twice-yearly hour of error, which is a better failure than a silent dependency
    on whatever tz data happens to be installed on the host.
    """
    moment = parse_ts(at) if at else tenant.runtime.clock.now()
    shifted = moment.astimezone(UTC) + datetime.timedelta(
        minutes=tenant.config.tz_offset_minutes)
    return shifted.hour


def in_quiet_hours(tenant: Tenant, *, at: Optional[str] = None) -> bool:
    """Whether now is inside the merchant's do-not-disturb window.

    Handles the wrap: 21:00 to 08:00 is a window that crosses midnight, which a
    naive `start <= hour < end` would evaluate as empty. Equal bounds mean no quiet
    hours at all rather than a 24-hour block -- a merchant who sets both to 0 wants
    the feature off, and a system that read that as "never contact anyone" would be
    obeying the letter of a config nobody meant to write.
    """
    start = tenant.config.quiet_hours_start
    end = tenant.config.quiet_hours_end
    if start == end:
        return False
    hour = local_hour(tenant, at=at)
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def hours_ago(tenant: Tenant, hours: int) -> str:
    """A timestamp `hours` before now, from the injected clock."""
    return tenant.runtime.clock.plus_hours(-hours)


def month_start(tenant: Tenant) -> str:
    """The first instant of the current calendar month, UTC.

    The incentive budget is monthly and calendar-aligned rather than a rolling
    30-day window, because that is how a merchant reads their own P&L. A rolling
    window would make "how much have I spent this month" unanswerable from the
    console without explaining the window.
    """
    now = tenant.runtime.clock.now().astimezone(UTC)
    return now.replace(day=1, hour=0, minute=0, second=0,
                       microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


# ==========================================================================
# the pass
# ==========================================================================

def check(tenant: Tenant, order: Order, buyer: Buyer, action: ActionKind, *,
          incentive_paise: Paise = 0, paid_actions_taken: int = 0,
          messages_last_24h: int = 0, spent_this_month: Paise = 0,
          has_channel: bool = True) -> GuardResult:
    """Every reason `action` may not be taken on `order` right now.

    The counts are passed in rather than queried here, so this function is pure and
    the caller decides which of the three arms' spending it is checking against. It
    also means the whole compliance layer is testable without a database, which is
    how the wrap-around quiet-hours cases get exercised.
    """
    if action is ActionKind.NO_ACTION or action is ActionKind.ESCALATE:
        # Neither touches the buyer or the money. Escalation is how a blocked
        # decision reaches a human, so guarding it would close the only door left.
        return ALLOWED

    blocks: List[Block] = []

    if action in CONTACT_ACTIONS:
        if buyer.never_contact:
            # Returned alone. See `GuardResult`: there is nothing else to compute
            # and no other reason worth reporting.
            return GuardResult((Block(
                NEVER_CONTACT,
                "buyer has asked never to be contacted"),))
        if not buyer.consents_to(tenant.config.default_channel):
            return GuardResult((Block(
                NO_CONSENT,
                "no consent on record for %s"
                % tenant.config.default_channel.value),))
        if not has_channel:
            blocks.append(Block(
                NO_CHANNEL,
                "no messaging channel is configured for this deployment"))
        if in_quiet_hours(tenant):
            blocks.append(Block(
                QUIET_HOURS,
                "local hour %02d is inside %02d:00-%02d:00"
                % (local_hour(tenant), tenant.config.quiet_hours_start,
                   tenant.config.quiet_hours_end)))
        if messages_last_24h >= tenant.config.max_messages_per_buyer_24h:
            blocks.append(Block(
                FREQUENCY_CAP,
                "%d message(s) already sent to this buyer in 24h, cap is %d"
                % (messages_last_24h, tenant.config.max_messages_per_buyer_24h)))

    if action in PAID_ACTIONS:
        if paid_actions_taken >= tenant.config.max_paid_actions_per_order:
            blocks.append(Block(
                ACTION_CAP,
                "%d paid action(s) already taken on this order, cap is %d"
                % (paid_actions_taken, tenant.config.max_paid_actions_per_order)))
        if incentive_paise > tenant.config.max_incentive_paise:
            blocks.append(Block(
                INCENTIVE_CEILING,
                "offer of %d paise exceeds the ceiling of %d"
                % (incentive_paise, tenant.config.max_incentive_paise)))
        cost = tenant.config.economics(action).cost_paise + incentive_paise
        budget = tenant.config.monthly_incentive_budget_paise
        if spent_this_month + cost > budget:
            blocks.append(Block(
                BUDGET_EXHAUSTED,
                "%d paise spent this month, this action costs %d, budget is %d"
                % (spent_this_month, cost, budget)))

    if action is ActionKind.CANCEL:
        if not tenant.config.auto_cancel_enabled:
            blocks.append(Block(
                CANCEL_DISABLED,
                "automatic cancellation is switched off for this merchant"))
        if order.value_paise >= tenant.config.value_review_threshold_paise:
            blocks.append(Block(
                CANCEL_NEEDS_REVIEW,
                "order value %d is at or above the review threshold of %d"
                % (order.value_paise, tenant.config.value_review_threshold_paise)))

    return GuardResult(tuple(blocks))
