"""The guardrails: the vetoes that run after the gate decides and before we act.

These checks are the part of the system a merchant's legal team reads. That makes
them worth testing directly rather than through the API, for two reasons.

The first is coverage. `tests/test_api.py` reaches exactly two of the ten block
codes, because the others need a clock at 03:00, a buyer who was already messaged
today, or a month's budget already spent -- states that are awkward to arrange over
HTTP and trivial to arrange against a pure function.

The second is that the *order* of the checks is itself a promise. Consent is
evaluated first and stops the pass, so a buyer who withdrew consent never has their
message count or the month's remaining budget consulted. That is not an
optimisation; looking is itself a small processing of their data. A test that only
asserted "the action was blocked" would pass just as happily if the order changed,
so the tests below assert the *number* of blocks as well as their codes.

Nothing here sleeps or reads the wall clock: `tenant.runtime.clock` is a `SimClock`
and the quiet-hours cases move it.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from precourier import guards
from precourier.domain import (ActionKind, Address, Buyer, Channel,
                               CONTACT_ACTIONS, Order, PAID_ACTIONS)
from precourier.money import to_paise
from precourier.tenancy import Tenant
from tests.fixtures import GOOD_ADDRESS, make_tenant

ORDER_VALUE = to_paise("1200")

#: Every code the guard layer may write into `actions.blocked_reason`. Duplicated
#: here as literals on purpose. These strings land in compliance exports, so a
#: rename makes historical rows unattributable -- and a test that imported the
#: constants would agree with any rename, which is the opposite of what is wanted.
KNOWN_CODES = frozenset({
    "no_consent", "never_contact", "quiet_hours", "frequency_cap", "action_cap",
    "budget_exhausted", "incentive_ceiling", "cancel_disabled",
    "cancel_needs_review", "no_channel",
})


# ==========================================================================
# helpers
# ==========================================================================

def buyer(tenant: Tenant, *, consents: bool = True,
          never_contact: bool = False,
          channel: Optional[Channel] = None) -> Buyer:
    """A buyer who consents on the merchant's own channel unless told otherwise.

    Consent defaults to `tenant.config.default_channel` rather than to a hardcoded
    one, because consent is per channel: a helper that always granted `whatsapp`
    would have every contact action in this file blocked on `no_consent` and would
    quietly stop testing the checks below it.
    """
    channel = channel or tenant.config.default_channel
    return Buyer(id="byr_1", merchant_id=tenant.merchant_id,
                 phone_hash="0" * 64, phone="+919000000001",
                 consent={channel.value: True} if consents else {},
                 never_contact=never_contact)


def order(tenant: Tenant, *, value_paise: int = ORDER_VALUE,
          address: Address = GOOD_ADDRESS) -> Order:
    return Order(id="ord_1", merchant_id=tenant.merchant_id, buyer_id="byr_1",
                 value_paise=value_paise, address=address)


def at_utc(tenant: Tenant, moment: str) -> Tenant:
    """Move the tenant's simulated clock. Returns the tenant, for chaining."""
    tenant.runtime.clock.set_to(moment)
    return tenant


def codes(result: guards.GuardResult) -> List[str]:
    return [b.code for b in result.blocks]


# ==========================================================================
# consent: first, and final
# ==========================================================================

def test_consent_is_checked_before_anything_else_is_computed():
    """A buyer without consent gets one block, not five.

    The tenant here is at 03:00 local, the buyer is over the frequency cap, the
    action cap is already spent and the month's budget is gone. Every one of those
    would block. Only `no_consent` is returned, which is how the promise in
    `guards.py` -- that a withdrawn consent stops the pass -- is made checkable.
    """
    tenant = at_utc(make_tenant(), "2026-03-01T21:30:00Z")       # 03:00 local

    result = guards.check(
        tenant, order(tenant), buyer(tenant, consents=False),
        ActionKind.ADDRESS_REPAIR,
        messages_last_24h=99, paid_actions_taken=99,
        spent_this_month=tenant.config.monthly_incentive_budget_paise,
        has_channel=False)

    assert codes(result) == [guards.NO_CONSENT]
    assert not result.allowed


def test_never_contact_outranks_a_granted_consent():
    """A `never_contact` flag beats a consent row, and is reported as itself.

    The two can disagree: a buyer ticks the box at checkout and later asks to be
    left alone through support. The later, stronger instruction wins, and the block
    says `never_contact` rather than `no_consent` so the merchant can tell a
    withdrawal from an omission.
    """
    tenant = make_tenant()
    result = guards.check(tenant, order(tenant),
                          buyer(tenant, consents=True, never_contact=True),
                          ActionKind.CONTACT_VERIFY)

    assert codes(result) == [guards.NEVER_CONTACT]


def test_consent_is_per_channel_not_per_buyer():
    """Consent on WhatsApp is not consent on the channel this merchant uses.

    A single "contactable" boolean would be the easy modelling choice and the wrong
    one: under DPDP, consent is given for a purpose and a means. The merchant here
    sends on `simulated`, the buyer ticked `whatsapp`, and the answer is no.
    """
    tenant = make_tenant()
    assert tenant.config.default_channel is Channel.SIMULATED

    result = guards.check(tenant, order(tenant),
                          buyer(tenant, channel=Channel.WHATSAPP),
                          ActionKind.ADDRESS_REPAIR)

    assert codes(result) == [guards.NO_CONSENT]
    assert "simulated" in result.reason()


@pytest.mark.parametrize("action", sorted(CONTACT_ACTIONS, key=lambda a: a.value))
def test_every_contact_action_needs_consent(action):
    """No contact action is exempt.

    Parametrised over the set rather than listing four cases, so adding a fifth way
    to message a buyer cannot ship without a consent check -- the test fails the
    moment `CONTACT_ACTIONS` grows.
    """
    tenant = make_tenant()
    result = guards.check(tenant, order(tenant), buyer(tenant, consents=False),
                          action)
    assert codes(result) == [guards.NO_CONSENT]


# ==========================================================================
# quiet hours
# ==========================================================================

@pytest.mark.parametrize("utc, local_hour, quiet", [
    ("2026-03-01T06:00:00Z", 11, False),    # midday: contactable
    ("2026-03-01T14:30:00Z", 20, False),    # 20:00, the last allowed hour
    ("2026-03-01T15:30:00Z", 21, True),     # 21:00, the inclusive start
    ("2026-03-01T18:00:00Z", 23, True),     # late evening
    ("2026-03-01T21:30:00Z", 3, True),      # 03:00 the next day: past midnight
    ("2026-03-01T02:00:00Z", 7, True),      # 07:30, still inside
    ("2026-03-01T02:30:00Z", 8, False),     # 08:00, the exclusive end
])
def test_quiet_hours_wrap_around_midnight(utc, local_hour, quiet):
    """21:00-08:00 is one window, not an empty one.

    A naive `start <= hour < end` evaluates this range as empty and would message
    buyers at 3am -- the single most expensive compliance bug in the file, since it
    is invisible in any test that happens to run during the day. Both bounds are
    asserted because the inclusive start and exclusive end are a deliberate choice.

    `local_hour` is also asserted, so a failure distinguishes "the window is wrong"
    from "the +05:30 offset is wrong".
    """
    tenant = at_utc(make_tenant(), utc)

    assert guards.local_hour(tenant) == local_hour
    assert guards.in_quiet_hours(tenant) is quiet

    result = guards.check(tenant, order(tenant), buyer(tenant),
                          ActionKind.ADDRESS_REPAIR)
    assert (guards.QUIET_HOURS in codes(result)) is quiet


def test_equal_quiet_hour_bounds_switch_the_feature_off():
    """Both bounds at zero means "no quiet hours", not "never contact anyone".

    A merchant who sets start and end to the same value wants the feature off. The
    other reading -- a 24-hour block -- is defensible from the arithmetic and
    indefensible from intent, and it would silently disable every contact action
    for that merchant with no error anywhere.
    """
    tenant = at_utc(make_tenant(quiet_hours_start=0, quiet_hours_end=0),
                    "2026-03-01T21:30:00Z")                      # 03:00 local

    assert guards.in_quiet_hours(tenant) is False
    result = guards.check(tenant, order(tenant), buyer(tenant),
                          ActionKind.ADDRESS_REPAIR)
    assert result.allowed


def test_a_daytime_quiet_window_does_not_wrap():
    """The non-wrapping branch still works: 08:00-21:00 blocks the middle of the day.

    Nobody configures this, but the code has two branches and only one of them is
    exercised by the default config. An untested branch in a compliance check is
    the branch that breaks when the defaults change.
    """
    tenant = at_utc(make_tenant(quiet_hours_start=8, quiet_hours_end=21),
                    "2026-03-01T06:00:00Z")                      # 11:30 local
    assert guards.in_quiet_hours(tenant) is True

    at_utc(tenant, "2026-03-01T21:30:00Z")                       # 03:00 local
    assert guards.in_quiet_hours(tenant) is False


def test_the_merchants_own_offset_decides_the_local_hour():
    """Quiet hours are local to the buyer, per the merchant's stated offset.

    A merchant shipping from a +00:00 origin at the same instant is at 22:00, not
    at 03:30, and the check must follow the config rather than the host's timezone
    -- which is why `tz_offset_minutes` exists instead of a call to `localtime`.
    """
    instant = "2026-03-01T22:00:00Z"
    kolkata = at_utc(make_tenant(), instant)
    utc_shop = at_utc(make_tenant(tz_offset_minutes=0), instant)

    assert guards.local_hour(kolkata) == 3
    assert guards.local_hour(utc_shop) == 22
    assert guards.in_quiet_hours(kolkata) is True
    assert guards.in_quiet_hours(utc_shop) is True               # 22:00 is late too


# ==========================================================================
# the caps
# ==========================================================================

def test_the_frequency_cap_counts_messages_not_actions():
    """One message per buyer per 24h, and the cap is a floor not a ceiling.

    `>=` rather than `>`: with a cap of 1, a buyer who has had one message is done.
    The off-by-one here is worth a test because its failure mode is doubling every
    merchant's message volume, which reads as a successful campaign until the
    complaints arrive.
    """
    tenant = make_tenant()
    assert tenant.config.max_messages_per_buyer_24h == 1

    fresh = guards.check(tenant, order(tenant), buyer(tenant),
                         ActionKind.CONTACT_VERIFY, messages_last_24h=0)
    contacted = guards.check(tenant, order(tenant), buyer(tenant),
                             ActionKind.CONTACT_VERIFY, messages_last_24h=1)

    assert fresh.allowed
    assert codes(contacted) == [guards.FREQUENCY_CAP]
    assert "cap is 1" in contacted.reason()


def test_one_paid_action_per_order():
    """The action cap stops message stacking on a single order.

    Two paid actions on one order is how a well-meaning agent turns a Rs 3
    intervention into a Rs 6 one and a single question into an argument. The cap is
    per order and configurable; the default is one.
    """
    tenant = make_tenant()
    assert tenant.config.max_paid_actions_per_order == 1

    first = guards.check(tenant, order(tenant), buyer(tenant),
                         ActionKind.ADDRESS_REPAIR, paid_actions_taken=0)
    second = guards.check(tenant, order(tenant), buyer(tenant),
                          ActionKind.ADDRESS_REPAIR, paid_actions_taken=1)

    assert first.allowed
    assert codes(second) == [guards.ACTION_CAP]


@pytest.mark.parametrize("action", sorted(PAID_ACTIONS, key=lambda a: a.value))
def test_every_paid_action_is_capped_and_budgeted(action):
    """No paid action escapes the cap.

    Same reasoning as the consent parametrisation: the guarantee is about the set,
    so the test iterates the set. A new paid action added to `PAID_ACTIONS` without
    a cap check would be the one action a merchant could be billed for without
    limit.
    """
    tenant = make_tenant()
    result = guards.check(tenant, order(tenant), buyer(tenant), action,
                          paid_actions_taken=1)
    assert guards.ACTION_CAP in codes(result)


def test_an_offer_above_the_ceiling_is_refused_but_the_ceiling_itself_is_not():
    """`>` not `>=`: an offer of exactly the ceiling is a legal offer.

    A ceiling nobody may reach is a different number from the one the merchant
    typed. Both sides of the boundary are asserted so the comparison cannot be
    quietly tightened.
    """
    tenant = make_tenant()
    ceiling = tenant.config.max_incentive_paise

    at_ceiling = guards.check(tenant, order(tenant), buyer(tenant),
                              ActionKind.PREPAID_CONVERSION,
                              incentive_paise=ceiling)
    over = guards.check(tenant, order(tenant), buyer(tenant),
                        ActionKind.PREPAID_CONVERSION,
                        incentive_paise=ceiling + 1)

    assert at_ceiling.allowed
    assert codes(over) == [guards.INCENTIVE_CEILING]


def test_the_monthly_budget_counts_the_action_cost_not_just_the_incentive():
    """A Rs 2 message spends the budget too.

    The interesting bug is a budget that only debits discounts: the messages are
    cheap individually and a merchant running a hundred thousand orders a month
    discovers the aggregate on their invoice. So the check is
    `spent + cost + incentive > budget`, and this test sits one paise from the edge
    with a zero incentive, where an incentive-only implementation would allow it.
    """
    tenant = make_tenant()
    budget = tenant.config.monthly_incentive_budget_paise
    cost = tenant.config.economics(ActionKind.ADDRESS_REPAIR).cost_paise
    assert cost > 0

    exact = guards.check(tenant, order(tenant), buyer(tenant),
                         ActionKind.ADDRESS_REPAIR,
                         spent_this_month=budget - cost)
    over = guards.check(tenant, order(tenant), buyer(tenant),
                        ActionKind.ADDRESS_REPAIR,
                        spent_this_month=budget - cost + 1)

    assert exact.allowed
    assert codes(over) == [guards.BUDGET_EXHAUSTED]
    assert "budget is %d" % budget in over.reason()


def test_the_budget_is_the_calendar_month_a_merchant_reads_in_their_pandl():
    """`month_start` is the 1st at midnight UTC, not 30 days ago.

    A rolling window would make "how much have I spent this month" unanswerable
    from the console without also explaining the window, and the merchant's own P&L
    is calendar-aligned. Asserted on a mid-month instant so a naive
    `now - 30 days` cannot pass.
    """
    tenant = at_utc(make_tenant(), "2026-03-15T13:45:11Z")
    assert guards.month_start(tenant) == "2026-03-01T00:00:00Z"


# ==========================================================================
# cancellation: the irreversible one
# ==========================================================================

def test_automatic_cancellation_is_off_until_a_merchant_turns_it_on():
    """The default is off, and the default is the point.

    Cancelling is the only irreversible action in the system and the only one that
    destroys revenue the merchant might have collected. A platform default of "on"
    would mean the first thing a new merchant experiences is us cancelling their
    orders on a model they have not yet seen evidence from.
    """
    tenant = make_tenant()
    assert tenant.config.auto_cancel_enabled is False

    result = guards.check(tenant, order(tenant, value_paise=to_paise("400")),
                          buyer(tenant), ActionKind.CANCEL)
    assert guards.CANCEL_DISABLED in codes(result)


def test_a_high_value_cancel_needs_a_human_even_when_auto_cancel_is_on():
    """At or above the review threshold, the agent escalates instead of cancelling.

    `>=`, because a merchant setting the threshold to Rs 5,000 means "Rs 5,000 is
    already a lot of money". Below it the agent may act; at it, a person decides.
    """
    tenant = make_tenant(auto_cancel_enabled=True)
    threshold = tenant.config.value_review_threshold_paise

    below = guards.check(tenant, order(tenant, value_paise=threshold - 1),
                         buyer(tenant), ActionKind.CANCEL)
    at = guards.check(tenant, order(tenant, value_paise=threshold),
                      buyer(tenant), ActionKind.CANCEL)

    assert below.allowed
    assert codes(at) == [guards.CANCEL_NEEDS_REVIEW]


def test_cancelling_does_not_need_consent_or_wait_for_morning():
    """A cancel is not a message, so the contact guards do not apply to it.

    Worth stating: a buyer who never consented to being messaged has not thereby
    bought a guarantee that their order ships. Cancellation is judged on the
    merchant's own policy -- the switch and the value threshold -- and nothing else.
    """
    tenant = at_utc(make_tenant(auto_cancel_enabled=True),
                    "2026-03-01T21:30:00Z")                      # 03:00 local

    result = guards.check(tenant, order(tenant, value_paise=to_paise("400")),
                          buyer(tenant, consents=False), ActionKind.CANCEL,
                          messages_last_24h=99)

    assert result.allowed


# ==========================================================================
# the doors that must stay open
# ==========================================================================

@pytest.mark.parametrize("action", [ActionKind.NO_ACTION, ActionKind.ESCALATE])
def test_doing_nothing_and_asking_a_human_are_never_blocked(action):
    """Both are unguarded, deliberately.

    Neither touches the buyer or the money. Escalation matters most: it is how a
    decision that every other guard refused still reaches a person. Guarding it
    would close the last door and leave the order in the hold queue until the
    timeout -- turning a compliance check into a fulfilment delay.

    The buyer here has asked never to be contacted and the budget is gone.
    """
    tenant = at_utc(make_tenant(), "2026-03-01T21:30:00Z")

    result = guards.check(
        tenant, order(tenant), buyer(tenant, never_contact=True), action,
        messages_last_24h=99, paid_actions_taken=99,
        incentive_paise=to_paise("10000"),
        spent_this_month=tenant.config.monthly_incentive_budget_paise,
        has_channel=False)

    assert result.allowed
    assert result.code == ""


def test_escalation_survives_being_reclassified_as_a_contact_action(monkeypatch):
    """The early return protects escalation on its own, not just by set membership.

    Today `ESCALATE` is in neither `CONTACT_ACTIONS` nor `PAID_ACTIONS`, so the test
    above passes even if the early return is deleted -- the two mechanisms are
    redundant, and a mutation that removes one is invisible. That redundancy is
    worth keeping and worth pinning, because the sets are the thing likely to
    change: escalation does arguably notify a human, and somebody will eventually
    reason their way to adding it.

    So this test forces the collision. With `ESCALATE` in both sets, a never-contact
    buyer, no consent, 03:00 local and an exhausted budget, five separate checks
    would fire -- and the answer must still be `allowed`, because escalation is the
    only door a blocked decision has left. Closing it would convert every guard
    refusal into a hold that runs to timeout.
    """
    monkeypatch.setattr(
        guards, "CONTACT_ACTIONS", CONTACT_ACTIONS | {ActionKind.ESCALATE})
    monkeypatch.setattr(
        guards, "PAID_ACTIONS", PAID_ACTIONS | {ActionKind.ESCALATE})

    tenant = at_utc(make_tenant(), "2026-03-01T21:30:00Z")       # 03:00 local
    result = guards.check(
        tenant, order(tenant), buyer(tenant, consents=False, never_contact=True),
        ActionKind.ESCALATE, messages_last_24h=99, paid_actions_taken=99,
        spent_this_month=tenant.config.monthly_incentive_budget_paise,
        has_channel=False)

    assert result.allowed, "escalation was blocked: %s" % result.reason()


def test_a_deployment_with_no_channel_blocks_rather_than_pretends():
    """No configured channel is a block, not a silent success.

    This is the day-one state of every deployment: keys are not wired yet. The
    order must still reach dispatch, and the action must be recorded as blocked
    with a reason -- because the alternative is an order sitting in `awaiting_buyer`
    waiting for a reply to a message that was never sent.
    """
    tenant = make_tenant()
    result = guards.check(tenant, order(tenant), buyer(tenant),
                          ActionKind.ADDRESS_REPAIR, has_channel=False)
    assert codes(result) == [guards.NO_CHANNEL]


# ==========================================================================
# the shape of the report
# ==========================================================================

def test_policy_blocks_are_reported_together_rather_than_one_at_a_time():
    """Four separate reasons, all four returned.

    A merchant fixing one problem, retrying and finding a second waiting is a worse
    experience than being told all four at once -- and the counts here are asserted
    exactly, so a short-circuit added for tidiness shows up as a failing test
    rather than as a merchant's fourth support ticket.
    """
    tenant = at_utc(make_tenant(), "2026-03-01T21:30:00Z")       # 03:00 local

    result = guards.check(
        tenant, order(tenant), buyer(tenant), ActionKind.PREPAID_CONVERSION,
        incentive_paise=tenant.config.max_incentive_paise + 1,
        paid_actions_taken=1, messages_last_24h=1,
        spent_this_month=tenant.config.monthly_incentive_budget_paise)

    assert set(codes(result)) == {
        guards.QUIET_HOURS, guards.FREQUENCY_CAP, guards.ACTION_CAP,
        guards.INCENTIVE_CEILING, guards.BUDGET_EXHAUSTED}
    assert len(result.blocks) == 5

    # One code goes on the action row, and it is the first -- stable, because the
    # checks run in a fixed order rather than over a set.
    assert result.code == guards.QUIET_HOURS


def test_every_block_carries_a_detail_a_merchant_can_act_on():
    """A code alone is not an explanation.

    PRD section 5 requires the refusal to be logged rather than silent, and a log
    line reading `frequency_cap` does not tell a merchant which cap or what the
    limit was. Every block gets a sentence with the numbers in it.
    """
    tenant = at_utc(make_tenant(auto_cancel_enabled=True),
                    "2026-03-01T21:30:00Z")

    seen = set()
    for result in (
        guards.check(tenant, order(tenant), buyer(tenant, consents=False),
                     ActionKind.ADDRESS_REPAIR),
        guards.check(tenant, order(tenant), buyer(tenant, never_contact=True),
                     ActionKind.ADDRESS_REPAIR),
        guards.check(tenant, order(tenant), buyer(tenant),
                     ActionKind.PREPAID_CONVERSION,
                     incentive_paise=tenant.config.max_incentive_paise + 1,
                     paid_actions_taken=1, messages_last_24h=1,
                     spent_this_month=(
                         tenant.config.monthly_incentive_budget_paise),
                     has_channel=False),
        guards.check(tenant, order(tenant, value_paise=to_paise("9000")),
                     buyer(tenant), ActionKind.CANCEL),
        guards.check(make_tenant(), order(tenant), buyer(tenant),
                     ActionKind.CANCEL),
    ):
        for block in result.blocks:
            assert block.detail, "%s has no detail" % block.code
            assert str(block).startswith(block.code + ": ")
            seen.add(block.code)

    # Every declared code is reachable. A code that no input can produce is either
    # dead policy or a check that silently stopped running.
    assert seen == KNOWN_CODES


def test_the_block_codes_are_stable_strings():
    """These are written into `actions.blocked_reason` and exported to auditors.

    Compared against literals rather than against the constants, so a rename fails
    here instead of silently making every historical row unattributable. Additions
    are cheap: add the literal. Renames are not, which is the whole point.
    """
    declared = {value for name, value in vars(guards).items()
                if name.isupper() and isinstance(value, str)
                and not name.startswith("_")}
    assert declared == KNOWN_CODES


def test_an_allowed_result_is_falsy_about_nothing():
    """`ALLOWED` is empty, allowed, and has no code or reason.

    The shared constant is returned by the two unguarded actions, so it must be
    immutable in effect: a caller that appended to `.blocks` would corrupt every
    future pass. `blocks` is a tuple, which is that guarantee.
    """
    assert guards.ALLOWED.allowed
    assert guards.ALLOWED.blocks == ()
    assert guards.ALLOWED.code == ""
    assert guards.ALLOWED.reason() == ""
    assert isinstance(guards.ALLOWED.blocks, tuple)
