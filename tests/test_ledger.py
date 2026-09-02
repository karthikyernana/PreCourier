"""The ledger: does the arithmetic in PRD section 3 actually come out?

The four numbers in `test_prd_worked_example` are the ones the PRD publishes and
the demo narrates. If they drift, either the spec or the cost model is wrong and
somebody has to decide which -- so they are asserted exactly, not approximately.

The rest of the file asserts the properties that make those numbers trustworthy:
that they are sums rather than stored totals, that they move when a merchant's
freight moves, and that no code path can rewrite one after the fact.
"""

from __future__ import annotations

import pytest
import sqlite3

from precourier import ledger
from precourier.config import MerchantConfig
from precourier.domain import ActionKind, LedgerReason, OutcomeKind
from precourier.money import to_paise
from precourier.store.db import StorageError
from precourier.store.repo import (ACTION_SETTLED_AUDIT, ActionRepo, BuyerRepo,
                                    Conflict, LedgerRepo, OrderRepo)
from tests.fixtures import GOOD_ADDRESS, make_tenant, two_tenants

ORDER_VALUE = to_paise("1200")

#: PRD line 140 spells the prepaid figure as `265 - 24 gateway fee - 20
#: incentive`, so +Rs 221 is the *incentivised* conversion. Without a discount it
#: is +Rs 241. Both are asserted, because the difference is exactly what the
#: prepaid threshold trades away and a test that conflated them would hide a
#: mis-priced incentive.
PREPAID_INCENTIVE = to_paise("20")


# ==========================================================================
# the published numbers
# ==========================================================================

def test_prd_worked_example():
    """Rs 1,200 at 30% margin: +265 delivered, -195 returned, 0 cancelled, +221 prepaid.

    PRD section 3's worked example, asserted to the paise. The swing between
    delivering and returning is Rs 460, and that asymmetry is the entire product
    -- every threshold in the system is a ratio involving it.
    """
    cfg = make_tenant().config

    delivered = ledger.breakdown(cfg, ORDER_VALUE, OutcomeKind.DELIVERED)
    rto = ledger.breakdown(cfg, ORDER_VALUE, OutcomeKind.RTO)
    cancelled = ledger.breakdown(cfg, ORDER_VALUE, OutcomeKind.CANCELLED)
    prepaid = ledger.breakdown(cfg, ORDER_VALUE, OutcomeKind.PREPAID_DELIVERED,
                               incentive_paise=PREPAID_INCENTIVE)

    assert delivered.net_paise == to_paise("265")
    assert rto.net_paise == -to_paise("195")
    assert cancelled.net_paise == 0
    assert prepaid.net_paise == to_paise("221")

    # The same conversion with no discount offered.
    assert ledger.breakdown(
        cfg, ORDER_VALUE, OutcomeKind.PREPAID_DELIVERED).net_paise == to_paise("241")

    assert ledger.margin_on_delivery(cfg, ORDER_VALUE) == to_paise("265")
    assert ledger.loss_on_rto(cfg, ORDER_VALUE) == to_paise("195")
    assert ledger.swing(cfg, ORDER_VALUE) == to_paise("460")


def test_cancel_threshold_is_57_6_percent():
    """The PRD's headline threshold, derived rather than chosen.

    `(M + c) / (L + M)` = 265 / 460 = 57.6%. With the Rs 1 notification cost it is
    57.8%. Nobody picked either number; changing the freight moves both.
    """
    cfg = make_tenant().config
    free = cfg.replace(action_economics={
        **cfg.action_economics,
        ActionKind.CANCEL: cfg.economics(ActionKind.CANCEL).__class__(
            cost_paise=0, harm_bps=0, reversible=False, contacts_buyer=True)})

    assert ledger.derive_cancel_threshold(free, ORDER_VALUE).threshold_ppm == 576_087
    assert ledger.derive_cancel_threshold(cfg, ORDER_VALUE).threshold_ppm == 578_261


def test_repair_threshold_matches_prd():
    """~2.95% for an address repair -- PRD section 5 states ~2.8%.

    The difference is the harm term. Without `h` the threshold is 1.19%: the agent
    would repair almost everything, because a wasted Rs 3 is trivial against a
    Rs 460 swing. Charging it 1% of the swing for damaging orders that would have
    delivered anyway is what produces a number in the PRD's stated range, and what
    stops the agent from touching every order it sees.
    """
    cfg = make_tenant().config
    derived = ledger.derive_repair_threshold(
        cfg, ORDER_VALUE, ActionKind.ADDRESS_REPAIR, effectiveness_ppm=550_000)

    assert 29_000 <= derived.threshold_ppm <= 30_000, derived.explanation
    assert derived.feasible

    harmless = cfg.replace(action_economics={
        **cfg.action_economics,
        ActionKind.ADDRESS_REPAIR: cfg.economics(
            ActionKind.ADDRESS_REPAIR).__class__(
                cost_paise=300, harm_bps=0, reversible=True,
                contacts_buyer=True)})
    without_harm = ledger.derive_repair_threshold(
        harmless, ORDER_VALUE, ActionKind.ADDRESS_REPAIR,
        effectiveness_ppm=550_000)
    assert without_harm.threshold_ppm < derived.threshold_ppm / 2


def test_ladder_orders_cheap_reversible_before_destructive():
    """PRD section 5: cheap reversible actions fire often, destructive ones rarely.

    Asserted as an ordering over derived thresholds rather than as a hand-written
    priority list. If a cost model ever made cancelling easier to justify than
    repairing, this fails -- which is the signal that the cost model is wrong, not
    that the ladder needs reordering.
    """
    cfg = make_tenant().config
    rungs = ledger.derive_ladder(cfg, ORDER_VALUE)

    assert [r.threshold_ppm for r in rungs] == sorted(r.threshold_ppm
                                                      for r in rungs)
    kinds = [r.action for r in rungs]
    assert kinds.index(ActionKind.ADDRESS_REPAIR) < kinds.index(ActionKind.CANCEL)
    assert kinds.index(ActionKind.CONTACT_VERIFY) < kinds.index(ActionKind.CANCEL)
    assert kinds[-1] is ActionKind.CANCEL, (
        "cancel must be the last rung: it is the only irreversible action")


def test_thresholds_differ_between_merchants():
    """Two merchants, two cost models, two different thresholds.

    The plan's M1 criterion. A single global threshold would be the tell that this
    is a heuristic wearing a formula's clothes.
    """
    a, b = two_tenants()
    ta = ledger.derive_cancel_threshold(a.config, ORDER_VALUE)
    tb = ledger.derive_cancel_threshold(b.config, ORDER_VALUE)

    assert ta.threshold_ppm != tb.threshold_ppm
    # Rival's margin is thinner and their freight heavier, so an RTO hurts more
    # relative to the margin at stake -- they should cancel sooner, not later.
    assert tb.threshold_ppm < ta.threshold_ppm


def test_threshold_infeasible_when_no_gain_is_possible():
    """A zero-margin, zero-loss order can never justify spending anything."""
    cfg = MerchantConfig(merchant_id="m", monthly_incentive_budget_paise=0,
                         gross_margin_bps=0, packaging_paise=0,
                         forward_freight_paise=0, failed_attempt_fee_paise=0,
                         return_freight_paise=0, goodwill_cost_paise=0)
    derived = ledger.derive_repair_threshold(
        cfg, 0, ActionKind.ADDRESS_REPAIR, effectiveness_ppm=550_000)
    assert not derived.feasible
    assert "no achievable gain" in derived.explanation


def test_render_shows_the_arithmetic():
    """PRD section 10: the arithmetic is rendered as arithmetic.

    A merchant who can check the sum by hand extends more trust than one shown a
    confidence score, so the rendering keeps every line and its reason.
    """
    cfg = make_tenant().config
    text = ledger.breakdown(cfg, ORDER_VALUE, OutcomeKind.RTO).render()

    assert "rto:" in text
    assert "packaging" in text and "return_freight" in text
    assert text.endswith("-₹195.00")


# ==========================================================================
# the stored ledger
# ==========================================================================

def _order(tenant, value_paise=ORDER_VALUE):
    buyer = BuyerRepo(tenant).upsert(phone="+91 98765 43210")
    return OrderRepo(tenant).create(buyer_id=buyer.id, value_paise=value_paise,
                                    address=GOOD_ADDRESS)


def _action_row(tenant, order_id, kind=ActionKind.ADDRESS_REPAIR):
    """Insert a minimal `actions` row so a ledger charge has something to cite.

    Written directly rather than through a repo because `ActionRepo` arrives with
    the gate in M4. The foreign key is the point: a cost must reference the action
    that incurred it, or the ledger cannot answer "what did we spend this on".
    """
    action_id = tenant.runtime.new_id("action")
    tenant.db.insert("actions", {
        "id": action_id, "merchant_id": tenant.merchant_id,
        "order_id": order_id, "kind": kind.value, "status": "pending",
        "idempotency_key": action_id, "created_at": tenant.runtime.stamp()})
    return action_id


def test_posting_an_outcome_sums_to_the_breakdown():
    """The stored balance equals the computed one, for all four outcomes.

    Two implementations of the same arithmetic is how a ROI report comes to
    disagree with the ledger it claims to summarise. The stored rows are the
    breakdown's lines, so there is only one implementation.
    """
    tenant = make_tenant()
    book = LedgerRepo(tenant)

    for outcome, expected in [(OutcomeKind.DELIVERED, to_paise("265")),
                              (OutcomeKind.RTO, -to_paise("195")),
                              (OutcomeKind.CANCELLED, 0),
                              (OutcomeKind.PREPAID_DELIVERED, to_paise("241"))]:
        order = _order(tenant)
        breakdown = ledger.breakdown(tenant.config, order.value_paise, outcome)
        book.post(breakdown, order_id=order.id)
        assert book.net(order_id=order.id) == expected, outcome


def test_posting_the_same_outcome_twice_is_refused():
    """A carrier redelivering a webhook must not double-count a Rs 195 loss."""
    tenant = make_tenant()
    book = LedgerRepo(tenant)
    order = _order(tenant)
    breakdown = ledger.breakdown(tenant.config, order.value_paise,
                                 OutcomeKind.RTO)

    book.post(breakdown, order_id=order.id)
    with pytest.raises(Conflict):
        book.post(breakdown, order_id=order.id)
    assert book.net(order_id=order.id) == -to_paise("195")


def test_cancelled_outcome_is_idempotent_despite_costing_nothing():
    """The zero-line case. Without the marker row this would post twice."""
    tenant = make_tenant()
    book = LedgerRepo(tenant)
    order = _order(tenant)
    breakdown = ledger.breakdown(tenant.config, order.value_paise,
                                 OutcomeKind.CANCELLED)

    book.post(breakdown, order_id=order.id)
    with pytest.raises(Conflict):
        book.post(breakdown, order_id=order.id)

    entries = book.entries(order_id=order.id)
    assert len(entries) == 1
    assert entries[0]["reason"] == LedgerReason.OUTCOME_POSTED.value
    assert entries[0]["amount_paise"] == 0


def test_ledger_rows_cannot_be_updated_or_deleted():
    """Append-only, enforced by the database rather than by convention.

    The triggers hold against the application, against a migration script, and
    against someone opening the file in a SQLite browser -- which is the only
    version of this guarantee worth asserting.
    """
    tenant = make_tenant()
    book = LedgerRepo(tenant)
    order = _order(tenant)
    book.post(ledger.breakdown(tenant.config, order.value_paise,
                               OutcomeKind.RTO), order_id=order.id)

    import sqlite3
    for statement in ("UPDATE ledger_entries SET amount_paise = 0",
                      "DELETE FROM ledger_entries"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            tenant.db.execute(statement)

    assert book.net(order_id=order.id) == -to_paise("195")


def test_a_mistake_is_fixed_by_a_compensating_entry():
    """The accountant's answer: correct by adding, never by editing."""
    tenant = make_tenant()
    book = LedgerRepo(tenant)
    order = _order(tenant)
    book.post(ledger.breakdown(tenant.config, order.value_paise,
                               OutcomeKind.RTO), order_id=order.id)

    book.correct(order_id=order.id, amount_paise=to_paise("195"),
                 note="carrier reversed the RTO: parcel was delivered on retry")
    assert book.net(order_id=order.id) == 0

    reasons = book.by_reason()
    assert reasons[LedgerReason.CORRECTION.value] == to_paise("195")
    # Both the error and its correction remain visible.
    assert reasons[LedgerReason.RETURN_FREIGHT.value] == -to_paise("70")


def test_action_cost_is_charged_on_attempt_not_on_success():
    """A message that got no reply still cost money.

    Charging only successful actions would flatter the agent's ROI, which is the
    single easiest way to make this product look better than it is.
    """
    tenant = make_tenant()
    book = LedgerRepo(tenant)
    order = _order(tenant)
    action_id = _action_row(tenant, order.id)

    book.post_action_cost(order_id=order.id, action_id=action_id,
                          cost_paise=300, note="address repair message sent")
    assert book.net(order_id=order.id) == -300

    with pytest.raises(Conflict):
        book.post_action_cost(order_id=order.id, action_id=action_id,
                              cost_paise=300)
    assert book.net(order_id=order.id) == -300


def test_action_settlement_allows_pending_to_terminal_once_and_audits_it():
    tenant = make_tenant()
    order = _order(tenant)
    action = ActionRepo(tenant).record(
        order.id, ActionKind.ADDRESS_REPAIR, status="pending")
    repo = ActionRepo(tenant)

    settled = repo.settle(action.id, "succeeded")
    assert settled.status == "succeeded"
    events = [e for e in tenant.audit.for_order(tenant.merchant_id, order.id)
              if e.kind == ACTION_SETTLED_AUDIT]
    assert len(events) == 1
    assert events[0].payload["from"] == "pending"
    assert events[0].payload["to"] == "succeeded"

    # A provider retry is an idempotent read, not another state change or audit row.
    assert repo.settle(action.id, "succeeded").status == "succeeded"
    assert len([e for e in tenant.audit.for_order(tenant.merchant_id, order.id)
                if e.kind == ACTION_SETTLED_AUDIT]) == 1


def test_action_settlement_rejects_non_pending_rewrites():
    tenant = make_tenant()
    order = _order(tenant)
    repo = ActionRepo(tenant)
    action = repo.record(order.id, ActionKind.ADDRESS_REPAIR, status="pending")
    with pytest.raises(StorageError, match="cannot settle"):
        repo.settle(action.id, "pending")
    repo.settle(action.id, "failed")

    with pytest.raises(StorageError, match="cannot settle"):
        repo.settle(action.id, "succeeded")
    with pytest.raises(StorageError, match="cannot settle"):
        repo.settle(action.id, "pending")


def test_action_settlement_transition_is_guarded_in_sql_too():
    tenant = make_tenant()
    order = _order(tenant)
    action = ActionRepo(tenant).record(
        order.id, ActionKind.ADDRESS_REPAIR, status="succeeded")

    with pytest.raises(sqlite3.IntegrityError, match="illegal settlement"):
        tenant.db.execute(
            "UPDATE actions SET status = 'failed' WHERE id = :id",
            {"id": action.id})


def test_arms_are_summed_separately():
    """The backtest's three arms share the table and never each other's totals."""
    tenant = make_tenant()
    book = LedgerRepo(tenant)
    order = _order(tenant)

    book.post(ledger.breakdown(tenant.config, order.value_paise,
                               OutcomeKind.RTO), order_id=order.id,
              arm="baseline")
    book.post(ledger.breakdown(tenant.config, order.value_paise,
                               OutcomeKind.CANCELLED), order_id=order.id,
              arm="agent")

    assert book.net(arm="baseline") == -to_paise("195")
    assert book.net(arm="agent") == 0
    assert book.net(arm="live") == 0
    assert book.arms() == ["agent", "baseline"]
