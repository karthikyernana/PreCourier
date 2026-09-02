"""The order lifecycle, and the promise that fulfilment always wins.

PRD section 1 makes one hard commitment: PreCourier may hold an order, but never
indefinitely, and never in a way that requires the agent to succeed. Everything
in this file is that commitment expressed as something a machine can check.

Two of the tests are graph properties over `domain.TRANSITIONS` rather than
scenarios. They matter more than the scenarios do: a scenario proves one path
releases, a graph property proves *no* path strands an order. A future milestone
that adds a state and forgets an edge fails here, at the definition, instead of
in production on a Tuesday afternoon with a warehouse waiting.
"""

from __future__ import annotations

import sqlite3

import pytest

from precourier.audit import AuditKind
from precourier.domain import (HOLDING_STATES, OPEN_STATES, TERMINAL_STATES,
                               TRANSITIONS, Address, IllegalTransition, Mode,
                               OrderState, can_transition)
from precourier.money import to_paise
from precourier.store.db import StorageError
from precourier.store.repo import BuyerRepo, Conflict, OrderRepo
from tests.fixtures import (DEFECTIVE_ADDRESS, GOOD_ADDRESS, make_tenant,
                            two_tenants)

VALUE = to_paise("1200")


def _reachable(start: OrderState) -> set:
    """States reachable from `start`, following `TRANSITIONS`."""
    seen, frontier = {start}, [start]
    while frontier:
        for nxt in TRANSITIONS.get(frontier.pop(), frozenset()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def _order(tenant, *, address=GOOD_ADDRESS, value_paise=VALUE, **kw):
    buyer_phone = kw.pop("phone", "+919876543210")
    buyer = BuyerRepo(tenant).upsert(phone=buyer_phone)
    return OrderRepo(tenant).create(buyer_id=buyer.id, value_paise=value_paise,
                                    address=address, **kw)


# ==========================================================================
# the lifecycle as a graph
# ==========================================================================

def test_every_state_is_reachable_from_received():
    """No orphan states. An unreachable state is dead code with a name."""
    unreachable = set(TRANSITIONS) - _reachable(OrderState.RECEIVED)
    assert not unreachable, sorted(s.value for s in unreachable)


def test_every_open_state_can_still_reach_release():
    """Fulfilment is sacred, stated as a graph property.

    For every state in which PreCourier still holds an opinion, there must exist a
    path to `released_for_dispatch`. This is what makes the auto-release timer
    implementable at all: the sweep can always move a stuck order forward, because
    the lifecycle guarantees a legal edge exists.

    Scoped to the open states on purpose. `shipped`, `delivered` and `rto` are
    already past release, and requiring them to reach it again would be asserting
    something false.
    """
    for state in OPEN_STATES:
        assert OrderState.RELEASED_FOR_DISPATCH in _reachable(state), (
            "an order in %s could never be released" % state.value)


def test_holding_states_can_be_released_in_one_hop():
    """The states the timeout sweep acts on must not need an intermediate step.

    A sweep that had to walk two edges to release could be interrupted between
    them, and the order would be left in a state nobody chose.
    """
    for state in HOLDING_STATES:
        assert can_transition(state, OrderState.RELEASED_FOR_DISPATCH), state


def test_terminal_states_are_exactly_the_three_endings():
    assert TERMINAL_STATES == {OrderState.CANCELLED, OrderState.DELIVERED,
                               OrderState.RTO}
    for state in TERMINAL_STATES:
        assert TRANSITIONS[state] == frozenset()


def test_no_state_transitions_to_itself():
    """Self-edges would make the idempotent-repeat path indistinguishable from a
    real change, and every state change writes an audit row."""
    for state, nxt in TRANSITIONS.items():
        assert state not in nxt, state


# ==========================================================================
# transitions
# ==========================================================================

def test_a_legal_transition_writes_a_row_and_an_audit_entry():
    tenant = make_tenant()
    order = _order(tenant)
    repo = OrderRepo(tenant)

    moved = repo.transition(order.id, OrderState.SCORING, reason="intake done")
    assert moved.state is OrderState.SCORING

    rows = tenant.db.all(
        "SELECT * FROM order_state_transitions WHERE merchant_id = :merchant_id "
        "AND order_id = :o ORDER BY created_at", tenant.scope(o=order.id))
    assert [(r["from_state"], r["to_state"]) for r in rows] == [
        ("received", "scoring")]
    assert rows[0]["reason"] == "intake done"

    kinds = [e.kind for e in tenant.audit.for_order(tenant.merchant_id, order.id)]
    assert kinds == [AuditKind.ORDER_RECEIVED, AuditKind.ORDER_STATE_CHANGED]


def test_an_illegal_transition_is_refused_and_says_what_was_legal():
    """Raised, not coerced.

    An error message that lists the legal next states turns a production incident
    into a two-minute fix. Silently coercing the transition would instead produce
    an audit trail that disagrees with what happened -- and the audit trail is the
    product.
    """
    tenant = make_tenant()
    order = _order(tenant)

    with pytest.raises(IllegalTransition) as caught:
        OrderRepo(tenant).transition(order.id, OrderState.DELIVERED)

    message = str(caught.value)
    assert "received" in message and "delivered" in message
    assert "scoring" in message, "the error must list the legal next states"
    assert OrderRepo(tenant).get(order.id).state is OrderState.RECEIVED


def test_a_terminal_order_refuses_every_further_transition():
    tenant = make_tenant()
    order = _order(tenant)
    repo = OrderRepo(tenant)
    repo.transition(order.id, OrderState.CANCELLED, reason="buyer asked")

    for target in (OrderState.RELEASED_FOR_DISPATCH, OrderState.SCORING,
                   OrderState.DELIVERED):
        with pytest.raises(IllegalTransition, match="terminal"):
            repo.transition(order.id, target)


def test_repeating_a_transition_is_idempotent():
    """A carrier or storefront redelivering an event is not an error.

    Idempotent rather than refused, because the alternative is that a duplicate
    webhook produces a 500 and the sender retries forever.
    """
    tenant = make_tenant()
    order = _order(tenant)
    repo = OrderRepo(tenant)

    repo.transition(order.id, OrderState.SCORING)
    repo.transition(order.id, OrderState.SCORING)

    count = tenant.db.scalar(
        "SELECT COUNT(*) FROM order_state_transitions "
        "WHERE merchant_id = :merchant_id AND order_id = :o", tenant.scope(o=order.id))
    assert count == 1, "a repeated event must not write a second history row"


def test_transition_history_cannot_be_rewritten():
    tenant = make_tenant()
    order = _order(tenant)
    OrderRepo(tenant).transition(order.id, OrderState.SCORING)

    for statement in ("UPDATE order_state_transitions SET to_state = 'delivered'",
                      "DELETE FROM order_state_transitions"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            tenant.db.execute(statement)


# ==========================================================================
# holds and the auto-release guarantee
# ==========================================================================

def test_a_hold_is_capped_at_the_merchant_s_tolerance():
    """Asking for longer than `max_hold_minutes` gets `max_hold_minutes`.

    Capped at the write rather than validated at the caller, so no code path --
    present or future, agent or human -- can extend a hold past what the merchant
    agreed to. A merchant's latency promise to their buyers is not ours to spend.
    """
    tenant = make_tenant(max_hold_minutes=240)
    order = _order(tenant)
    repo = OrderRepo(tenant)

    capped = repo.hold(order.id, minutes=10_000, reason="agent asked for a week")
    expected = tenant.runtime.clock.plus_minutes(240)
    assert capped.held_until == expected

    entry = tenant.audit.recent(tenant.merchant_id,
                                kind=AuditKind.ORDER_HELD)[0]
    assert entry.payload["minutes"] == 240
    assert entry.payload["cap_minutes"] == 240


def test_a_shorter_hold_is_honoured_as_asked():
    tenant = make_tenant(max_hold_minutes=240)
    order = _order(tenant)
    held = OrderRepo(tenant).hold(order.id, minutes=30)
    assert held.held_until == tenant.runtime.clock.plus_minutes(30)


def test_expired_holds_are_exactly_the_orders_the_sweep_must_release():
    """The auto-release query, tested against the three cases it must separate.

    One order past its deadline, one still inside it, one already released. Only
    the first is the sweep's business; returning the third would re-release an
    order and emit a duplicate contract event.
    """
    tenant = make_tenant(mode=Mode.GATEWAY)
    repo = OrderRepo(tenant)
    clock = tenant.runtime.clock

    late = _order(tenant, phone="+919876500001")
    early = _order(tenant, phone="+919876500002")
    done = _order(tenant, phone="+919876500003")

    repo.hold(late.id, minutes=30)
    repo.hold(early.id, minutes=200)
    repo.hold(done.id, minutes=30)
    repo.release(done.id, reason="buyer confirmed")

    clock.advance(minutes=60)
    expired = repo.expired_holds(tenant.runtime.stamp())

    assert [o.id for o in expired] == [late.id]


def test_the_sweep_releases_and_records_it_as_an_auto_release():
    """A timed-out order is released, and the log says who released it.

    Distinguished from an agent release because they mean different things when a
    merchant reads them back: one is a decision, the other is the absence of one.
    """
    tenant = make_tenant(mode=Mode.GATEWAY)
    repo = OrderRepo(tenant)
    order = _order(tenant)
    repo.transition(order.id, OrderState.SCORING)
    repo.transition(order.id, OrderState.DIAGNOSING)
    repo.transition(order.id, OrderState.DECIDING)
    repo.transition(order.id, OrderState.ACTING)
    repo.transition(order.id, OrderState.AWAITING_BUYER, reason="asked to confirm")
    repo.hold(order.id, minutes=30)

    tenant.runtime.clock.advance(minutes=45)
    for stuck in repo.expired_holds(tenant.runtime.stamp()):
        repo.transition(stuck.id, OrderState.RELEASED_FOR_DISPATCH,
                        reason="hold expired: fulfilment wins", actor="timeout")

    released = repo.get(order.id)
    assert released.state is OrderState.RELEASED_FOR_DISPATCH
    assert released.held_until is None, "a released order is no longer held"
    assert repo.expired_holds(tenant.runtime.stamp()) == []

    kinds = [e.kind for e in tenant.audit.for_order(tenant.merchant_id, order.id)]
    assert AuditKind.ORDER_AUTO_RELEASED in kinds
    assert AuditKind.ORDER_RELEASED not in kinds


def test_an_agent_release_and_a_timeout_release_are_logged_differently():
    tenant = make_tenant(mode=Mode.GATEWAY)
    order = _order(tenant)
    OrderRepo(tenant).release(order.id, reason="risk below threshold")

    kinds = [e.kind for e in tenant.audit.for_order(tenant.merchant_id, order.id)]
    assert AuditKind.ORDER_RELEASED in kinds
    assert AuditKind.ORDER_AUTO_RELEASED not in kinds


def test_release_is_idempotent_and_never_refused():
    """The contract event. Called twice, it must not raise.

    A release that could fail would put the caller in the position of deciding
    whether to print the label anyway, which is exactly the decision this system
    exists to take off their plate.
    """
    tenant = make_tenant()
    order = _order(tenant)
    repo = OrderRepo(tenant)

    first = repo.release(order.id, reason="no risk")
    again = repo.release(order.id, reason="duplicate call")
    assert first.state is again.state is OrderState.RELEASED_FOR_DISPATCH

    count = tenant.db.scalar(
        "SELECT COUNT(*) FROM order_state_transitions "
        "WHERE merchant_id = :merchant_id AND order_id = :o AND to_state = :s",
        tenant.scope(o=order.id, s=OrderState.RELEASED_FOR_DISPATCH.value))
    assert count == 1


def test_holding_an_already_released_order_does_nothing():
    """Once the label may be printed, no later hold can take that back."""
    tenant = make_tenant()
    order = _order(tenant)
    repo = OrderRepo(tenant)
    repo.release(order.id, reason="no risk")

    repo.hold(order.id, minutes=60)
    assert repo.get(order.id).held_until is None


def test_negative_and_terminal_holds_are_refused():
    tenant = make_tenant()
    repo = OrderRepo(tenant)
    order = _order(tenant)

    with pytest.raises(StorageError, match="negative"):
        repo.hold(order.id, minutes=-1)

    repo.transition(order.id, OrderState.CANCELLED, reason="buyer asked")
    with pytest.raises(StorageError, match="terminal"):
        repo.hold(order.id, minutes=1)


def test_terminal_hold_is_rejected_by_the_database_too():
    tenant = make_tenant()
    order = _order(tenant)
    with pytest.raises(sqlite3.IntegrityError, match="terminal"):
        tenant.db.execute(
            "UPDATE orders SET state = 'cancelled', held_until = :until "
            "WHERE id = :id",
            {"id": order.id, "until": "2026-03-01T10:00:00Z"})


def test_a_release_records_when_it_happened():
    tenant = make_tenant()
    order = _order(tenant)
    at = tenant.runtime.stamp()
    OrderRepo(tenant).release(order.id, reason="no risk")

    stored = tenant.db.one(
        "SELECT released_at FROM orders WHERE merchant_id = :merchant_id AND id = :o",
        tenant.scope(o=order.id))
    assert stored["released_at"] == at


# ==========================================================================
# addresses: repaired by appending
# ==========================================================================

def test_intake_stores_revision_one_and_the_order_points_at_it():
    tenant = make_tenant()
    order = _order(tenant, address=DEFECTIVE_ADDRESS)
    repo = OrderRepo(tenant)

    current = repo.current_revision(order.id)
    assert current.revision == 1
    assert current.source == "submitted"
    assert current.address == DEFECTIVE_ADDRESS
    assert repo.address(order.id) == DEFECTIVE_ADDRESS


def test_a_repair_appends_a_revision_instead_of_overwriting():
    """Before and after, both kept, for free.

    The submitted address stays at revision 1 forever. That is what makes a repair
    reviewable rather than merely reported: the merchant sees what the buyer
    originally typed next to what we changed it to, and can disagree.
    """
    tenant = make_tenant()
    order = _order(tenant, address=DEFECTIVE_ADDRESS)
    repo = OrderRepo(tenant)

    fixed = Address(line1="Flat 402, Ashirvad Residency", line2="12th Main Road",
                    locality="Koramangala 4th Block", city="Bengaluru",
                    state="Karnataka", pincode="560034")
    revision = repo.patch_address(order.id, fixed, source="buyer_confirmed",
                                  evidence={"reply_id": "msg_1"})

    assert revision.revision == 2
    assert repo.address(order.id) == fixed

    history = repo.address_history(order.id)
    assert [r.revision for r in history] == [1, 2]
    assert history[0].address == DEFECTIVE_ADDRESS, (
        "revision 1 must survive the repair verbatim")

    entry = tenant.audit.recent(tenant.merchant_id,
                                kind=AuditKind.ADDRESS_PATCHED)[0]
    assert entry.payload["from_revision"] == 1
    assert entry.payload["to_revision"] == 2
    assert entry.payload["before"]["pincode"] == DEFECTIVE_ADDRESS.pincode
    assert entry.payload["after"]["pincode"] == "560034"
    assert entry.payload["evidence"] == {"reply_id": "msg_1"}


def test_address_revisions_cannot_be_edited_or_deleted():
    tenant = make_tenant()
    order = _order(tenant)
    OrderRepo(tenant).patch_address(order.id, GOOD_ADDRESS)

    for statement in ("UPDATE order_addresses SET pincode = '000000'",
                      "DELETE FROM order_addresses"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            tenant.db.execute(statement)


# ==========================================================================
# intake
# ==========================================================================

def test_the_commercial_facts_of_an_order_are_immutable():
    """Value and buyer are fixed at intake, enforced by the database.

    A decision explains itself in terms of the order's value. If the value could
    be edited afterwards, every stored decision would become unfalsifiable -- the
    arithmetic would no longer reproduce, and there would be no way to tell
    whether the gate was wrong or the row was.
    """
    tenant = make_tenant()
    order = _order(tenant)

    for column, value in (("value_paise", 1), ("buyer_id", "byr_someone"),
                          ("payment_method", "prepaid"), ("item_count", 99),
                          ("category", "supplements"), ("discount_bps", 500),
                          ("is_first_time_buyer", 0),
                          ("placed_at", "2020-01-01T00:00:00Z")):
        with pytest.raises(sqlite3.IntegrityError, match="only state, mode"):
            tenant.db.execute(
                "UPDATE orders SET %s = :v WHERE id = :o" % column,
                {"v": value, "o": order.id})

    assert OrderRepo(tenant).get(order.id).value_paise == VALUE


def test_an_order_reference_cannot_be_added_after_the_fact():
    """The nullable column in the immutability list, which needs `IS NOT`.

    `NULL <> 'x'` is NULL, not true, so a `<>` comparison would have left this one
    column silently unguarded -- and it is the column a merchant would use to
    re-point our record at a different order in their own system.
    """
    tenant = make_tenant()
    order = _order(tenant)
    assert order.external_id == ""

    with pytest.raises(sqlite3.IntegrityError, match="only state, mode"):
        tenant.db.execute(
            "UPDATE orders SET external_id = 'SHOP-9999' WHERE id = :o",
            {"o": order.id})


def test_orders_are_never_deleted():
    tenant = make_tenant()
    _order(tenant)
    with pytest.raises(sqlite3.IntegrityError, match="never deleted"):
        tenant.db.execute("DELETE FROM orders")


def test_many_orders_can_have_no_external_reference():
    """Absent is not a value.

    Stored as NULL rather than '', because two orders that carry no reference from
    the merchant's system are not duplicates of each other -- and a `''` default
    under the unique index would have let exactly one such order exist per
    merchant. Most storefronts send no reference at all.
    """
    tenant = make_tenant()
    ids = {_order(tenant).id for _ in range(4)}
    assert len(ids) == 4

    stored = tenant.db.all(
        "SELECT external_id FROM orders WHERE merchant_id = :merchant_id", tenant.scope())
    assert all(row["external_id"] is None for row in stored)


def test_a_duplicate_external_reference_is_refused():
    """Present, it must be unique: it is the merchant's idempotency handle.

    Two orders under one reference would mean the merchant's retry created a
    second parcel.
    """
    tenant = make_tenant()
    _order(tenant, external_id="SHOP-1001")
    with pytest.raises(Conflict, match="SHOP-1001"):
        _order(tenant, external_id="SHOP-1001")

    found = OrderRepo(tenant).by_external_id("SHOP-1001")
    assert found is not None and found.external_id == "SHOP-1001"
    assert OrderRepo(tenant).by_external_id("") is None
    assert OrderRepo(tenant).by_external_id("SHOP-NOPE") is None


def test_the_same_external_reference_is_free_for_a_different_merchant():
    """Uniqueness is per tenant. Two shops both number their orders from 1."""
    a, b = two_tenants()
    _order(a, external_id="1001")
    _order(b, external_id="1001")

    assert OrderRepo(a).by_external_id("1001").merchant_id == a.merchant_id
    assert OrderRepo(b).by_external_id("1001").merchant_id == b.merchant_id


def test_first_time_buyer_is_inferred_from_history_not_taken_on_trust():
    """A feature the storefront could get wrong, so it is derived here.

    `is_first_time_buyer` feeds the risk model. Trusting the caller's flag would
    let a misconfigured integration mark every order as a first purchase, and the
    model would quietly learn from a constant.
    """
    tenant = make_tenant()
    first = _order(tenant, phone="+919876500009")
    second = _order(tenant, phone="+919876500009")

    assert first.is_first_time_buyer
    assert not second.is_first_time_buyer
    assert BuyerRepo(tenant).get(second.buyer_id).prior_orders == 2


def test_an_order_may_override_the_merchant_s_mode():
    """How a canary runs: a slice of traffic in gateway while the rest is advisory."""
    tenant = make_tenant(mode=Mode.ADVISORY)
    default = _order(tenant)
    canary = _order(tenant, mode=Mode.GATEWAY)

    assert default.mode is Mode.ADVISORY
    assert canary.mode is Mode.GATEWAY


def test_a_negative_order_value_is_refused():
    tenant = make_tenant()
    with pytest.raises(Exception):
        _order(tenant, value_paise=-1)


def test_orders_can_be_listed_by_state_and_counted():
    tenant = make_tenant()
    repo = OrderRepo(tenant)
    a = _order(tenant, phone="+919876500011")
    b = _order(tenant, phone="+919876500012")
    repo.transition(a.id, OrderState.SCORING)

    assert [o.id for o in repo.in_states([OrderState.SCORING])] == [a.id]
    assert [o.id for o in repo.in_states([OrderState.RECEIVED])] == [b.id]
    assert repo.count_by_state() == {"scoring": 1, "received": 1}
    assert repo.in_states([]) == []
