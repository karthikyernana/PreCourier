"""The audit chain: can a row be changed after the fact without it showing?

Two layers, and the tests are deliberately split along them.

The triggers stop the application -- and anything else speaking SQL -- from
updating or deleting an audit row at all. That is the layer that actually holds
in production.

The hash chain is what holds when the triggers do not: someone with the database
file, a `DROP TRIGGER`, and a motive. So the tamper tests here drop the triggers
first. That is not working around the protection, it is modelling the threat the
chain exists for -- a chain that was only ever tested against an attacker who
politely leaves the triggers in place proves nothing.
"""

from __future__ import annotations

import sqlite3

import pytest

from precourier import audit as audit_mod
from precourier import canonical
from precourier.audit import GENESIS, AuditKind, AuditLog, compute_hash, envelope
from precourier.domain import OrderState
from precourier.money import to_paise
from precourier.store.repo import BuyerRepo, OrderRepo
from tests.fixtures import (GOOD_ADDRESS, deterministic_runtime, fresh_db,
                            make_tenant, two_tenants)


def _unlock(db) -> None:
    """Remove the append-only triggers, so a tamper can actually be attempted.

    Every caller of this is simulating an attacker who has the file. The
    `test_triggers_...` tests below assert that this step is necessary.
    """
    db.raw.executescript("DROP TRIGGER IF EXISTS audit_log_no_update;"
                         "DROP TRIGGER IF EXISTS audit_log_no_delete;")


def _some_history(tenant, count: int = 5):
    """A handful of realistic entries, written the way the system writes them."""
    buyer = BuyerRepo(tenant).upsert(phone="+919876543210")
    orders = []
    for _ in range(count):
        order = OrderRepo(tenant).create(buyer_id=buyer.id,
                                        value_paise=to_paise("1200"),
                                        address=GOOD_ADDRESS)
        OrderRepo(tenant).transition(order.id, OrderState.SCORING,
                                     reason="intake complete")
        orders.append(order)
    return orders


# ==========================================================================
# the happy path
# ==========================================================================

def test_a_freshly_written_chain_verifies():
    tenant = make_tenant()
    _some_history(tenant)

    result = tenant.audit.verify(tenant.merchant_id)
    assert result.ok, result.summary()
    assert result.entries_checked >= 10          # order + transition, five times
    assert result.first_bad_seq is None


def test_the_first_entry_links_to_genesis():
    """A chain has to start somewhere, and where it starts must be fixed.

    If the first entry's `prev_hash` were the empty string or NULL, an attacker
    could delete the first N entries and re-point entry N+1 at genesis.
    Verification checks position as well as linkage, so that fails -- but only
    because genesis is a known constant rather than "whatever the first row says".

    Written against a bare `AuditLog` rather than a provisioned tenant, because
    provisioning already writes `merchant.created` as entry 1.
    """
    db, runtime = fresh_db(), deterministic_runtime()
    log = AuditLog(db, runtime)
    db.insert("merchants", {"id": "mch_bare", "name": "Bare",
                            "status": "active", "created_at": runtime.stamp()})

    first = log.append("mch_bare", AuditKind.MERCHANT_CREATED, {"note": "first"})
    assert first.prev_hash == GENESIS
    assert first.merchant_seq == 1
    assert log.verify("mch_bare").ok


def test_head_is_the_last_hash_and_an_empty_chain_heads_at_genesis():
    db, runtime = fresh_db(), deterministic_runtime()
    log = AuditLog(db, runtime)
    assert log.head("nobody") == GENESIS

    tenant = make_tenant(db, runtime)
    last = tenant.audit.recent(tenant.merchant_id, limit=1)[0]
    assert tenant.audit.head(tenant.merchant_id) == last.hash


def test_an_empty_chain_verifies_rather_than_erroring():
    """A merchant who has done nothing has an intact log of nothing."""
    db, runtime = fresh_db(), deterministic_runtime()
    result = AuditLog(db, runtime).verify("nobody")
    assert result.ok and result.entries_checked == 0


def test_a_float_in_a_payload_is_refused_before_the_write():
    """Money is integer paise everywhere. A float here means precision was lost.

    Refused rather than coerced, and refused *before* the insert, so a rejected
    payload cannot leave a gap in the sequence.
    """
    tenant = make_tenant()
    before = tenant.audit.head(tenant.merchant_id)

    with pytest.raises(canonical.NotCanonical):
        tenant.audit.append(tenant.merchant_id, AuditKind.RISK_SCORED,
                            {"p_rto": 0.42})

    assert tenant.audit.head(tenant.merchant_id) == before
    assert tenant.audit.verify(tenant.merchant_id).ok


# ==========================================================================
# per-merchant chains
# ==========================================================================

def test_each_merchant_has_their_own_contiguous_chain():
    """Interleaved writes, two independent chains.

    Per merchant rather than global, for a reason that only shows up at export
    time: a merchant must be able to hand their whole log to an auditor and have
    it verify standalone. A global chain would make every merchant's proof depend
    on rows they are not allowed to see.
    """
    a, b = two_tenants()
    for i in range(3):
        a.audit.append(a.merchant_id, AuditKind.ORDER_RECEIVED, {"i": i})
        b.audit.append(b.merchant_id, AuditKind.ORDER_RECEIVED, {"i": i})

    for tenant in (a, b):
        result = tenant.audit.verify(tenant.merchant_id)
        assert result.ok, result.summary()
        seqs = [r.merchant_seq for r in
                reversed(tenant.audit.recent(tenant.merchant_id, limit=100))]
        assert seqs == list(range(1, len(seqs) + 1)), (
            "merchant_seq must be gapless per merchant, whatever the global order")

    # The global sequence interleaves; the per-merchant one does not.
    globals_ = [r["seq"] for r in a.db.all(
        "SELECT seq, merchant_id FROM audit_log ORDER BY seq")]
    assert globals_ == sorted(globals_)


def test_verifying_one_merchant_ignores_the_other_s_tampering():
    a, b = two_tenants()
    a.audit.append(a.merchant_id, AuditKind.ORDER_RECEIVED, {"i": 1})
    b.audit.append(b.merchant_id, AuditKind.ORDER_RECEIVED, {"i": 1})

    _unlock(a.db)
    a.db.raw.execute("UPDATE audit_log SET payload_json = :p "
                     "WHERE merchant_id = :m",
                     {"p": '{"i":99}', "m": b.merchant_id})

    assert not b.audit.verify(b.merchant_id).ok
    assert a.audit.verify(a.merchant_id).ok, (
        "one tenant's corruption must not invalidate another's proof")


# ==========================================================================
# tamper detection
# ==========================================================================

def test_triggers_block_an_update_and_a_delete_outright():
    """The layer that holds in production: no SQL path can rewrite history."""
    tenant = make_tenant()
    _some_history(tenant, count=1)

    for statement in ("UPDATE audit_log SET payload_json = '{}'",
                      "DELETE FROM audit_log"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            tenant.db.execute(statement)
    assert tenant.audit.verify(tenant.merchant_id).ok


def test_an_edited_payload_is_detected():
    """The obvious attack: change what the record says happened."""
    tenant = make_tenant()
    _some_history(tenant, count=3)
    _unlock(tenant.db)

    target = tenant.audit.recent(tenant.merchant_id, limit=100)[-2]
    tenant.db.raw.execute(
        "UPDATE audit_log SET payload_json = :p WHERE seq = :s",
        {"p": canonical.dumps({"order_id": "ord_forged", "value_paise": 1}),
         "s": target.seq})

    result = tenant.audit.verify(tenant.merchant_id)
    assert not result.ok
    assert result.first_bad_seq == target.merchant_seq
    assert "contents" in result.detail


def test_an_edited_timestamp_is_detected():
    """The attack an envelope over the payload alone would have missed.

    Backdating an entry is how a decision made after the fact is made to look
    contemporaneous, which is precisely the claim an audit log exists to support.
    The envelope covers the whole row -- actor, kind, order_id, created_at,
    merchant_id -- not just the payload.
    """
    tenant = make_tenant()
    _some_history(tenant, count=2)
    _unlock(tenant.db)

    target = tenant.audit.recent(tenant.merchant_id, limit=100)[-1]
    tenant.db.raw.execute(
        "UPDATE audit_log SET created_at = :t WHERE seq = :s",
        {"t": "2020-01-01T00:00:00Z", "s": target.seq})

    assert not tenant.audit.verify(tenant.merchant_id).ok


def test_re_attributing_an_entry_to_another_merchant_is_detected():
    """Moving a row between tenants breaks both chains, by construction.

    `merchant_id` is inside the hashed envelope, so the row does not verify in its
    new home; and it is missing from its old one, so that chain now has a gap.

    A *middle* entry, deliberately. Re-attributing the tail is a truncation of the
    source chain, and truncation is the one case `verify` cannot see on its own --
    see `test_truncating_the_tail_...`. Testing the easy case only would have
    hidden which of the two mechanisms is actually doing the work.
    """
    a, b = two_tenants()
    for n in range(3):
        a.audit.append(a.merchant_id, AuditKind.ACTION_ATTEMPTED, {"n": n})
    b.audit.append(b.merchant_id, AuditKind.ACTION_ATTEMPTED, {"n": 1})

    chain = a.audit.recent(a.merchant_id, limit=100)[::-1]
    victim = chain[len(chain) // 2]
    assert victim.merchant_seq < chain[-1].merchant_seq, "must not be the tail"

    _unlock(a.db)
    a.db.raw.execute("UPDATE audit_log SET merchant_id = :m WHERE seq = :s",
                     {"m": b.merchant_id, "s": victim.seq})

    source = a.audit.verify(a.merchant_id)
    assert not source.ok, "the source chain lost an entry"
    assert "removed" in source.detail
    assert not b.audit.verify(b.merchant_id).ok, "the destination hash is wrong"


def test_a_deleted_middle_entry_is_detected():
    """Deletion shows up as a sequence gap, not as a hash mismatch.

    Worth asserting separately: an attacker who deletes a row and does not touch
    anything else leaves every remaining hash internally valid. Only the numbering
    gives it away, which is why `merchant_seq` is checked against position.
    """
    tenant = make_tenant()
    _some_history(tenant, count=3)
    records = tenant.audit.recent(tenant.merchant_id, limit=100)[::-1]
    victim = records[len(records) // 2]

    _unlock(tenant.db)
    tenant.db.raw.execute("DELETE FROM audit_log WHERE seq = :s",
                          {"s": victim.seq})

    result = tenant.audit.verify(tenant.merchant_id)
    assert not result.ok
    assert "removed" in result.detail
    assert result.first_bad_seq == victim.merchant_seq + 1


def test_swapping_two_entries_is_detected():
    """Re-ordering changes what each entry's predecessor was."""
    tenant = make_tenant()
    _some_history(tenant, count=3)
    records = tenant.audit.recent(tenant.merchant_id, limit=100)[::-1]
    first, second = records[1], records[2]

    _unlock(tenant.db)
    tenant.db.raw.execute(
        "UPDATE audit_log SET merchant_seq = :tmp WHERE seq = :s",
        {"tmp": 10_000, "s": first.seq})
    tenant.db.raw.execute(
        "UPDATE audit_log SET merchant_seq = :n WHERE seq = :s",
        {"n": first.merchant_seq, "s": second.seq})
    tenant.db.raw.execute(
        "UPDATE audit_log SET merchant_seq = :n WHERE seq = :s",
        {"n": second.merchant_seq, "s": first.seq})

    assert not tenant.audit.verify(tenant.merchant_id).ok


def test_a_wholesale_forgery_of_the_tail_is_detected():
    """Re-hashing an edited entry is not enough: the successor still disagrees.

    This is the attacker who knows how the chain works and recomputes the hash of
    the row they edited. It fails because every later entry committed to that
    hash, so the forgery has to be carried forward to the head -- and the head is
    published.
    """
    tenant = make_tenant()
    _some_history(tenant, count=3)
    records = tenant.audit.recent(tenant.merchant_id, limit=100)[::-1]
    target = records[1]
    published_head = tenant.audit.head(tenant.merchant_id)

    _unlock(tenant.db)
    forged_payload = {"order_id": "ord_forged"}
    env = envelope(merchant_id=target.merchant_id,
                   merchant_seq=target.merchant_seq, kind=target.kind,
                   actor=target.actor, order_id=target.order_id,
                   created_at=target.created_at, payload=forged_payload)
    tenant.db.raw.execute(
        "UPDATE audit_log SET payload_json = :p, hash = :h WHERE seq = :s",
        {"p": canonical.dumps(forged_payload),
         "h": compute_hash(target.prev_hash, env), "s": target.seq})

    result = tenant.audit.verify(tenant.merchant_id)
    assert not result.ok
    assert result.first_bad_seq == target.merchant_seq + 1
    assert "prev_hash" in result.detail
    assert tenant.audit.head(tenant.merchant_id) == published_head


def test_truncating_the_tail_is_not_self_detectable_but_the_head_catches_it():
    """The one attack the chain cannot catch alone, asserted rather than glossed.

    A shorter valid chain is still a valid chain, so `verify` returns ok. That is
    not a bug to be papered over -- it is the reason the head hash is published in
    the console and included in every export. The test asserts both halves so that
    nobody later mistakes the ok for a stronger guarantee than it is.
    """
    tenant = make_tenant()
    _some_history(tenant, count=3)
    published_head = tenant.audit.head(tenant.merchant_id)
    published_length = tenant.audit.verify(tenant.merchant_id).entries_checked

    _unlock(tenant.db)
    tenant.db.raw.execute(
        "DELETE FROM audit_log WHERE merchant_id = :m AND merchant_seq > :n",
        {"m": tenant.merchant_id, "n": published_length - 2})

    truncated = tenant.audit.verify(tenant.merchant_id)
    assert truncated.ok, "a truncated chain is internally consistent"
    assert truncated.entries_checked < published_length
    # And this is what makes it visible anyway.
    assert tenant.audit.head(tenant.merchant_id) != published_head


def test_every_envelope_field_is_covered_by_the_hash():
    """Structural: change any one field, get a different hash.

    Written as a loop over the envelope's own keys so that a field added to the
    envelope later is covered without anyone remembering to extend this test. A
    field present in the row but absent from the hash is editable without
    detection, and that is a silent hole rather than a loud one.
    """
    base = envelope(merchant_id="mch_1", merchant_seq=7, kind="order.received",
                    actor="api", order_id="ord_1",
                    created_at="2026-03-01T09:00:00Z",
                    payload={"value_paise": 120000})
    baseline = compute_hash(GENESIS, base)

    for key in base:
        mutated = dict(base)
        if key == "merchant_seq":
            mutated[key] = base[key] + 1
        elif key == "payload":
            mutated[key] = {"value_paise": 120001}
        else:
            mutated[key] = str(base[key]) + "_x"
        assert compute_hash(GENESIS, mutated) != baseline, (
            "envelope field %r does not affect the hash" % key)

    # And the link itself is covered: same envelope, different predecessor.
    assert compute_hash("1" * 64, base) != baseline


# ==========================================================================
# what the log is for
# ==========================================================================

def test_an_order_s_timeline_is_the_audit_trail():
    """The console's order page reads this, not a separately maintained history."""
    tenant = make_tenant()
    order = _some_history(tenant, count=1)[0]

    timeline = tenant.audit.for_order(tenant.merchant_id, order.id)
    kinds = [r.kind for r in timeline]
    assert kinds[0] == AuditKind.ORDER_RECEIVED
    assert AuditKind.ORDER_STATE_CHANGED in kinds
    assert all(r.order_id == order.id for r in timeline)
    assert [r.merchant_seq for r in timeline] == sorted(
        r.merchant_seq for r in timeline)


def test_intake_audit_carries_the_commercial_facts_and_no_phone_number():
    """PRD section 14: audit rows are exported, so PII must not be in them.

    The pincode and the value are in the payload because a merchant reviewing a
    decision needs them. The buyer's phone number is not, because an export lands
    in a spreadsheet and a spreadsheet lands in an inbox.
    """
    tenant = make_tenant()
    phone = "9876543210"
    buyer = BuyerRepo(tenant).upsert(phone=phone)
    order = OrderRepo(tenant).create(buyer_id=buyer.id,
                                     value_paise=to_paise("1200"),
                                     address=GOOD_ADDRESS)

    entry = tenant.audit.for_order(tenant.merchant_id, order.id)[0]
    assert entry.payload["value_paise"] == to_paise("1200")
    assert entry.payload["pincode"] == GOOD_ADDRESS.pincode
    serialised = canonical.dumps(tenant.audit.export(tenant.merchant_id))
    assert phone not in serialised
    assert GOOD_ADDRESS.line1 not in serialised, (
        "a street address is PII too, and revision 1 is already stored in "
        "order_addresses where the console can show it under access control")


def test_an_export_is_verifiable_without_this_codebase():
    """Everything a third party needs: algorithm, genesis, head, entries."""
    tenant = make_tenant()
    _some_history(tenant, count=2)
    export = tenant.audit.export(tenant.merchant_id)

    assert export["genesis"] == GENESIS
    assert export["head"] == tenant.audit.head(tenant.merchant_id)
    assert "sha256" in export["algorithm"]
    assert canonical.dumps(export)          # the export itself is float-free

    # Re-walk it here the way an outside auditor would, from the export alone.
    prev = export["genesis"]
    for entry in export["entries"]:
        env = envelope(merchant_id=entry["merchant_id"],
                       merchant_seq=entry["merchant_seq"], kind=entry["kind"],
                       actor=entry["actor"], order_id=entry["order_id"],
                       created_at=entry["created_at"], payload=entry["payload"])
        assert compute_hash(prev, env) == entry["hash"]
        prev = entry["hash"]
    assert prev == export["head"]


def test_recent_can_be_filtered_by_kind():
    tenant = make_tenant()
    _some_history(tenant, count=3)
    received = tenant.audit.recent(tenant.merchant_id,
                                   kind=AuditKind.ORDER_RECEIVED)
    assert len(received) == 3
    assert {r.kind for r in received} == {AuditKind.ORDER_RECEIVED}


def test_audit_kinds_are_plain_strings_so_a_retired_kind_still_reads():
    """Deliberately not an enum.

    The log outlives the code. If a kind is removed in a later version, an entry
    written under it must still render as the words it was written with, rather
    than raising on the way out of the database.
    """
    tenant = make_tenant()
    tenant.audit.append(tenant.merchant_id, "some.retired.kind", {"n": 1})
    record = tenant.audit.recent(tenant.merchant_id, limit=1)[0]
    assert record.kind == "some.retired.kind"
    assert tenant.audit.verify(tenant.merchant_id).ok
    assert isinstance(AuditKind.ORDER_RECEIVED, str)
    assert not hasattr(audit_mod.AuditKind, "__members__")


def test_an_audit_row_and_the_change_it_describes_commit_together():
    """A state change without its audit entry would be worse than a failed request.

    The write is inside the same transaction as the change, so a failure part-way
    leaves neither. Forced here by making the audit append fail on a float.
    """
    tenant = make_tenant()
    order = _some_history(tenant, count=1)[0]
    state_before = OrderRepo(tenant).get(order.id).state
    head_before = tenant.audit.head(tenant.merchant_id)

    with pytest.raises(canonical.NotCanonical):
        with tenant.db.tx():
            OrderRepo(tenant).transition(order.id, OrderState.DIAGNOSING,
                                         reason="ok")
            tenant.audit.append(tenant.merchant_id, AuditKind.RISK_SCORED,
                                {"p_rto": 0.5})

    assert OrderRepo(tenant).get(order.id).state is state_before
    assert tenant.audit.head(tenant.merchant_id) == head_before
    assert tenant.audit.verify(tenant.merchant_id).ok
