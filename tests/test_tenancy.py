"""Multi-tenancy: can merchant A ever see, or be priced by, merchant B?

Two kinds of test here, and both are needed.

The behavioural tests ask the obvious question -- does a read scoped to A return
B's rows -- and they can only ever cover the code paths somebody thought to
exercise. A cross-tenant leak is exactly the bug nobody thought to exercise.

So the centrepiece is `test_every_tenant_scoped_statement_filters_on_merchant_id`,
which parses the SQL out of the source and fails any statement that touches a
table with a `merchant_id` column without filtering on it. That one covers
statements written after this file was, which is the only coverage worth having
for an isolation property.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sqlite3
from typing import Dict, List, Set, Tuple

import pytest

from precourier import tenancy
from precourier.audit import AuditKind
from precourier.config import ActionEconomics, MerchantConfig
from precourier.domain import Channel, Mode, OutcomeKind
from precourier.money import to_paise
from precourier.store.repo import BuyerRepo, LedgerRepo, NotFound, OrderRepo
from precourier.tenancy import AuthError, TenantError
from tests.fixtures import (GOOD_ADDRESS, deterministic_runtime, fresh_db,
                            make_tenant, two_tenants)

PACKAGE = pathlib.Path(tenancy.__file__).parent

#: The modules allowed to contain SQL. Scanned by the structural test below.
#: `store/db.py` is excluded deliberately: its statements are table-agnostic
#: templates (`INSERT INTO %s ...`), so there is no table name to check and no
#: merchant to scope to -- that is the layer being wrapped, not a query.
SQL_MODULES = ("store/repo.py", "tenancy.py", "audit.py")


# ==========================================================================
# the structural test
# ==========================================================================

def _tenant_scoped_tables() -> Set[str]:
    """Tables carrying a `merchant_id` column, read from the schema itself.

    Derived rather than listed, so a table added in a later milestone is covered
    the moment it is created. A hand-maintained list would have to be remembered
    by whoever adds the table, which is the same person who forgot the WHERE
    clause.
    """
    with fresh_db() as db:
        scoped = set()
        for table in db.table_names():
            columns = [r["name"] for r in
                       db.raw.execute("PRAGMA table_info(%s)" % table)]
            if "merchant_id" in columns:
                scoped.add(table)
        return scoped


_STATEMENT = re.compile(r"\b(select|insert\s+into|update|delete\s+from)\b",
                        re.IGNORECASE)
_FILTERS = re.compile(r"merchant_id\s*=\s*[:?]|merchant_id\s*=\s*\w+\.\w*"
                      r"|\bmerchant_id\b\s*IN\s*\(", re.IGNORECASE)
_EXEMPT = re.compile(r"/\*\s*tenancy-exempt:", re.IGNORECASE)


def _sql_literals(path: pathlib.Path) -> List[Tuple[int, str]]:
    """Every string constant in the file that looks like a SQL statement.

    Adjacent string literals are concatenated by the parser, so a statement
    written across several source lines arrives here whole. `%`-formatted
    statements arrive as their template, which is what we want to check: the
    interpolated part is a generated column or placeholder list, never a WHERE
    clause.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if _STATEMENT.search(text) and " " in text.strip():
                found.append((node.lineno, text))
    return found


def _tables_mentioned(sql: str, scoped: Set[str]) -> Set[str]:
    hit = set()
    for table in scoped:
        if re.search(r"\b%s\b" % re.escape(table), sql, re.IGNORECASE):
            hit.add(table)
    return hit


def test_every_tenant_scoped_statement_filters_on_merchant_id():
    """No statement may touch a tenant-scoped table without naming merchant_id.

    Read the failure message literally: it is not a style complaint. A SELECT
    without the filter returns another merchant's orders; an UPDATE without it
    rewrites their state. The escape hatch is a `/* tenancy-exempt: reason */`
    comment inside the SQL, which is deliberately visible at the call site and
    has to justify itself -- there are two in the codebase, both in
    `authenticate`, where the merchant is the thing being discovered.
    """
    scoped = _tenant_scoped_tables()
    assert "orders" in scoped and "ledger_entries" in scoped, scoped

    offenders = []
    for relative in SQL_MODULES:
        path = PACKAGE / relative
        for lineno, sql in _sql_literals(path):
            tables = _tables_mentioned(sql, scoped)
            if not tables:
                continue
            if _EXEMPT.search(sql) or _FILTERS.search(sql):
                continue
            # An INSERT supplies merchant_id as a column rather than a filter.
            if re.search(r"insert\s+into", sql, re.IGNORECASE) and \
                    re.search(r"\bmerchant_id\b", sql, re.IGNORECASE):
                continue
            offenders.append("%s:%d touches %s without a merchant_id filter:\n"
                             "    %s" % (relative, lineno,
                                         ", ".join(sorted(tables)),
                                         " ".join(sql.split())[:160]))

    assert not offenders, (
        "%d statement(s) can read or write across tenants:\n\n%s"
        % (len(offenders), "\n\n".join(offenders)))


def test_dict_inserts_carry_a_merchant_id():
    """`db.insert("orders", {...})` builds its own SQL, so scan the dict instead.

    Same property as the test above, different syntax. Without this, the most
    common write in the codebase would be the one the structural test cannot see.
    """
    scoped = _tenant_scoped_tables()
    offenders = []
    for relative in SQL_MODULES:
        path = PACKAGE / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(
                func, "id", "")
            if name != "insert" or not node.args:
                continue
            table = node.args[0]
            if not (isinstance(table, ast.Constant)
                    and table.value in scoped):
                continue
            if len(node.args) < 2 or not isinstance(node.args[1], ast.Dict):
                offenders.append("%s:%d insert into %s with a non-literal dict"
                                 % (relative, node.lineno, table.value))
                continue
            keys = {k.value for k in node.args[1].keys
                    if isinstance(k, ast.Constant)}
            if "merchant_id" not in keys:
                offenders.append("%s:%d inserts into %s without merchant_id"
                                 % (relative, node.lineno, table.value))

    assert not offenders, "\n".join(offenders)


def test_scope_refuses_a_caller_supplied_merchant_id():
    """The mechanism that makes the structural test enforceable.

    Every scoped query gets its merchant_id from `Tenant.scope()`, and `scope()`
    raises if the caller tries to supply one. So "the query filters on
    merchant_id" and "the merchant_id is the authenticated one" are the same
    fact, and the structural test only has to check the first.
    """
    tenant = make_tenant()
    assert tenant.scope(id="x") == {"id": "x",
                                    "merchant_id": tenant.merchant_id}
    with pytest.raises(TenantError, match="must come from the authenticated"):
        tenant.scope(merchant_id="someone_else")


# ==========================================================================
# behavioural isolation
# ==========================================================================

def test_neither_merchant_can_read_the_other_s_orders():
    a, b = two_tenants()
    order = OrderRepo(a).create(
        buyer_id=BuyerRepo(a).upsert(phone="+919876543210").id,
        value_paise=to_paise("1200"), address=GOOD_ADDRESS)

    assert OrderRepo(a).get(order.id).id == order.id
    with pytest.raises(NotFound):
        OrderRepo(b).get(order.id)
    assert OrderRepo(b).find(order.id) is None
    assert OrderRepo(b).in_states([order.state]) == []
    assert OrderRepo(b).count_by_state() == {}


def test_a_missing_order_and_another_merchant_s_order_fail_identically():
    """Not an existence oracle.

    If "not yours" and "never existed" produced different errors, a merchant
    could enumerate another merchant's order ids by watching which lookups fail
    differently -- and order ids appear in URLs and webhooks.
    """
    a, b = two_tenants()
    order = OrderRepo(a).create(
        buyer_id=BuyerRepo(a).upsert(phone="+919876543210").id,
        value_paise=to_paise("1200"), address=GOOD_ADDRESS)

    with pytest.raises(NotFound) as theirs:
        OrderRepo(b).get(order.id)
    with pytest.raises(NotFound) as absent:
        OrderRepo(b).get("ord_does_not_exist")

    assert type(theirs.value) is type(absent.value)
    # Same shape of message, differing only in the id the caller already knew.
    assert str(theirs.value).replace(order.id, "X") == \
        str(absent.value).replace("ord_does_not_exist", "X")


def test_the_same_phone_is_two_different_buyers():
    """One human, two merchants, two independent reputations.

    A buyer who returned three parcels to merchant A is not a repeat returner at
    merchant B, and treating them as one would leak A's commercial data into B's
    decisions -- and be wrong besides, since RTO behaviour is partly a property
    of the merchant's own catalogue and packaging.
    """
    a, b = two_tenants()
    phone = "+919876500001"
    buyer_a = BuyerRepo(a).upsert(phone=phone)
    buyer_b = BuyerRepo(b).upsert(phone=phone)

    assert buyer_a.id != buyer_b.id
    # The hash is deterministic per merchant pepper; identity is not shared.
    with pytest.raises(NotFound):
        BuyerRepo(b).get(buyer_a.id)

    BuyerRepo(a).record_outcome(buyer_a.id, OutcomeKind.RTO,
                                at="2026-03-01T10:00:00Z")
    assert BuyerRepo(a).get(buyer_a.id).prior_rtos == 1
    assert BuyerRepo(b).get(buyer_b.id).prior_rtos == 0


def test_consent_granted_to_one_merchant_does_not_travel():
    """PRD section 14. Consent is per merchant, per channel, and never inferred."""
    a, b = two_tenants()
    phone = "+919876500002"
    buyer_a = BuyerRepo(a).upsert(phone=phone)
    buyer_b = BuyerRepo(b).upsert(phone=phone)

    BuyerRepo(a).set_consent(buyer_a.id, Channel.TELEGRAM, True)
    assert BuyerRepo(a).get(buyer_a.id).consents_to(Channel.TELEGRAM)
    assert not BuyerRepo(b).get(buyer_b.id).consents_to(Channel.TELEGRAM)
    # And not across channels within one merchant either.
    assert not BuyerRepo(a).get(buyer_a.id).consents_to(Channel.WHATSAPP)


def test_ledgers_do_not_mingle():
    a, b = two_tenants()
    from precourier import ledger
    order = OrderRepo(a).create(
        buyer_id=BuyerRepo(a).upsert(phone="+919876500003").id,
        value_paise=to_paise("1200"), address=GOOD_ADDRESS)
    LedgerRepo(a).post(ledger.breakdown(a.config, order.value_paise,
                                        OutcomeKind.RTO), order_id=order.id)

    assert LedgerRepo(a).net() == -to_paise("195")
    assert LedgerRepo(b).net() == 0
    assert LedgerRepo(b).by_reason() == {}


def test_a_cross_tenant_revoke_is_refused():
    a, b = two_tenants()
    _, key = tenancy.issue_api_key(a)

    with pytest.raises(TenantError):
        tenancy.revoke_api_key(b, key.id)
    assert tenancy.list_api_keys(a)[0].active
    assert tenancy.list_api_keys(b) == []


# ==========================================================================
# configuration
# ==========================================================================

def test_config_round_trips_through_the_database_exactly():
    """Every field, including the per-action economics.

    Asserted as whole-object equality rather than field by field, because the
    failure mode is a field somebody added to the dataclass and forgot to add to
    the schema -- which a field-by-field test written at the same time would also
    forget.
    """
    db, runtime = fresh_db(), deterministic_runtime()
    tenant = make_tenant(db, runtime)
    assert tenancy.load_config(db, tenant.merchant_id) == tenant.config


def test_each_merchant_is_priced_by_their_own_config():
    """The isolation bug that matters more than reads.

    A leaked read is a privacy incident. A merchant whose cancel threshold was
    computed from someone else's freight is silently making the wrong decision on
    every order, and nothing in the audit trail would look wrong.
    """
    a, b = two_tenants()
    assert a.config.forward_freight_paise != b.config.forward_freight_paise
    assert tenancy.load_config(a.db, "kolam_ceramics").gross_margin_bps == \
        a.config.gross_margin_bps
    assert tenancy.load_config(a.db, "rival_traders").gross_margin_bps == \
        b.config.gross_margin_bps

    reloaded = tenancy.load_tenant(a.db, a.runtime, "rival_traders")
    assert reloaded.config == b.config


def test_a_config_change_bumps_the_revision_and_audits_the_diff():
    """A merchant who changed their freight last Tuesday can see that they did.

    The history row and the audit entry both exist: the first so an old decision
    can be re-priced against the config that was live at the time, the second so
    the change itself is attributable.
    """
    tenant = make_tenant()
    before = tenancy.config_revision(tenant.db, tenant.merchant_id)

    updated = tenancy.save_config(
        tenant, tenant.config.replace(forward_freight_paise=9000,
                                      mode=Mode.GATEWAY))
    assert tenancy.config_revision(tenant.db, tenant.merchant_id) == before + 1
    assert updated.config.forward_freight_paise == 9000
    assert updated.mode is Mode.GATEWAY

    entry = [e for e in tenant.audit.recent(tenant.merchant_id)
             if e.kind == AuditKind.CONFIG_CHANGED][0]
    assert entry.payload["changed"]["forward_freight_paise"] == [7000, 9000]
    assert entry.payload["changed"]["mode"] == ["shadow", "gateway"]
    assert "gross_margin_bps" not in entry.payload["changed"], (
        "an unchanged field must not appear in the diff")


def test_dropping_an_override_reverts_to_the_default_and_never_to_free():
    """Delete-and-rewrite, and a floor under what a missing row can mean.

    Two properties in one test because they are two halves of the same decision.
    The rows are deleted and rewritten rather than upserted, so a merchant's old
    custom price genuinely goes away instead of lingering. And what it reverts to
    is the platform default, not an unpriced action -- because an unpriced action
    reads as free, and a free action clears every threshold the gate derives. The
    agent would take it on every order.
    """
    tenant = make_tenant()
    from precourier.config import DEFAULT_ACTION_ECONOMICS
    from precourier.domain import ActionKind

    expensive = tenant.config.replace(action_economics={
        **tenant.config.action_economics,
        ActionKind.PART_PAYMENT: ActionEconomics(
            cost_paise=5000, harm_bps=900, reversible=True,
            contacts_buyer=True)})
    tenant = tenancy.save_config(tenant, expensive)
    assert tenancy.load_config(
        tenant.db, tenant.merchant_id).economics(
            ActionKind.PART_PAYMENT).cost_paise == 5000

    trimmed = dict(tenant.config.action_economics)
    trimmed.pop(ActionKind.PART_PAYMENT)
    tenancy.save_config(tenant, tenant.config.replace(action_economics=trimmed))

    reloaded = tenancy.load_config(tenant.db, tenant.merchant_id)
    assert reloaded.economics(ActionKind.PART_PAYMENT) == \
        DEFAULT_ACTION_ECONOMICS[ActionKind.PART_PAYMENT]
    assert reloaded.economics(ActionKind.PART_PAYMENT).cost_paise > 0, (
        "a dropped override must not make the action free")


def test_an_action_absent_from_the_config_is_still_priced():
    """The same floor, for a config that never had the entry at all.

    `MerchantConfig` built by hand -- as onboarding and the backtest both do --
    can omit `action_economics` entirely. Every action must still carry its cost
    and its harm rate, or the gate's arithmetic silently loses the terms that
    stop it from acting on everything.
    """
    from precourier.config import DEFAULT_ACTION_ECONOMICS
    from precourier.domain import ActionKind

    bare = MerchantConfig(merchant_id="m", monthly_incentive_budget_paise=0,
                          action_economics={})
    for action in ActionKind:
        assert bare.economics(action) == DEFAULT_ACTION_ECONOMICS[action]
    assert bare.economics(ActionKind.ADDRESS_REPAIR).harm_bps > 0


def test_provisioning_the_same_merchant_twice_is_refused():
    db, runtime = fresh_db(), deterministic_runtime()
    make_tenant(db, runtime, merchant_id="kolam_ceramics")
    with pytest.raises(TenantError):
        tenancy.provision_merchant(db, runtime, "Impostor",
                                   merchant_id="kolam_ceramics")


# ==========================================================================
# authentication
# ==========================================================================

def test_a_key_authenticates_to_exactly_its_own_merchant():
    a, b = two_tenants()
    plaintext_a, _ = tenancy.issue_api_key(a)
    plaintext_b, _ = tenancy.issue_api_key(b, mode="live")

    resolved_a = tenancy.authenticate(a.db, a.runtime, plaintext_a)
    resolved_b = tenancy.authenticate(a.db, a.runtime, plaintext_b)

    assert resolved_a.merchant_id == "kolam_ceramics"
    assert resolved_b.merchant_id == "rival_traders"
    assert not resolved_a.is_live and resolved_b.is_live
    # Authentication also loads that merchant's own pricing.
    assert resolved_b.config.forward_freight_paise == \
        b.config.forward_freight_paise


def test_the_plaintext_key_is_nowhere_in_the_database():
    """Scans every text column of every table, not just `api_keys.key_hash`.

    Targeted assertions miss the copy: the audit payload, a config blob, an
    idempotency record. A key that leaked into an audit row is still a leaked
    key, and audit rows are the ones we export.
    """
    tenant = make_tenant()
    plaintext, record = tenancy.issue_api_key(tenant)

    for table in tenant.db.table_names():
        for row in tenant.db.raw.execute("SELECT * FROM %s" % table):
            for value in tuple(row):
                if isinstance(value, str):
                    assert plaintext not in value, \
                        "plaintext API key found in %s" % table

    # The prefix is stored, deliberately, so a merchant can recognise the key.
    assert record.key_prefix in plaintext
    assert len(record.key_prefix) < len(plaintext) / 2


def test_every_authentication_failure_looks_the_same():
    """Malformed, unknown, revoked, suspended -- one bare `AuthError`.

    Distinguishing them tells an attacker that a key they found was once real,
    which is most of what they need to know.
    """
    tenant = make_tenant()
    plaintext, record = tenancy.issue_api_key(tenant)
    tenancy.revoke_api_key(tenant, record.id)

    messages = set()
    for candidate in ("", "not-a-key", "pk_test_" + "0" * 40, plaintext):
        with pytest.raises(AuthError) as caught:
            tenancy.authenticate(tenant.db, tenant.runtime, candidate)
        messages.add(str(caught.value))
    assert len(messages) == 1, messages


def test_a_suspended_merchant_cannot_authenticate():
    tenant = make_tenant()
    plaintext, _ = tenancy.issue_api_key(tenant)
    assert tenancy.authenticate(tenant.db, tenant.runtime, plaintext)

    tenant.db.execute("UPDATE merchants SET status = 'suspended' WHERE id = :id",
                      {"id": tenant.merchant_id})
    with pytest.raises(AuthError):
        tenancy.authenticate(tenant.db, tenant.runtime, plaintext)


def test_authentication_records_last_used():
    """So a merchant can tell whether the key they are about to revoke is live."""
    tenant = make_tenant()
    plaintext, record = tenancy.issue_api_key(tenant)
    assert tenancy.list_api_keys(tenant)[0].last_used_at is None

    tenancy.authenticate(tenant.db, tenant.runtime, plaintext)
    assert tenancy.list_api_keys(tenant)[0].last_used_at is not None


def test_two_keys_for_one_merchant_both_work_and_revoke_independently():
    tenant = make_tenant()
    first, first_record = tenancy.issue_api_key(tenant, label="server")
    second, _ = tenancy.issue_api_key(tenant, label="worker")

    tenancy.revoke_api_key(tenant, first_record.id)
    with pytest.raises(AuthError):
        tenancy.authenticate(tenant.db, tenant.runtime, first)
    assert tenancy.authenticate(tenant.db, tenant.runtime,
                                second).merchant_id == tenant.merchant_id


def test_revoking_a_revoked_key_is_refused_rather_than_silently_accepted():
    tenant = make_tenant()
    _, record = tenancy.issue_api_key(tenant)
    tenancy.revoke_api_key(tenant, record.id)
    with pytest.raises(TenantError):
        tenancy.revoke_api_key(tenant, record.id)
