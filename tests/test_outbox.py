"""Durable provider intent is idempotent and tenant-scoped."""

from precourier.domain import ActionKind
from precourier.store.repo import ActionRepo, BuyerRepo, OrderRepo, OutboxRepo
from tests.fixtures import GOOD_ADDRESS, make_tenant, two_tenants


def _action(tenant, suffix=""):
    buyer = BuyerRepo(tenant).upsert(phone="+9198765432%s" % (10 + len(suffix)))
    order = OrderRepo(tenant).create(
        buyer_id=buyer.id, value_paise=120_000, address=GOOD_ADDRESS,
        external_id="outbox-order-%s" % suffix)
    action = ActionRepo(tenant).record(order.id, ActionKind.ADDRESS_REPAIR)
    return order, action


def test_outbox_enqueue_is_idempotent_and_tracks_provider_lifecycle():
    tenant = make_tenant()
    repo = OutboxRepo(tenant)
    order, action = _action(tenant, "one")
    first = repo.enqueue(
        order.id, action.id, ActionKind.ADDRESS_REPAIR,
        {"action": "address_repair"}, idempotency_key="action:action-1")
    replay = repo.enqueue(
        order.id, action.id, ActionKind.ADDRESS_REPAIR,
        {"action": "address_repair"}, idempotency_key="action:action-1")
    assert replay.id == first.id
    assert replay.status == "pending"
    processing = repo.mark_processing(first.id)
    assert processing.status == "processing"
    succeeded = repo.succeed(first.id, provider_ref="provider-1")
    assert succeeded.status == "succeeded"
    assert succeeded.provider_ref == "provider-1"
    assert repo.ready() == []


def test_failed_outbox_work_is_retryable_and_auditable():
    tenant = make_tenant()
    repo = OutboxRepo(tenant)
    order, action = _action(tenant, "two")
    record = repo.enqueue(
        order.id, action.id, ActionKind.ADDRESS_REPAIR,
        {"action": "address_repair"}, idempotency_key="action:action-1")
    repo.mark_processing(record.id)
    failed = repo.fail(record.id, "provider_timeout", retry_minutes=0)
    assert failed.status == "failed"
    assert failed.last_error == "provider_timeout"
    assert failed.attempts == 1
    assert [item.id for item in repo.ready()] == [record.id]
    assert tenant.audit.verify(tenant.merchant_id).ok


def test_outbox_queries_cannot_cross_tenant_boundaries():
    a, b = two_tenants()
    order, action = _action(a, "three")
    record = OutboxRepo(a).enqueue(
        order.id, action.id, ActionKind.ADDRESS_REPAIR,
        {"action": "address_repair"}, idempotency_key="action:%s" % action.id)
    assert OutboxRepo(a).get(record.id).id == record.id
    assert OutboxRepo(b).ready() == []
