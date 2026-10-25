"""Payment-link execution and the signed Razorpay lifecycle boundary."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from precourier import states, tenancy
from precourier.adapters import (RazorpayPaymentLinkProvider,
                                 default_payment_link_provider)
from precourier.api import create_app
from precourier.config import kolam_config
from precourier.domain import ActionKind, Cause, Mode, Order, Address, OrderState
from precourier.runtime import CountingIds, Runtime, SimClock
from precourier.store.db import Database
from precourier.store.repo import OutboxRepo
from precourier.webhooks import sign_payload


SECRET = "webhook-secret"


def _verdict(action: ActionKind = ActionKind.PART_PAYMENT) -> states.Verdict:
    return states.Verdict(
        action=action, p_rto_ppm=800_000, cause=Cause.WEAK_INTENT,
        reason="payment offer is economically preferred", model_version="test")


class PaymentHarness:
    def __init__(self, *, fallback: str = "ship_cod",
                 action: ActionKind = ActionKind.PART_PAYMENT) -> None:
        self.db = Database.connect(":memory:")
        self.clock = SimClock("2026-03-01T09:00:00Z")
        self.runtime = Runtime(clock=self.clock, ids=CountingIds())
        config = kolam_config(merchant_id="payment_test").replace(
            mode=Mode.GATEWAY, payment_fallback=fallback,
            quiet_hours_start=0, quiet_hours_end=0)
        self.tenant = tenancy.provision_merchant(
            self.db, self.runtime, "Payment Test", merchant_id="payment_test",
            config=config)
        self.key, _ = tenancy.issue_api_key(self.tenant, mode="test")
        self.app = create_app(
            db=self.db, runtime=self.runtime,
            decider=states.StaticDecider(_verdict(action)),
            synthetic=True, sweeper=False)
        self.client = TestClient(self.app)

    @property
    def headers(self):
        return {"Authorization": "Bearer %s" % self.key}

    def order(self):
        return self.client.post("/v1/orders", headers=self.headers, json={
            "phone": "+919876543210", "value": "1200.00",
            "consent": {"simulated": True},
            "address": {"line1": "Flat 402", "city": "Bengaluru",
                        "state": "Karnataka", "pincode": "560034"},
        })


def _event(harness: PaymentHarness, order: dict, name: str,
           *, event_id: str = "evt_1", created_at: int | None = None) -> bytes:
    return json.dumps({
        "id": event_id, "event": name,
        "created_at": (created_at if created_at is not None else
                       int(harness.clock.now().timestamp())),
        "payload": {"payment_link": {"entity": {
            "id": order["payment_link_id"],
            "reference_id": order["order_id"],
            "amount": order["payment_link_amount_paise"],
            "created_at": int(harness.clock.now().timestamp()),
        }}},
    }).encode()


def _post_webhook(harness: PaymentHarness, body: bytes):
    return harness.client.post(
        "/v1/webhooks/razorpay", content=body,
        headers={"X-Razorpay-Signature": sign_payload(SECRET, body)})


def test_razorpay_signature_is_checked_before_event_processing(monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)
    h = PaymentHarness()
    body = json.dumps({
        "id": "evt_signature", "event": "payment_link.paid",
        "created_at": int(h.clock.now().timestamp()),
        "payload": {"payment_link": {"entity": {"reference_id": "missing"}}},
    }).encode()

    bad = h.client.post("/v1/webhooks/razorpay", content=body,
                        headers={"X-Razorpay-Signature": "bad"})
    assert bad.status_code == 401

    good = _post_webhook(h, body)
    assert good.status_code == 404  # HMAC passed; only the order is absent.


def test_old_razorpay_event_is_rejected(monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)
    h = PaymentHarness()
    body = _event(h, {"order_id": "missing", "payment_link_id": "plink",
                      "payment_link_amount_paise": 120000},
                   "payment_link.paid", created_at=int(h.clock.now().timestamp()) - 301)
    response = _post_webhook(h, body)
    assert response.status_code == 400
    assert "replay window" in response.json()["detail"]


def test_paid_event_is_idempotent_and_releases_a_converted_order(monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)
    h = PaymentHarness()
    created = h.order()
    assert created.status_code == 200, created.text
    order = created.json()
    assert order["state"] == "awaiting_payment"
    assert order["payment_link_amount_paise"] == 120000
    assert order["payment_link_url"].startswith("https://pay.invalid/")

    outbox = OutboxRepo(h.tenant).ready()
    assert outbox == []
    stored_outbox = h.db.one(
        "SELECT status, provider_ref, attempts FROM action_outbox "
        "WHERE merchant_id = :merchant_id AND order_id = :order_id",
        h.tenant.scope(order_id=order["order_id"]))
    assert stored_outbox["status"] == "succeeded"
    assert stored_outbox["attempts"] == 1
    assert stored_outbox["provider_ref"] == order["payment_link_id"]

    body = _event(h, order, "payment_link.paid")
    paid = _post_webhook(h, body)
    assert paid.status_code == 200, paid.text
    assert paid.json()["release"] is True
    assert paid.json()["state"] == "released_for_dispatch"

    detail = h.client.get("/v1/orders/%s" % order["order_id"],
                          headers=h.headers).json()
    assert detail["state"] == "released_for_dispatch"
    assert [(x["from"], x["to"]) for x in detail["transitions"]][-2:] == [
        ("awaiting_payment", "converted"),
        ("converted", "released_for_dispatch"),
    ]

    duplicate = _post_webhook(h, body)
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert h.db.scalar(
        "SELECT COUNT(*) FROM provider_events WHERE merchant_id = :merchant_id",
        h.tenant.scope(), default=0) == 1

    outcome = h.client.post("/v1/outcomes", headers=h.headers, json={
        "order_id": order["order_id"], "kind": "prepaid_delivered"})
    assert outcome.status_code == 200, outcome.text
    assert outcome.json()["kind"] == "prepaid_delivered"
    assert h.client.get("/v1/orders/%s" % order["order_id"],
                        headers=h.headers).json()["outcome"]["kind"] == (
                            "prepaid_delivered")


@pytest.mark.parametrize("fallback,expected", [
    ("ship_cod", "released_for_dispatch"),
    ("cancel", "cancelled"),
])
@pytest.mark.parametrize("event", ["payment_link.expired", "payment_link.cancelled"])
def test_expired_event_applies_merchant_fallback(monkeypatch, fallback, expected,
                                                 event):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)
    h = PaymentHarness(fallback=fallback)
    response = h.order()
    order = response.json()
    expired = _post_webhook(h, _event(h, order, event))
    assert expired.status_code == 200, expired.text
    assert expired.json()["state"] == expected
    detail = h.client.get("/v1/orders/%s" % order["order_id"],
                          headers=h.headers).json()
    assert any(t["to"] == "payment_expired" for t in detail["transitions"])


def test_prepaid_nudge_creates_a_non_partial_payment_link(monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)
    h = PaymentHarness(action=ActionKind.PREPAID_CONVERSION)

    response = h.order()

    assert response.status_code == 200, response.text
    order = response.json()
    assert order["state"] == "awaiting_payment"
    assert order["payment_link_amount_paise"] == 120000
    outbox = h.db.one(
        "SELECT payload_json FROM action_outbox WHERE merchant_id = :merchant_id "
        "AND order_id = :order_id", h.tenant.scope(order_id=order["order_id"]))
    assert json.loads(outbox["payload_json"])["accept_partial"] is False


def test_payment_provider_selection_is_fake_without_complete_razorpay_credentials(
        monkeypatch):
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    assert getattr(default_payment_link_provider(), "provider_name") == "fake"

    monkeypatch.setenv("RAZORPAY_KEY_ID", "key")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")
    assert getattr(default_payment_link_provider(), "provider_name") == "razorpay"


def test_payment_hold_deadline_uses_the_same_fallback(monkeypatch):
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)
    h = PaymentHarness(fallback="cancel")
    order = h.order().json()
    h.clock.advance(minutes=h.tenant.config.max_hold_minutes + 1)
    swept = h.client.post("/v1/holds/sweep", headers=h.headers)
    assert swept.status_code == 200
    assert swept.json()["released"] == 1
    assert swept.json()["orders"][0]["state"] == "cancelled"
    assert h.client.get("/v1/orders/%s" % order["order_id"],
                        headers=h.headers).json()["state"] == "cancelled"


def test_razorpay_adapter_sends_integer_paise_and_link_contract():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        seen["idempotency"] = request.headers.get("x-idempotency-key")
        return httpx.Response(200, json={"id": "plink_123",
                                         "short_url": "https://rzp.io/i/123"})

    order = Order(id="ord_123", merchant_id="m", buyer_id="b",
                  value_paise=120000, address=Address(line1="Flat 1"))
    provider = RazorpayPaymentLinkProvider(
        httpx.Client(transport=httpx.MockTransport(handler)),
        key_id="key", key_secret="secret", callback_url="https://merchant/cb")
    result = provider.create(
        order, amount_paise=120000, idempotency_key="payment:act_1",
        expire_by=1772357400, notify={"sms": True, "email": True},
        accept_partial=True)

    assert result["id"] == "plink_123"
    assert seen["auth"] == "Basic a2V5OnNlY3JldA=="
    assert seen["idempotency"] == "payment:act_1"
    assert seen["body"] == {
        "amount": 120000, "currency": "INR", "accept_partial": True,
        "description": "PreCourier order ord_123", "reference_id": "ord_123",
        "expire_by": 1772357400,
        "notify": {"sms": True, "email": True},
        "customer_notify": True, "callback_url": "https://merchant/cb",
        "callback_method": "get",
    }
