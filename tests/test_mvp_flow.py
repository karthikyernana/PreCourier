"""The India-first MVP vertical slice over real HTTP."""

from fastapi.testclient import TestClient

from precourier.api import create_app
from precourier import states
from precourier.config import kolam_config
from precourier.domain import Mode
from precourier.runtime import CountingIds, Runtime, SimClock
from precourier.store.db import Database
from precourier import tenancy


def _harness():
    db = Database.connect(":memory:")
    runtime = Runtime(clock=SimClock("2026-03-01T09:00:00Z"), ids=CountingIds())
    config = kolam_config().replace(mode=Mode.GATEWAY)
    tenant = tenancy.provision_merchant(
        db, runtime, "Kolam Ceramics", merchant_id="kolam_ceramics",
        config=config)
    key, _ = tenancy.issue_api_key(tenant, mode="test")
    app = create_app(db=db, runtime=runtime, actor=states.SimulatedActor(),
                     synthetic=True, sweeper=False)
    return tenant, key, TestClient(app)


def test_default_app_runs_the_real_rule_and_simulated_repair_path():
    tenant, key, client = _harness()
    headers = {"Authorization": "Bearer %s" % key}
    response = client.post("/v1/orders", headers=headers, json={
        "phone": "+919876543210",
        "value": "1200.00",
        "consent": {"simulated": True},
        "address": {"raw": "near temple", "city": "Bangalore",
                    "pincode": "110001"},
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["release"] is False
    assert body["state"] == "awaiting_buyer"
    assert body["cause"] == "address_defect"
    assert body["recommended"] == "address_repair"
    assert body["p_rto_ppm"] > 0
    assert body["decision_id"] and body["action_id"]

    queue = client.get("/v1/orders", headers=headers)
    assert queue.status_code == 200
    assert queue.json()["orders"][0]["order_id"] == body["order_id"]

    order_id = body["order_id"]
    detail = client.get("/v1/orders/%s" % order_id, headers=headers)
    assert detail.status_code == 200
    detail_body = detail.json()
    assert len(detail_body["decisions"]) == 1
    assert detail_body["decisions"][0]["gate_inputs"]["swing_paise"] == 46_000
    assert len(detail_body["features"]) == 1
    assert detail_body["features"][0]["pass_no"] == 1
    assert detail_body["risk_scores"][0]["model_version"] == "rule-v1"
    assert detail_body["diagnoses"][0]["cause"] == "address_defect"

    messages = client.get("/v1/orders/%s/messages" % order_id,
                          headers=headers)
    assert messages.status_code == 200
    assert messages.json()[0]["direction"] == "outbound"
    assert messages.json()[0]["status"] == "sent"

    reply = client.post(
        "/v1/orders/%s/messages/inbound" % order_id,
        headers=headers,
        json={
            "body": "Flat 402, Ashirvad Residency, Koramangala 4th Block, Bengaluru",
            "channel": "simulated",
            "address": {
                "line1": "Flat 402, Ashirvad Residency",
                "locality": "Koramangala 4th Block",
                "city": "Bengaluru", "state": "Karnataka",
                "pincode": "560034",
            },
        })
    assert reply.status_code == 200, reply.text
    assert reply.json()["release"] is True
    assert reply.json()["state"] == "released_for_dispatch"

    audit = client.get("/v1/audit", headers=headers)
    assert audit.status_code == 200
    assert audit.json()["verification"]["ok"] is True


def test_mvp_report_and_health_are_explicit_about_provenance():
    tenant, key, client = _harness()
    headers = {"Authorization": "Bearer %s" % key}
    health = client.get("/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["synthetic_data"] is True

    report = client.get("/v1/reports/roi", headers=headers)
    assert report.status_code == 200
    body = report.json()
    assert body["provenance"] == "persisted_merchant_outcomes"
    assert body["synthetic_data"] is False
    assert body["merchant_id"] == tenant.merchant_id


def test_outcome_is_reconciled_into_the_live_roi_report():
    tenant, key, client = _harness()
    headers = {"Authorization": "Bearer %s" % key}
    created = client.post("/v1/orders", headers=headers, json={
        "phone": "+919876543210", "value": "1200.00",
        "consent": {"simulated": True},
        "address": {"line1": "Flat 402, Residency",
                    "locality": "Koramangala", "city": "Bengaluru",
                    "state": "Karnataka", "pincode": "560034"},
    }).json()
    outcome = client.post("/v1/outcomes", headers=headers, json={
        "order_id": created["order_id"], "kind": "rto",
    })
    assert outcome.status_code == 200, outcome.text
    assert outcome.json()["net_paise"] == -19_500
    assert outcome.json()["release"] is True

    report = client.get("/v1/reports/roi", headers=headers).json()
    assert report["outcomes"] == {"rto": 1}
    # The realised report includes the attempted address-repair message (-₹3)
    # as well as the carrier's RTO outcome (-₹195).
    assert report["net_paise"] == -19_800
    assert report["action_spend_paise"] == 300
    assert report["ledger_by_reason_paise"]["return_freight"] == -7_000


def test_backtest_endpoint_returns_a_synthetic_report():
    _, key, client = _harness()
    response = client.post("/v1/backtest?seed=42&orders=10",
                           headers={"Authorization": "Bearer %s" % key})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["synthetic_data"] is True
    assert set(body["arms"]) == {"ship_all", "block_best", "agent"}


def test_rejected_patch_is_logged_and_does_not_change_the_current_address():
    tenant, key, client = _harness()
    headers = {"Authorization": "Bearer %s" % key}
    created = client.post("/v1/orders", headers=headers, json={
        "phone": "+919876543210", "value": "1200.00",
        "consent": {"simulated": True},
        "address": {"line1": "Flat 402, Residency",
                    "locality": "Koramangala", "city": "Bengaluru",
                    "state": "Karnataka", "pincode": "560034"},
    }).json()
    order_id = created["order_id"]
    rejected = client.post(
        "/v1/orders/%s/messages/inbound" % order_id,
        headers=headers,
        json={"body": "new address", "channel": "simulated", "address": {
            "line1": "Flat 402, Residency", "city": "Bengaluru",
            "state": "Karnataka", "pincode": "000000"}})
    assert rejected.status_code == 422
    detail = client.get("/v1/orders/%s" % order_id, headers=headers).json()
    assert detail["address"]["pincode"] == "560034"
    assert any(entry.kind == "address.patch_rejected"
               for entry in tenant.audit.for_order(tenant.merchant_id, order_id))


def test_provider_actor_runs_after_action_intent_commits():
    tenant, key, client = _harness()
    headers = {"Authorization": "Bearer %s" % key}
    app = client.app
    actor = app.state.precourier.actor
    assert actor is not None

    # The default simulated actor is deliberately local. This test verifies the
    # durable action/message result rather than relying on provider timing.
    response = client.post("/v1/orders", headers=headers, json={
        "phone": "+919876543210", "value": "1200.00",
        "consent": {"simulated": True},
        "address": {"raw": "near temple", "city": "Bangalore",
                    "pincode": "110001"},
    })
    assert response.status_code == 200
    order_id = response.json()["order_id"]
    action = response.json()["action_id"]
    stored = tenant.db.one(
        "SELECT status FROM actions WHERE merchant_id = :merchant_id AND id = :id",
        tenant.scope(id=action))
    assert stored["status"] == "succeeded"
    outbox = tenant.db.one(
        "SELECT status, attempts FROM action_outbox "
        "WHERE merchant_id = :merchant_id AND action_id = :id",
        tenant.scope(id=action))
    assert outbox["status"] == "succeeded"
    assert outbox["attempts"] == 1


def test_default_app_has_no_outbound_provider_or_synthetic_claim():
    db = Database.connect(":memory:")
    runtime = Runtime(clock=SimClock("2026-03-01T09:00:00Z"), ids=CountingIds())
    tenant = tenancy.provision_merchant(
        db, runtime, "Kolam Ceramics", merchant_id="kolam_ceramics",
        config=kolam_config().replace(mode=Mode.GATEWAY))
    app = create_app(db=db, runtime=runtime, sweeper=False)
    assert app.state.precourier.synthetic is False
    assert isinstance(app.state.precourier.actor, states.NoChannelActor)


def test_operator_console_is_a_real_read_only_api_view():
    db = Database.connect(":memory:")
    runtime = Runtime(clock=SimClock("2026-03-01T09:00:00Z"), ids=CountingIds())
    tenancy.provision_merchant(
        db, runtime, "Kolam Ceramics", merchant_id="kolam_ceramics",
        config=kolam_config().replace(mode=Mode.GATEWAY))
    app = create_app(db=db, runtime=runtime, sweeper=False)
    response = TestClient(app).get("/console")
    assert response.status_code == 200
    assert "Decision queue" in response.text
    assert "Gate arithmetic" in response.text
    assert "marketing simulator" not in response.text
