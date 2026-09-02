"""Merchant configuration and key lifecycle are tenant-scoped control-plane APIs."""

from fastapi.testclient import TestClient

from precourier.api import create_app
from precourier.config import kolam_config
from precourier.domain import Mode
from precourier.runtime import CountingIds, Runtime, SimClock
from precourier.store.db import Database
from precourier.tenancy import issue_api_key, provision_merchant


def _app():
    db = Database.connect(":memory:")
    runtime = Runtime(clock=SimClock("2026-03-01T09:00:00Z"), ids=CountingIds())
    tenant = provision_merchant(
        db, runtime, "Kolam Ceramics", merchant_id="kolam_ceramics",
        config=kolam_config().replace(mode=Mode.SHADOW))
    key, _ = issue_api_key(tenant, mode="test")
    return tenant, key, TestClient(create_app(db=db, runtime=runtime,
                                              sweeper=False))


def test_config_update_is_versioned_and_changes_the_decision_policy():
    tenant, key, client = _app()
    headers = {"Authorization": "Bearer %s" % key}
    before = client.get("/v1/merchant/config", headers=headers).json()
    response = client.patch("/v1/merchant/config", headers=headers, json={
        "mode": "gateway", "forward_freight": "90.00",
        "monthly_incentive_budget": "5000.00"})
    assert response.status_code == 200, response.text
    after = response.json()
    assert after["mode"] == "gateway"
    assert after["forward_freight_paise"] == 9000
    assert after["revision"] == before["revision"] + 1
    assert tenant.audit.verify(tenant.merchant_id).ok


def test_api_key_plaintext_is_returned_once_and_revoke_is_scoped():
    _, key, client = _app()
    headers = {"Authorization": "Bearer %s" % key}
    created = client.post("/v1/merchant/api-keys", headers=headers,
                          json={"mode": "test", "label": "integration"})
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["plaintext"].startswith("pk_test_")
    listed = client.get("/v1/merchant/api-keys", headers=headers).json()
    assert all(item["plaintext"] is None for item in listed)
    revoked = client.post("/v1/merchant/api-keys/%s/revoke" % body["id"],
                          headers=headers)
    assert revoked.status_code == 200
    assert any(item["id"] == body["id"] and item["revoked_at"]
               for item in client.get("/v1/merchant/api-keys",
                                      headers=headers).json())
