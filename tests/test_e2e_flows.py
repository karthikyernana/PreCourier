"""Black-box checks against a running local API.

Run with PRECOURIER_E2E_API_KEY set after following README's uvicorn steps.
These tests intentionally use httpx over TCP rather than TestClient.
"""
import os

import httpx
import pytest


BASE = os.getenv("PRECOURIER_E2E_BASE", "http://127.0.0.1:8000")
KEY = os.getenv("PRECOURIER_E2E_API_KEY")
pytestmark = pytest.mark.skipif(not KEY, reason="requires a running README API")


def _order(client, external_id="e2e-1"):
    response = client.post(
        "/v1/orders",
        headers={"Idempotency-Key": external_id},
        json={"phone": "+919876543210", "value": "1200.00",
              "external_id": external_id,
              "consent": {"simulated": True},
              "address": {"raw": "Flat 4", "city": "Bengaluru", "pincode": "560034"}},
    )
    assert response.is_success, response.text
    return response.json()


def test_running_api_order_outcome_reports_and_console_contract():
    with httpx.Client(base_url=BASE, headers={"X-API-Key": KEY}, timeout=10) as client:
        assert client.get("/v1/health").is_success
        decision = _order(client)
        order_id = decision["order_id"]
        assert client.get(f"/v1/orders/{order_id}").json()["order_id"] == order_id
        assert client.get(f"/v1/orders/{order_id}/messages").is_success
        assert client.post(f"/v1/orders/{order_id}/release", json={"reason": "e2e"}).is_success
        outcome = client.post("/v1/outcomes", headers={"Idempotency-Key": "outcome-e2e"},
                              json={"order_id": order_id, "kind": "delivered"})
        assert outcome.is_success, outcome.text
        assert client.get("/v1/reports/roi").is_success
        assert client.get("/v1/audit").json()["verification"]["ok"]
        assert client.get("/console").is_success


def test_running_api_rejects_malformed_and_negative_money():
    with httpx.Client(base_url=BASE, headers={"X-API-Key": KEY}, timeout=10) as client:
        bad = client.post("/v1/orders", json={"address": {}})
        assert bad.status_code == 422
        negative = client.post("/v1/orders", json={"phone": "+919876543210",
            "value": "-1.00", "address": {"raw": "x"}})
        assert negative.status_code == 422
