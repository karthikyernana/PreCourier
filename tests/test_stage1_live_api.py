"""Stage 1's real order-intake path, exercised through in-process HTTP."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi.testclient import TestClient

from precourier import states, tenancy
from precourier.api import create_app
from precourier.config import kolam_config
from precourier.domain import Mode
from precourier.runtime import CountingIds, Runtime, SimClock
from precourier.store.db import Database
from precourier.store.repo import ActionRepo
from tests.fixtures import DEFECTIVE_ADDRESS, GOOD_ADDRESS, fresh_db


class LiveHarness:
    def __init__(self, *, mode: Mode = Mode.ADVISORY,
                 actor: Optional[states.Actor] = None,
                 holdout_rate_percent: int = 0,
                 **config_overrides: Any) -> None:
        self.db: Database = fresh_db()
        self.runtime = Runtime(clock=SimClock("2026-03-01T09:00:00Z"),
                               ids=CountingIds())
        config = kolam_config(merchant_id="stage1_live").replace(
            mode=mode, holdout_rate_percent=holdout_rate_percent,
            **config_overrides)
        self.tenant = tenancy.provision_merchant(
            self.db, self.runtime, "Stage 1 Live", merchant_id="stage1_live",
            config=config)
        self.key, _ = tenancy.issue_api_key(self.tenant, mode="test")
        self.client = TestClient(create_app(
            db=self.db, runtime=self.runtime, actor=actor, synthetic=True,
            sweeper=False), raise_server_exceptions=False)

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": "Bearer %s" % self.key}

    def post_order(self, *, address: Optional[Dict[str, str]] = None,
                   phone: str = "+919876543210",
                   consent: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        response = self.client.post(
            "/v1/orders", headers=self.headers,
            json={"phone": phone,
                  "address": address or {
                      "line1": GOOD_ADDRESS.line1,
                      "line2": GOOD_ADDRESS.line2,
                      "landmark": GOOD_ADDRESS.landmark,
                      "locality": GOOD_ADDRESS.locality,
                      "city": GOOD_ADDRESS.city,
                      "state": GOOD_ADDRESS.state,
                      "pincode": GOOD_ADDRESS.pincode},
                  "value": "1200.00",
                  "consent": {"simulated": True}
                  if consent is None else consent})
        assert response.status_code == 200, response.text
        return response.json()

    def report_rto(self, order_id: str) -> None:
        response = self.client.post(
            "/v1/outcomes", headers=self.headers,
            json={"order_id": order_id, "kind": "rto"})
        assert response.status_code == 200, response.text


def _build_refusal_history(harness: LiveHarness, count: int = 3) -> None:
    """Create reported RTO labels for the same buyer before the test order."""
    for _ in range(count):
        body = harness.post_order()
        harness.report_rto(body["order_id"])


def test_default_live_path_chooses_a_paid_action_and_persists_rupee_explanation():
    harness = LiveHarness(actor=states.SimulatedActor(), max_messages_per_buyer_24h=10)
    _build_refusal_history(harness)

    body = harness.post_order()

    assert body["chosen_action"] in {"partial_cod", "prepaid_nudge"}
    assert body["expected_saving_paise"] > 0
    assert body["runner_up"]
    assert body["top_drivers"]
    assert body["explanation"]["cause_probabilities"]["refusal"] > 300_000
    assert body["holdout"] is False

    order = harness.client.get(
        "/v1/orders/%s" % body["order_id"], headers=harness.headers)
    assert order.status_code == 200
    explanation = order.json()["decisions"][-1]
    assert explanation["chosen_action"] == body["chosen_action"]
    assert explanation["expected_saving_paise"] == body["expected_saving_paise"]
    assert explanation["explanation"]["evaluations"]


def test_low_risk_order_chooses_ship_as_is():
    harness = LiveHarness(actor=states.SimulatedActor())

    body = harness.post_order()

    assert body["chosen_action"] == "ship_as_is"
    assert body["action"] == "no_action"
    assert body["expected_saving_paise"] == 0
    assert body["explanation"]["evaluations"]


def test_holdout_is_scored_stored_and_ships_as_is():
    harness = LiveHarness(actor=states.SimulatedActor(), holdout_rate_percent=100)

    body = harness.post_order(address={
        "raw": DEFECTIVE_ADDRESS.raw,
        "line1": DEFECTIVE_ADDRESS.line1,
        "city": DEFECTIVE_ADDRESS.city,
        "pincode": DEFECTIVE_ADDRESS.pincode})

    assert body["holdout"] is True
    assert body["release"] is True
    assert body["action"] == "no_action"
    assert body["explanation"]["cause_probabilities"]
    assert body["explanation"]["evaluations"]
    stored = harness.client.get(
        "/v1/orders/%s" % body["order_id"], headers=harness.headers).json()
    assert stored["decisions"][-1]["holdout"] is True


def test_guard_block_records_the_blocked_action_and_fallback_release():
    harness = LiveHarness()  # default NoChannelActor: consent passes, channel does not
    _build_refusal_history(harness)

    body = harness.post_order()

    assert body["chosen_action"] in {"partial_cod", "prepaid_nudge"}
    assert body["blocked_reason"] == "no_channel"
    assert body["release"] is True
    actions = ActionRepo(harness.tenant).for_order(body["order_id"])
    assert actions[-1].status == "blocked"
    assert actions[-1].blocked_reason == "no_channel"


def test_shadow_mode_records_the_counterfactual_but_does_not_act():
    harness = LiveHarness(mode=Mode.SHADOW, actor=states.SimulatedActor())

    body = harness.post_order(
        address={"raw": DEFECTIVE_ADDRESS.raw,
                 "line1": DEFECTIVE_ADDRESS.line1,
                 "city": DEFECTIVE_ADDRESS.city,
                 "pincode": DEFECTIVE_ADDRESS.pincode})

    assert body["release"] is True
    assert body["suppressed"] is True
    assert body["action"] == "no_action"
    assert body["chosen_action"] == "fix_address"
    assert ActionRepo(harness.tenant).for_order(body["order_id"]) == []
