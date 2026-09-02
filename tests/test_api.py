"""The HTTP surface, and the three promises it must keep under it.

PRD section 1 makes one hard commitment, and this file is that commitment expressed
over real HTTP rather than at the service layer:

1.  **An order posted over HTTP reaches `released_for_dispatch`** -- in every mode,
    including gateway, and including when the agent has nothing useful to say.
2.  **The `max_hold_minutes` timeout releases it** even if nobody replies and
    nothing resolves.
3.  **An internal exception releases it too**, with `degraded: true` so the event is
    countable rather than invisible.

Every test drives the *real* application. `create_app` takes an injected `Database`
and `Runtime`, so a test gets production code on a simulated clock rather than a
parallel implementation of it. There is no test-only code path to keep in sync, and
`SimClock` is how a four-hour hold expires in a millisecond.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

from precourier import states, tenancy
from precourier.api import create_app
from precourier.api.deps import UNAUTHORIZED
from precourier.audit import AuditKind
from precourier.config import MerchantConfig, kolam_config
from precourier.domain import ActionKind, Cause, Channel, Mode, OrderState
from precourier.runtime import CountingIds, Runtime, SimClock
from precourier.states import Dispatch, StaticDecider, Verdict
from precourier.store.db import Database
from precourier.store.repo import ActionRepo, DecisionRepo, OrderRepo
from tests.fixtures import fresh_db

ADDRESS = {"line1": "Flat 402, Ashirvad Residency", "line2": "12th Main Road",
           "locality": "Koramangala 4th Block", "city": "Bengaluru",
           "state": "Karnataka", "pincode": "560034"}

DEFECTIVE = {"raw": "near the big temple, bangalore",
             "line1": "near the big temple", "city": "Bangalore",
             "pincode": "110001"}


# ==========================================================================
# harness
# ==========================================================================

class _SendingActor(states.Actor):
    """An actor that reports a successful send without sending anything.

    Distinct from `NoChannelActor`, which reports a *block*. This one exists to
    exercise the hold path: an order can only wait for a reply if a message
    genuinely went out, and until M7 wires Telegram there is no channel that can
    say so truthfully. The dispatch is labelled `SIMULATED`, which is what the
    `actions` row and the console will show.
    """

    def __init__(self) -> None:
        self.sent: list = []

    def perform(self, tenant, order, buyer, verdict) -> Dispatch:
        self.sent.append(order.id)
        return Dispatch(sent=True, channel=Channel.SIMULATED,
                        detail="simulated send", message_id="msg_test")


class _BrokenDecider(states.Decider):
    """Raises. Stands in for a model server that is down mid-request."""

    def decide(self, tenant, order, buyer):
        raise RuntimeError("model server unreachable")


def repair_verdict(*, p_rto_ppm: int = 400_000,
                   threshold_ppm: int = 29_500) -> Verdict:
    """A verdict that asks for an address repair, with plausible arithmetic.

    The numbers are the derived ones from PRD section 5 -- a 40% RTO probability
    against a 2.95% repair threshold -- so a test failure that changes them is
    visible as a changed *number*, not as a changed boolean.
    """
    return Verdict(action=ActionKind.ADDRESS_REPAIR, p_rto_ppm=p_rto_ppm,
                   cause=Cause.ADDRESS_DEFECT, threshold_ppm=threshold_ppm,
                   expected_delta_micro=1_800_000,
                   reason="no house number, and the pincode is not in the "
                          "stated city",
                   model_version="test-1",
                   gate_inputs={"p_rto_ppm": p_rto_ppm,
                                "threshold_ppm": threshold_ppm})


class Harness:
    """One merchant, one app, one client, and the plumbing to inspect the result."""

    def __init__(self, *, mode: Mode = Mode.GATEWAY,
                 decider: Optional[states.Decider] = None,
                 actor: Optional[states.Actor] = None,
                 start: str = "2026-03-01T09:00:00Z",
                 **config_overrides: Any) -> None:
        self.db: Database = fresh_db()
        # The clock is held by name so a test can advance it. Same object the app
        # reads, so moving it forward moves the application's notion of "now" --
        # which is how a four-hour hold expires inside a millisecond.
        self.clock = SimClock(start)
        self.runtime = Runtime(clock=self.clock, ids=CountingIds())

        config = kolam_config(merchant_id="kolam_ceramics").replace(
            mode=mode, **config_overrides)
        self.tenant = tenancy.provision_merchant(
            self.db, self.runtime, "Kolam Ceramics",
            merchant_id="kolam_ceramics", config=config)
        self.key, _ = tenancy.issue_api_key(self.tenant, mode="test")

        self.app = create_app(
            db=self.db, runtime=self.runtime,
            decider=decider if decider is not None else states.NullDecider(),
            actor=actor if actor is not None else states.NoChannelActor(),
            synthetic=True, sweeper=False)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    # ------------------------------------------------------------------ HTTP

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": "Bearer %s" % self.key}

    def post_order(self, *, address: Dict[str, str] = None, value: str = "1200.00",
                   external_id: str = "", consent: Optional[Dict] = None,
                   key: Optional[str] = None, **extra: Any):
        # Consent defaults to the merchant's *own* channel. Consent is per channel,
        # and a flag for a channel the merchant does not use is not consent -- so a
        # harness that always sent `whatsapp: true` would have every contact action
        # blocked on `no_consent` and would silently stop testing the paths below it.
        default_consent = {self.tenant.config.default_channel.value: True}
        body: Dict[str, Any] = {
            "phone": "+919876543210",
            "address": dict(address if address is not None else ADDRESS),
            "value": value,
            "consent": default_consent if consent is None else consent}
        if external_id:
            body["external_id"] = external_id
        body.update(extra)
        headers = dict(self.headers)
        if key:
            headers["Idempotency-Key"] = key
        return self.client.post("/v1/orders", json=body, headers=headers)

    def get_order(self, order_id: str):
        return self.client.get("/v1/orders/%s" % order_id, headers=self.headers)

    def sweep(self):
        return self.client.post("/v1/holds/sweep", headers=self.headers)

    # ---------------------------------------------------------------- reading

    def state(self, order_id: str) -> OrderState:
        return OrderRepo(self.tenant).get(order_id).state

    def audit_kinds(self, order_id: str):
        return [r.kind for r in self.tenant.audit.for_order(
            self.tenant.merchant_id, order_id)]


@pytest.fixture
def gateway():
    return Harness(mode=Mode.GATEWAY, decider=StaticDecider(repair_verdict()),
                   actor=_SendingActor())


# ==========================================================================
# 1. an order posted over HTTP reaches released_for_dispatch
# ==========================================================================

@pytest.mark.parametrize("mode", [Mode.SHADOW, Mode.ADVISORY, Mode.GATEWAY])
def test_an_order_with_no_model_releases_in_every_mode(mode):
    """The day-one deployment. No model, no channel, and the parcel still ships.

    This is the shape of a merchant's first hour with PreCourier: nothing is trained
    and nothing is wired. A system that holds orders in that state -- or that returns
    an error the integration has to interpret -- has failed before it started.
    """
    h = Harness(mode=mode)
    response = h.post_order()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["release"] is True
    assert body["held_until"] is None
    assert body["state"] == OrderState.RELEASED_FOR_DISPATCH.value
    assert body["degraded"] is False
    assert body["action"] == ActionKind.NO_ACTION.value
    # Not silence: there is a decision row explaining why nothing happened.
    assert body["decision_id"]
    assert "no risk model" in body["reason"]


@pytest.mark.parametrize("mode,expect_hold", [(Mode.SHADOW, False),
                                              (Mode.ADVISORY, False),
                                              (Mode.GATEWAY, True)])
def test_mode_decides_whether_the_label_waits(mode, expect_hold):
    """One verdict, three modes, three different answers about the parcel.

    The verdict is identical in all three -- same probability, same threshold, same
    recommendation. Only the mode differs, which is the property that makes shadow
    mode a safe way to evaluate this thing: the reasoning is not a reduced version of
    gateway mode's, it is the same reasoning with the actions suppressed.
    """
    h = Harness(mode=mode, decider=StaticDecider(repair_verdict()),
                actor=_SendingActor())
    body = h.post_order().json()

    assert body["recommended"] == ActionKind.ADDRESS_REPAIR.value
    assert body["p_rto_ppm"] == 400_000
    assert body["threshold_ppm"] == 29_500

    if expect_hold:
        assert body["release"] is False
        assert body["held_until"] is not None
        assert body["action"] == ActionKind.ADDRESS_REPAIR.value
    else:
        assert body["release"] is True, body
        assert body["held_until"] is None


def test_shadow_mode_records_the_counterfactual_and_touches_nothing():
    """Shadow mode's entire value is the road not taken, so it must be stored.

    A shadow decision row that said `no_action` would be worthless -- it would look
    identical to a live deployment that decided to do nothing. The row records what
    the gate *chose*; the response records what was *done*.
    """
    h = Harness(mode=Mode.SHADOW, decider=StaticDecider(repair_verdict()),
                actor=_SendingActor())
    body = h.post_order().json()

    assert body["suppressed"] is True
    assert body["action"] == ActionKind.NO_ACTION.value
    assert body["recommended"] == ActionKind.ADDRESS_REPAIR.value

    stored = DecisionRepo(h.tenant).latest(body["order_id"])
    assert stored is not None
    assert stored.action is ActionKind.ADDRESS_REPAIR      # the counterfactual
    assert AuditKind.DECISION_SUPPRESSED in h.audit_kinds(body["order_id"])

    # And nothing was spent or sent.
    assert ActionRepo(h.tenant).for_order(body["order_id"]) == []
    assert h.tenant.db.scalar(
        "SELECT COUNT(*) FROM ledger_entries WHERE merchant_id = :merchant_id",
        h.tenant.scope(), default=0) == 0


def test_a_held_order_releases_when_the_buyer_replies(gateway):
    """The happy path: hold, buyer corrects the address, release."""
    order_id = gateway.post_order(address=DEFECTIVE).json()["order_id"]
    assert gateway.state(order_id) is OrderState.AWAITING_BUYER

    corrected = dict(ADDRESS)
    response = gateway.client.post(
        "/v1/orders/%s/resolve" % order_id,
        json={"resolution": "address_corrected", "address": corrected,
              "detail": "Flat 402, not the temple"},
        headers=gateway.headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["release"] is True
    assert gateway.state(order_id) is OrderState.RELEASED_FOR_DISPATCH

    # The correction is attributed to the buyer, not to us. `buyer_confirmed` and
    # `patched` are different claims and the console shows the difference.
    revisions = gateway.get_order(order_id).json()["address_revisions"]
    assert [r["source"] for r in revisions] == ["submitted", "buyer_confirmed"]
    assert "Ashirvad" in revisions[-1]["text"]


def test_address_corrected_without_an_address_is_refused(gateway):
    """A repair that changed nothing must not be recorded as a repair."""
    order_id = gateway.post_order(address=DEFECTIVE).json()["order_id"]
    response = gateway.client.post("/v1/orders/%s/resolve" % order_id,
                                   json={"resolution": "address_corrected"},
                                   headers=gateway.headers)
    assert response.status_code == 422
    assert "corrected address" in response.json()["detail"]
    assert gateway.state(order_id) is OrderState.AWAITING_BUYER


def test_the_merchant_can_always_override_a_hold(gateway):
    order_id = gateway.post_order(address=DEFECTIVE).json()["order_id"]
    response = gateway.client.post("/v1/orders/%s/release" % order_id,
                                   json={"reason": "packing this one anyway"},
                                   headers=gateway.headers)
    assert response.status_code == 200
    assert response.json()["release"] is True
    assert gateway.state(order_id) is OrderState.RELEASED_FOR_DISPATCH


def test_a_released_order_cannot_be_cancelled(gateway):
    """Once released, the parcel is the carrier's. Telling the merchant otherwise
    would report a stop that did not happen."""
    order_id = gateway.post_order(address=DEFECTIVE).json()["order_id"]
    gateway.client.post("/v1/orders/%s/release" % order_id,
                        json={"reason": "manual"}, headers=gateway.headers)
    response = gateway.client.post("/v1/orders/%s/cancel" % order_id,
                                   json={"reason": "changed my mind"},
                                   headers=gateway.headers)
    assert response.status_code == 409
    assert "already released" in response.json()["detail"]


def test_a_blocked_channel_releases_rather_than_phantom_holding():
    """No channel means no message, which means nothing to wait for.

    The plausible bug here is to hold the order anyway -- the gate did recommend a
    repair, after all. That would put a parcel in a four-hour queue waiting for a
    reply to a message that was never sent, which is the single most expensive way
    to be wrong in this system.
    """
    h = Harness(mode=Mode.GATEWAY, decider=StaticDecider(repair_verdict()))
    body = h.post_order().json()

    assert body["release"] is True
    assert body["held_until"] is None
    assert body["blocked_reason"] == "no_channel"
    assert body["recommended"] == ActionKind.ADDRESS_REPAIR.value

    action = ActionRepo(h.tenant).for_order(body["order_id"])[0]
    assert action.status == "blocked"
    assert action.blocked_reason == "no_channel"
    assert action.spent is False        # a blocked action costs nothing


def test_no_consent_means_no_message_and_the_refusal_is_recorded():
    h = Harness(mode=Mode.GATEWAY, decider=StaticDecider(repair_verdict()),
                actor=_SendingActor())
    body = h.post_order(consent={}).json()

    assert body["release"] is True
    assert body["blocked_reason"] == "no_consent"
    assert AuditKind.ACTION_BLOCKED in h.audit_kinds(body["order_id"])


def test_consent_for_a_channel_the_merchant_does_not_use_is_not_consent():
    """Consent is per channel, and this merchant does not send on WhatsApp.

    The tempting shortcut is a single boolean "contactable" flag. It is wrong: a
    buyer who ticked WhatsApp at checkout has not agreed to an SMS, and DPDP
    consent is specific to the purpose and the medium it was collected for.
    """
    h = Harness(mode=Mode.GATEWAY, decider=StaticDecider(repair_verdict()),
                actor=_SendingActor())
    assert h.tenant.config.default_channel is Channel.SIMULATED
    body = h.post_order(consent={"whatsapp": True, "sms": True}).json()

    assert body["blocked_reason"] == "no_consent"
    assert body["release"] is True


# ==========================================================================
# 2. the max_hold_minutes timeout path
# ==========================================================================

def test_the_hold_never_outlives_max_hold_minutes(gateway):
    """The timer. Nobody replies, and the parcel ships anyway.

    Drives the clock past the cap and then the same `sweep` the background task
    calls. The assertion that matters is not that the order released -- it is that it
    released with `actor="timeout"` in the audit trail, so an operator can tell an
    auto-release from a decision.
    """
    order_id = gateway.post_order(address=DEFECTIVE).json()["order_id"]
    assert gateway.state(order_id) is OrderState.AWAITING_BUYER

    held_until = gateway.get_order(order_id).json()["held_until"]
    assert held_until is not None

    # Still held at the cap.
    gateway.clock.advance(minutes=gateway.tenant.config.max_hold_minutes - 1)
    assert gateway.sweep().json()["released"] == 0
    assert gateway.state(order_id) is OrderState.AWAITING_BUYER

    # Past it, and the sweep lets go.
    gateway.clock.advance(minutes=2)
    swept = gateway.sweep().json()
    assert swept["released"] == 1
    assert swept["orders"][0]["order_id"] == order_id
    assert swept["orders"][0]["release"] is True
    assert gateway.state(order_id) is OrderState.RELEASED_FOR_DISPATCH
    assert AuditKind.ORDER_AUTO_RELEASED in gateway.audit_kinds(order_id)


def test_the_hold_window_is_capped_by_config_not_by_the_verdict():
    """A verdict asking for a week-long hold gets `max_hold_minutes`.

    The reply window is a merchant setting, not a model output. If the gate could
    extend a hold, then a miscalibrated model could stall a dispatch queue, and the
    fulfilment guarantee would depend on the model being correct -- which is exactly
    the dependency this design refuses.
    """
    greedy = dataclasses.replace(repair_verdict(), hold_minutes=10_000)
    h = Harness(mode=Mode.GATEWAY, decider=StaticDecider(greedy),
                actor=_SendingActor(), max_hold_minutes=240)
    order_id = h.post_order(address=DEFECTIVE).json()["order_id"]

    held_until = OrderRepo(h.tenant).get(order_id).held_until
    assert held_until == "2026-03-01T13:00:00Z"       # 09:00Z + 240 minutes


def test_a_reply_arriving_after_release_does_not_move_the_order_backwards():
    """Advisory mode released already. The reply is kept; the state does not rewind.

    A parcel may already be on a van. Moving it back to `repaired` would tell the
    merchant an intervention succeeded in time when it did not, and would corrupt the
    only number the backtest cares about.
    """
    h = Harness(mode=Mode.ADVISORY, decider=StaticDecider(repair_verdict()),
                actor=_SendingActor())
    order_id = h.post_order(address=DEFECTIVE).json()["order_id"]
    assert h.state(order_id) is OrderState.RELEASED_FOR_DISPATCH

    response = h.client.post("/v1/orders/%s/resolve" % order_id,
                             json={"resolution": "address_corrected",
                                   "address": dict(ADDRESS)},
                             headers=h.headers)
    assert response.status_code == 200
    assert response.json()["release"] is True
    assert h.state(order_id) is OrderState.RELEASED_FOR_DISPATCH
    # The correction was still stored -- it helps this buyer's next order.
    revisions = h.get_order(order_id).json()["address_revisions"]
    assert revisions[-1]["source"] == "buyer_confirmed"


# ==========================================================================
# 3. an internal exception releases the order
# ==========================================================================

def test_a_broken_decider_releases_the_order_and_says_so():
    """Fail open, loudly.

    Three things must all be true: the caller gets a 200 with `release: true`, the
    order is genuinely in `released_for_dispatch` in the database, and the audit log
    carries `decision.failed_open` so this order is never counted as one the agent
    chose to leave alone. The third is the one a naive implementation drops, and it
    is the difference between a measured failure and an invisible one.
    """
    h = Harness(mode=Mode.GATEWAY, decider=_BrokenDecider(),
                actor=_SendingActor())
    response = h.post_order()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["release"] is True
    assert body["degraded"] is True
    assert body["held_until"] is None
    assert h.state(body["order_id"]) is OrderState.RELEASED_FOR_DISPATCH
    assert AuditKind.DECISION_FAILED_OPEN in h.audit_kinds(body["order_id"])

    # The failure did not corrupt the trail it wrote.
    assert h.tenant.audit.verify(h.tenant.merchant_id).ok


def test_fail_open_reports_the_order_override_mode():
    h = Harness(mode=Mode.GATEWAY, decider=_BrokenDecider())
    response = h.post_order(mode="advisory")

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert body["mode"] == Mode.ADVISORY.value



def test_the_fail_open_does_not_leak_the_exception_text():
    """A stack trace in an API response is an information leak with a 200 status."""
    h = Harness(mode=Mode.GATEWAY, decider=_BrokenDecider())
    body = h.post_order().json()
    assert "model server unreachable" not in str(body).lower()
    assert "traceback" not in str(body).lower()


def test_every_error_body_tells_the_caller_to_print_the_label():
    """The integration contract, asserted on the errors rather than the successes.

    A gateway-mode integration blocked on a non-2xx with no instruction has to guess,
    and the guess that costs money is "wait". So `release: true` is on every error
    shape this API can produce.
    """
    h = Harness()
    cases = [
        h.client.post("/v1/orders", json={}),                       # 401
        h.client.post("/v1/orders", json={}, headers=h.headers),    # 422
        h.client.get("/v1/orders/ord_nope", headers=h.headers),     # 404
        h.client.get("/v1/orders/ord_nope"),                        # 401
    ]
    for response in cases:
        assert response.status_code >= 400
        body = response.json()
        assert body["release"] is True, (response.status_code, body)
        assert body["error"]


# ==========================================================================
# authentication and isolation
# ==========================================================================

def test_every_authentication_failure_looks_identical():
    """No oracle. Unknown, malformed, revoked and missing all answer the same.

    A 401 that distinguishes "no such key" from "revoked key" is an enumeration
    tool: it tells an attacker which of their guesses was once real.
    """
    h = Harness()
    revoked_plaintext, record = tenancy.issue_api_key(h.tenant, mode="test")
    tenancy.revoke_api_key(h.tenant, record.id)

    attempts = [
        {},
        {"Authorization": "Bearer pk_test_deadbeef"},
        {"Authorization": "Basic pk_test_deadbeef"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer %s" % revoked_plaintext},
        {"X-API-Key": "pk_test_deadbeef"},
    ]
    bodies = set()
    for headers in attempts:
        response = h.client.get("/v1/orders", headers=headers)
        assert response.status_code == 401, headers
        bodies.add(response.text)
    assert len(bodies) == 1, bodies
    assert UNAUTHORIZED in bodies.pop()


def test_x_api_key_header_works_too():
    h = Harness()
    response = h.client.get("/v1/orders", headers={"X-API-Key": h.key})
    assert response.status_code == 200


def test_one_merchants_key_cannot_read_anothers_order():
    """Tenant isolation at the socket, not just at the query layer.

    The order id is supplied verbatim in the path and is a valid id -- just not
    theirs. A 404 rather than a 403, because confirming the id exists is itself the
    leak.
    """
    a = Harness()
    b_tenant = tenancy.provision_merchant(
        a.db, a.runtime, "Rival Traders", merchant_id="rival_traders",
        config=MerchantConfig(merchant_id="rival_traders",
                              monthly_incentive_budget_paise=0,
                              origin_pincode="110001"))
    b_key, _ = tenancy.issue_api_key(b_tenant, mode="test")

    order_id = a.post_order().json()["order_id"]
    assert a.get_order(order_id).status_code == 200

    stolen = a.client.get("/v1/orders/%s" % order_id,
                          headers={"Authorization": "Bearer %s" % b_key})
    assert stolen.status_code == 404
    assert stolen.json()["error"] == "not_found"


def test_health_is_unauthenticated_and_discloses_synthetic_data():
    h = Harness()
    body = h.client.get("/health").json()
    assert body["status"] == "ok"
    assert body["synthetic_data"] is True        # PRD section 13
    assert body["schema_version"] == 1
    # And says nothing about the tenants.
    assert "kolam" not in str(body).lower()


def test_every_response_carries_a_latency_measurement():
    h = Harness()
    response = h.client.get("/health")
    assert int(response.headers["X-Response-Time-Ms"]) >= 0


# ==========================================================================
# money at the boundary
# ==========================================================================

def test_a_json_float_for_money_is_refused():
    """1200.00 as a JSON number becomes a float, and a float becomes 1199.9999...

    `money.to_paise` refuses floats at the bottom of the stack. This refuses them at
    the top, so the error names the field instead of surfacing as a `TypeError`.
    """
    h = Harness()
    response = h.client.post(
        "/v1/orders",
        json={"phone": "+919876543210", "address": ADDRESS, "value": 1200.00},
        headers=h.headers)
    assert response.status_code == 422
    assert "value" in response.json()["detail"]


def test_both_money_fields_at_once_is_refused():
    h = Harness()
    response = h.client.post(
        "/v1/orders",
        json={"phone": "+919876543210", "address": ADDRESS,
              "value": "1200.00", "value_paise": 120000},
        headers=h.headers)
    assert response.status_code == 422
    assert "exactly one" in response.json()["detail"]


def test_integer_paise_are_accepted_verbatim():
    h = Harness()
    body = h.client.post(
        "/v1/orders",
        json={"phone": "+919876543210", "address": ADDRESS,
              "value_paise": 120000},
        headers=h.headers).json()
    assert OrderRepo(h.tenant).get(body["order_id"]).value_paise == 120_000


def test_a_misspelled_field_is_refused_rather_than_ignored():
    """A dropped `pincode` is a defective address delivered on purpose."""
    h = Harness()
    response = h.client.post(
        "/v1/orders",
        json={"phone": "+919876543210", "address": ADDRESS, "value": "1200.00",
              "pincode": "560034"},
        headers=h.headers)
    assert response.status_code == 422
    assert "pincode" in response.json()["detail"]


def test_an_empty_address_is_a_missing_field_not_a_defect():
    h = Harness()
    response = h.client.post(
        "/v1/orders",
        json={"phone": "+919876543210", "address": {}, "value": "1200.00"},
        headers=h.headers)
    assert response.status_code == 422
    assert "address" in response.json()["detail"]


# ==========================================================================
# idempotency
# ==========================================================================

def test_a_replayed_idempotency_key_returns_the_original_decision():
    """A retried order feed must not produce two orders or two decisions."""
    h = Harness(mode=Mode.GATEWAY, decider=StaticDecider(repair_verdict()),
                actor=_SendingActor())
    first = h.post_order(address=DEFECTIVE, key="feed-1").json()
    second = h.post_order(address=DEFECTIVE, key="feed-1").json()

    assert first == second
    assert h.tenant.db.scalar(
        "SELECT COUNT(*) FROM orders WHERE merchant_id = :merchant_id",
        h.tenant.scope(), default=0) == 1


def test_a_duplicate_external_id_is_a_conflict_not_a_second_order():
    """Without an idempotency key, the merchant's own reference is the backstop."""
    h = Harness()
    first = h.post_order(external_id="SHOP-1")
    assert first.status_code == 200
    second = h.post_order(external_id="SHOP-1")
    assert second.status_code == 409
    assert second.json()["release"] is True


def test_the_same_key_with_a_different_body_is_a_conflict():
    """A reused key with changed contents is a bug in the caller, not a retry.

    Both silent answers are wrong. Returning the first response drops the second
    order; ignoring the key creates a duplicate the merchant believes was suppressed.
    A 409 surfaces the caller's bug without losing their order -- they retry under a
    correct key.
    """
    h = Harness()
    assert h.post_order(external_id="SHOP-1", key="feed-1").status_code == 200
    response = h.post_order(external_id="SHOP-2", key="feed-1", value="999.00")
    assert response.status_code == 409
    assert "already used for a different request" in response.json()["detail"]
    assert response.json()["release"] is True
    assert h.tenant.db.scalar(
        "SELECT COUNT(*) FROM orders WHERE merchant_id = :merchant_id",
        h.tenant.scope(), default=0) == 1


# ==========================================================================
# outcomes and the ledger
# ==========================================================================

def test_a_delivered_outcome_posts_the_prd_arithmetic():
    """The number from PRD section 3, over HTTP: a delivered Rs 1,200 order is +265.

    Asserted as an itemised breakdown rather than a total, because a total that
    happens to be right for the wrong reasons is the failure mode that survives a
    refactor.
    """
    h = Harness()
    order_id = h.post_order(external_id="SHOP-1").json()["order_id"]
    response = h.client.post("/v1/outcomes",
                             json={"kind": "delivered", "order_id": order_id},
                             headers=h.headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["net_paise"] == 26_500
    assert body["net_rupees"] == "+₹265.00"
    lines = {line["reason"]: line["amount_paise"] for line in body["lines"]}
    assert lines == {"gross_margin": 36_000, "packaging": -2_500,
                     "forward_freight": -7_000}


def test_an_rto_outcome_posts_the_prd_loss():
    """The other half of the asymmetry: the same order returning costs Rs 195."""
    h = Harness()
    order_id = h.post_order().json()["order_id"]
    body = h.client.post("/v1/outcomes",
                         json={"kind": "rto", "order_id": order_id},
                         headers=h.headers).json()
    assert body["net_paise"] == -19_500
    assert body["net_rupees"] == "-₹195.00"
    assert h.state(order_id) is OrderState.RTO


def test_an_outcome_can_be_reported_against_the_merchants_own_reference():
    h = Harness()
    h.post_order(external_id="SHOP-77")
    response = h.client.post("/v1/outcomes",
                             json={"kind": "delivered",
                                   "external_id": "SHOP-77"},
                             headers=h.headers)
    assert response.status_code == 200
    assert response.json()["net_paise"] == 26_500


def test_a_contradictory_outcome_is_refused():
    """Delivered then RTO is not a correction, it is a contradiction.

    Overwriting would silently rewrite the merchant's margin. A reversal is posted
    as a `correct` ledger entry with its own reason, which leaves both facts visible.
    """
    h = Harness()
    order_id = h.post_order().json()["order_id"]
    h.client.post("/v1/outcomes", json={"kind": "delivered",
                                        "order_id": order_id},
                  headers=h.headers)
    response = h.client.post("/v1/outcomes", json={"kind": "rto",
                                                   "order_id": order_id},
                             headers=h.headers)
    assert response.status_code == 409
    assert "contradicts" in response.json()["detail"]


def test_repeating_the_same_outcome_is_a_no_op():
    h = Harness()
    order_id = h.post_order().json()["order_id"]
    payload = {"kind": "delivered", "order_id": order_id}
    first = h.client.post("/v1/outcomes", json=payload, headers=h.headers)
    second = h.client.post("/v1/outcomes", json=payload, headers=h.headers)
    assert first.json() == second.json()
    assert h.tenant.db.scalar(
        "SELECT COUNT(*) FROM outcomes WHERE merchant_id = :merchant_id",
        h.tenant.scope(), default=0) == 1


def test_an_unknown_outcome_kind_lists_the_legal_ones():
    h = Harness()
    order_id = h.post_order().json()["order_id"]
    response = h.client.post("/v1/outcomes",
                             json={"kind": "eaten_by_a_dog",
                                   "order_id": order_id},
                             headers=h.headers)
    assert response.status_code == 422
    detail = response.json()["detail"]
    for kind in ("delivered", "rto", "cancelled", "prepaid_delivered"):
        assert kind in detail


def test_an_outcome_for_a_still_held_order_fills_in_the_release_we_never_recorded():
    """The carrier delivered an order we thought we were still holding.

    Real feeds skip events, and a merchant may pack a held order by hand. Refusing
    the outcome would lose the only label the model can learn from; jumping straight
    to `delivered` would leave a timeline with no release in it. So the gap is walked
    and the inferred step says it was inferred, which keeps "what we were told" and
    "what we filled in" distinguishable.
    """
    h = Harness(mode=Mode.GATEWAY, decider=StaticDecider(repair_verdict()),
                actor=_SendingActor())
    order_id = h.post_order(address=DEFECTIVE).json()["order_id"]
    assert h.state(order_id) is OrderState.AWAITING_BUYER

    h.client.post("/v1/outcomes", json={"kind": "delivered",
                                        "order_id": order_id},
                  headers=h.headers)

    transitions = h.get_order(order_id).json()["transitions"]
    inferred = [t for t in transitions if "inferred" in t["reason"]]
    assert [t["to"] for t in inferred] == [
        OrderState.RELEASED_FOR_DISPATCH.value]
    assert transitions[-1]["to"] == OrderState.DELIVERED.value
    assert transitions[-1]["reason"] == "outcome reported: delivered"


def test_an_outcome_requires_exactly_one_order_identifier():
    h = Harness()
    order_id = h.post_order(external_id="SHOP-EXACTLY-ONE").json()["order_id"]

    for payload in (
        {"kind": "delivered"},
        {"kind": "delivered", "order_id": " "},
        {"kind": "delivered", "order_id": order_id,
         "external_id": "SHOP-EXACTLY-ONE"},
    ):
        response = h.client.post("/v1/outcomes", json=payload,
                                 headers=h.headers)
        assert response.status_code == 422
        assert "exactly one" in str(response.json()["detail"])


# ==========================================================================
# the audit trail survives everything above
# ==========================================================================

def test_the_hash_chain_verifies_after_a_full_lifecycle(gateway):
    """One order through hold, reply, release and outcome, then verify the chain.

    The chain is the artefact that makes every other claim in this system checkable.
    If a normal lifecycle breaks it, nothing else here is worth reading.
    """
    order_id = gateway.post_order(address=DEFECTIVE,
                                  external_id="SHOP-9").json()["order_id"]
    gateway.client.post("/v1/orders/%s/resolve" % order_id,
                        json={"resolution": "address_corrected",
                              "address": dict(ADDRESS)},
                        headers=gateway.headers)
    gateway.client.post("/v1/outcomes",
                        json={"kind": "delivered", "order_id": order_id},
                        headers=gateway.headers)

    verification = gateway.tenant.audit.verify(gateway.tenant.merchant_id)
    assert verification.ok, verification.summary()
    assert verification.entries_checked > 10

    kinds = gateway.audit_kinds(order_id)
    for expected in (AuditKind.ORDER_RECEIVED, AuditKind.DECISION_MADE,
                     AuditKind.ACTION_ATTEMPTED, AuditKind.ORDER_HELD,
                     AuditKind.ADDRESS_PATCHED, AuditKind.ORDER_RELEASED,
                     AuditKind.OUTCOME_REPORTED):
        assert expected in kinds, expected


def test_the_open_api_schema_is_generated():
    """A merchant integrating against this reads `/docs`, not our source."""
    h = Harness()
    spec = h.client.get("/openapi.json").json()
    assert "/v1/orders" in spec["paths"]
    assert "/v1/outcomes" in spec["paths"]
    assert "release" in spec["components"]["schemas"]["DecisionOut"]["properties"]
