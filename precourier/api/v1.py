"""The v1 API. Five endpoints, one of which is the product.

`POST /v1/orders` is the product. Everything else exists to support it: outcomes
supply the labels that make the model honest, resolutions close holds, and the read
endpoints let a human check the machine's working.

Every mutating route holds `AppState.lock` for the whole of its database work and
runs inside `Idempotent`, in that order. Taking the lock first matters: two
concurrent retries of the same key must not both miss the lookup and both create an
order.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from .. import states
from ..domain import Address, Channel, Mode, OutcomeKind
from ..config import MerchantConfig
from ..geo import REFERENCE
from ..ledger import breakdown as ledger_breakdown
from ..money import format_paise, to_bps, to_paise
from ..verify import verify_address_patch
from ..reporting import roi_report
from ..historical import HistoricalImportError, project_csv
from ..store.repo import (ActionRepo, BuyerRepo, Conflict, DecisionRepo,
                          MessageRepo, NotFound, OrderRepo, OutcomeRepo,
                          ReasoningRepo)
from ..tenancy import (ApiKeyRecord, Tenant, issue_api_key, list_api_keys,
                       save_config, revoke_api_key)
from . import schemas
from .deps import (HTTP_422, Idempotent, get_tenant, idempotency_key,
                   state_of)

router = APIRouter(prefix="/v1", tags=["v1"])


def _config_out(tenant: Tenant) -> Dict[str, Any]:
    config = tenant.config
    return schemas.ConfigOut(
        merchant_id=tenant.merchant_id, revision=tenant.config_revision,
        mode=config.mode.value, default_channel=config.default_channel.value,
        gross_margin_bps=config.gross_margin_bps,
        packaging_paise=config.packaging_paise,
        forward_freight_paise=config.forward_freight_paise,
        failed_attempt_fee_paise=config.failed_attempt_fee_paise,
        return_freight_paise=config.return_freight_paise,
        goodwill_cost_paise=config.goodwill_cost_paise,
        gateway_fee_bps=config.gateway_fee_bps,
        max_incentive_paise=config.max_incentive_paise,
        monthly_incentive_budget_paise=config.monthly_incentive_budget_paise,
        max_hold_minutes=config.max_hold_minutes,
        reply_window_hours=config.reply_window_hours,
        decision_latency_budget_ms=config.decision_latency_budget_ms,
        max_paid_actions_per_order=config.max_paid_actions_per_order,
        max_messages_per_buyer_24h=config.max_messages_per_buyer_24h,
        holdout_rate_percent=config.holdout_rate_percent).model_dump()


def _key_out(record: ApiKeyRecord, plaintext: Optional[str] = None) -> Dict[str, Any]:
    return schemas.ApiKeyOut(
        id=record.id, key_prefix=record.key_prefix, mode=record.mode,
        label=record.label, created_at=record.created_at,
        revoked_at=record.revoked_at, plaintext=plaintext).model_dump()


def _updated_config(tenant: Tenant, payload: schemas.ConfigUpdateIn) -> MerchantConfig:
    current = tenant.config
    changes: Dict[str, Any] = {}
    money_fields = {
        "packaging": "packaging_paise", "forward_freight": "forward_freight_paise",
        "failed_attempt_fee": "failed_attempt_fee_paise",
        "return_freight": "return_freight_paise", "goodwill_cost": "goodwill_cost_paise",
        "max_incentive": "max_incentive_paise",
        "monthly_incentive_budget": "monthly_incentive_budget_paise",
        "value_review_threshold": "value_review_threshold_paise",
    }
    for source, target in money_fields.items():
        value = getattr(payload, source)
        if value is not None:
            changes[target] = to_paise(value)
    if payload.gross_margin_rate is not None:
        changes["gross_margin_bps"] = to_bps(payload.gross_margin_rate)
    if payload.gateway_fee_rate is not None:
        changes["gateway_fee_bps"] = to_bps(payload.gateway_fee_rate)
    direct = (
        "auto_cancel_enabled", "max_hold_minutes", "reply_window_hours",
        "decision_latency_budget_ms", "quiet_hours_start", "quiet_hours_end",
        "tz_offset_minutes", "max_paid_actions_per_order",
        "max_messages_per_buyer_24h", "holdout_rate_percent", "origin_pincode")
    for name in direct:
        value = getattr(payload, name)
        if value is not None:
            changes[name] = value
    if payload.mode is not None:
        changes["mode"] = _enum(Mode, payload.mode, "mode")
    if payload.default_channel is not None:
        changes["default_channel"] = _enum(Channel, payload.default_channel,
                                            "default_channel")
    return current.replace(**changes).validate()


# ==========================================================================
# helpers
# ==========================================================================

def _address(payload: schemas.AddressIn) -> Address:
    return Address(raw=payload.raw, line1=payload.line1, line2=payload.line2,
                   landmark=payload.landmark, locality=payload.locality,
                   city=payload.city, state=payload.state,
                   pincode=payload.pincode)


def _enum(cls: Any, value: str, field: str) -> Any:
    """Parse an enum member, or 422 with the legal values spelled out.

    Pydantic could do this with a `Literal`, but then adding a state to the domain
    would require editing the schema too, and the two would drift. Reading the legal
    values off the enum means the API's contract is derived from the domain rather
    than restated alongside it.
    """
    try:
        return cls(value)
    except ValueError:
        raise HTTPException(
            HTTP_422,
            "%s must be one of: %s" % (
                field, ", ".join(sorted(m.value for m in cls))))


def _decision_out(d: states.Disposition) -> Dict[str, Any]:
    explanation = dict(d.explanation or {})
    chosen_action = str(explanation.get("chosen_action", "no_action"))
    runner_up = str(explanation.get("runner_up", ""))
    expected_saving = int(explanation.get("expected_saving_paise", 0))
    top_drivers = list(explanation.get("top_drivers", []))
    return schemas.DecisionOut(
        order_id=d.order_id, state=d.state.value, mode=d.mode.value,
        release=d.release, held_until=d.held_until, action=d.action.value,
        recommended=d.recommended.value, suppressed=d.suppressed,
        blocked_reason=d.blocked_reason, p_rto_ppm=d.p_rto_ppm,
        threshold_ppm=d.threshold_ppm, cause=d.cause.value, reason=d.reason,
        decision_id=d.decision_id, action_id=d.action_id,
        degraded=d.degraded, chosen_action=chosen_action,
        runner_up=runner_up, expected_saving_paise=expected_saving,
        top_drivers=top_drivers, holdout=bool(explanation.get("holdout", False)),
        explanation=explanation).model_dump()


def _resolve_order_id(tenant: Tenant, order_id: Optional[str],
                      external_id: Optional[str]) -> str:
    if order_id:
        return order_id
    order = OrderRepo(tenant).by_external_id(external_id or "")
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "no order with that reference")
    return order.id


# ==========================================================================
# the decision endpoint
# ==========================================================================

@router.post("/orders", response_model=schemas.DecisionOut,
             summary="Submit an order and get a release decision")
def create_order(request: Request, payload: schemas.OrderIn,
                 tenant: Tenant = Depends(get_tenant),
                 key: Optional[str] = Depends(idempotency_key)
                 ) -> Dict[str, Any]:
    """Decide whether this order's label may be printed.

    In gateway mode a `release: false` response means hold the label until
    `held_until`, or until `/v1/orders/{id}` says otherwise. In advisory and shadow
    modes the response is always `release: true` -- those modes never block
    fulfilment, by definition.

    **Integration contract:** if this endpoint returns a non-2xx status, or times
    out, print the label. Every error body carries `release: true` for exactly that
    reason. A system that stops a merchant shipping when it is broken is worse than
    no system.
    """
    app_state = state_of(request)
    mode = _enum(Mode, payload.mode, "mode") if payload.mode else None

    with app_state.lock:
        with Idempotent(tenant, "POST /v1/orders", key,
                        payload.model_dump(mode="json")) as slot:
            if slot.replayed:
                return slot.response

            consent = {_enum(Channel, name, "consent channel"): granted
                       for name, granted in payload.consent.items()}

            service = app_state.service(tenant)
            try:
                disposition = service.accept(
                    phone=payload.phone, email=payload.email,
                    address=_address(payload.address),
                    value_paise=payload.paise(), external_id=payload.external_id,
                    placed_at=payload.placed_at,
                    payment_method=payload.payment_method,
                    item_count=payload.item_count, category=payload.category,
                    discount_bps=payload.discount_bps, mode=mode,
                    consent=consent)
            except Conflict as exc:
                # A duplicate `external_id`. Not fail-open territory: the order
                # already exists and already has a decision, so the right answer is
                # to point at it rather than to create a second one.
                raise HTTPException(status.HTTP_409_CONFLICT, str(exc))

            body = _decision_out(disposition)
            slot.store(body)
            return body


@router.get("/orders/{order_id}", response_model=schemas.OrderOut,
            summary="An order, its history, and every decision made about it")
def get_order(request: Request, order_id: str,
              tenant: Tenant = Depends(get_tenant)) -> Dict[str, Any]:
    """The full paper trail for one order.

    Returns the address history rather than just the current address, because the
    question a merchant actually asks is "what did you change and on whose word",
    and a single current address cannot answer it.
    """
    app_state = state_of(request)
    with app_state.lock:
        orders = OrderRepo(tenant)
        try:
            order = orders.get(order_id)
        except NotFound:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")

        revisions = orders.address_history(order_id)
        transitions = tenant.db.all(
            "SELECT from_state, to_state, reason, created_at "
            "FROM order_state_transitions "
            "WHERE merchant_id = :merchant_id AND order_id = :o "
            "ORDER BY created_at, id", tenant.scope(o=order_id))
        decisions = DecisionRepo(tenant).for_order(order_id)
        actions = ActionRepo(tenant).for_order(order_id)
        messages = MessageRepo(tenant).for_order(order_id)
        reasoning = ReasoningRepo(tenant).for_order(order_id)
        outcome = OutcomeRepo(tenant).find(order_id)

        return schemas.OrderOut(
            order_id=order.id, external_id=order.external_id,
            state=order.state.value, mode=order.mode.value,
            value_paise=order.value_paise,
            payment_method=order.payment_method, placed_at=order.placed_at,
            held_until=order.held_until,
            address={"text": order.address.as_text(),
                     "pincode": order.address.pincode,
                     "city": order.address.city},
            address_revisions=[
                {"revision": r.revision, "source": r.source,
                 "text": r.address.as_text(), "created_at": r.created_at}
                for r in revisions],
            transitions=[
                {"from": str(r["from_state"]), "to": str(r["to_state"]),
                 "reason": str(r["reason"]), "at": str(r["created_at"])}
                for r in transitions],
            decisions=[
                {"id": d.id, "action": d.action.value, "reason": d.reason,
                 "threshold_ppm": d.threshold_ppm,
                  "expected_delta_micro": d.expected_delta_micro,
                  "gate_inputs": d.gate_inputs,
                  "config_revision": d.config_revision,
                  "chosen_action": d.explanation.get("chosen_action", ""),
                  "runner_up": d.explanation.get("runner_up", ""),
                  "expected_saving_paise": d.explanation.get(
                      "expected_saving_paise", 0),
                  "top_drivers": d.explanation.get("top_drivers", []),
                  "holdout": d.explanation.get("holdout", False),
                  "explanation": d.explanation, "at": d.created_at}
                for d in decisions],
            features=reasoning["features"],
            risk_scores=reasoning["risk_scores"],
            diagnoses=reasoning["diagnoses"],
            actions=[
                {"id": a.id, "kind": a.kind.value, "status": a.status,
                 "blocked_reason": a.blocked_reason,
                 "cost_paise": a.cost_paise,
                 "incentive_paise": a.incentive_paise, "at": a.created_at}
                for a in actions],
            messages=messages,
            outcome=None if outcome is None else {
                "kind": outcome.kind.value, "reported_at": outcome.reported_at,
                "carrier_reason": outcome.carrier_reason,
                "attempts": outcome.attempts}).model_dump()


@router.post("/orders/{order_id}/resolve", response_model=schemas.DecisionOut,
             summary="Close a hold: the buyer replied, or a human decided")
def resolve_order(request: Request, order_id: str,
                  payload: schemas.ResolutionIn,
                  tenant: Tenant = Depends(get_tenant),
                  key: Optional[str] = Depends(idempotency_key)
                  ) -> Dict[str, Any]:
    app_state = state_of(request)
    resolution = _enum(states.Resolution, payload.resolution, "resolution")
    with app_state.lock:
        with Idempotent(tenant, "POST /v1/orders/resolve", key,
                        {"order_id": order_id,
                         **payload.model_dump(mode="json")}) as slot:
            if slot.replayed:
                return slot.response
            service = app_state.service(tenant)
            try:
                disposition = service.resolve(
                    order_id, resolution,
                    address=_address(payload.address) if payload.address else None,
                    detail=payload.detail)
            except NotFound:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")
            except ValueError as exc:
                raise HTTPException(HTTP_422, str(exc))
            body = _decision_out(disposition)
            slot.store(body)
            return body


@router.post("/orders/{order_id}/messages/inbound",
             response_model=schemas.DecisionOut,
             summary="Record a buyer reply and optionally resolve an address hold")
def inbound_message(request: Request, order_id: str,
                    payload: schemas.InboundMessageIn,
                    tenant: Tenant = Depends(get_tenant),
                    key: Optional[str] = Depends(idempotency_key)
                    ) -> Dict[str, Any]:
    """Persist a buyer reply before applying its verified structured proposal."""
    app_state = state_of(request)
    channel = _enum(Channel, payload.channel, "channel")
    with app_state.lock:
        with Idempotent(tenant, "POST /v1/orders/messages/inbound", key,
                        {"order_id": order_id,
                         **payload.model_dump(mode="json")}) as slot:
            if slot.replayed:
                return slot.response
            orders = OrderRepo(tenant)
            try:
                order = orders.get(order_id)
                MessageRepo(tenant).record_inbound(
                    order.id, order.buyer_id, channel=channel,
                    body=payload.body)
                if payload.address is not None:
                    verification = verify_address_patch(
                        order.address, _address(payload.address),
                        reference=REFERENCE)
                    if not verification.accepted:
                        tenant.audit.append(
                            tenant.merchant_id,
                            "address.patch_rejected",
                            {"order_id": order_id, "code": verification.code,
                             "detail": verification.detail},
                            order_id=order_id, actor="verifier")
                        raise HTTPException(
                            HTTP_422, "%s: %s" % (verification.code,
                                                   verification.detail))
                resolution = (states.Resolution.ADDRESS_CORRECTED
                              if payload.address is not None
                              else states.Resolution.ADDRESS_CONFIRMED)
                disposition = app_state.service(tenant).resolve(
                    order_id, resolution,
                    address=_address(payload.address)
                    if payload.address is not None else None,
                    detail=payload.detail or payload.body)
            except NotFound:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")
            except ValueError as exc:
                raise HTTPException(HTTP_422, str(exc))
            body = _decision_out(disposition)
            slot.store(body)
            return body


@router.get("/orders/{order_id}/messages",
            response_model=list[schemas.MessageOut],
            summary="Read the persisted buyer conversation")
def list_messages(request: Request, order_id: str,
                  tenant: Tenant = Depends(get_tenant)) -> Any:
    app_state = state_of(request)
    with app_state.lock:
        try:
            OrderRepo(tenant).get(order_id)
        except NotFound:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")
        return [schemas.MessageOut(**row).model_dump()
                for row in MessageRepo(tenant).for_order(order_id)]


@router.post("/orders/{order_id}/release", response_model=schemas.DecisionOut,
             summary="Release now, overriding any hold")
def release_order(request: Request, order_id: str, payload: schemas.ReleaseIn,
                  tenant: Tenant = Depends(get_tenant)) -> Dict[str, Any]:
    """The merchant's override. Always available, never refused for an open order.

    Deliberately not idempotency-keyed: releasing an already-released order is a
    no-op that returns the same body, so replay protection would add a failure mode
    without preventing one.
    """
    app_state = state_of(request)
    with app_state.lock:
        try:
            disposition = app_state.service(tenant).release(
                order_id, reason=payload.reason, actor="merchant")
        except NotFound:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")
    return _decision_out(disposition)


@router.post("/orders/{order_id}/cancel", response_model=schemas.DecisionOut,
             summary="Cancel before packing")
def cancel_order(request: Request, order_id: str, payload: schemas.ReleaseIn,
                 tenant: Tenant = Depends(get_tenant)) -> Dict[str, Any]:
    app_state = state_of(request)
    with app_state.lock:
        try:
            disposition = app_state.service(tenant).cancel(
                order_id, reason=payload.reason, actor="merchant")
        except NotFound:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")
        except ValueError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return _decision_out(disposition)


# ==========================================================================
# outcomes
# ==========================================================================

@router.post("/outcomes", response_model=schemas.OutcomeOut,
             summary="Report what the carrier did")
def create_outcome(request: Request, payload: schemas.OutcomeIn,
                   tenant: Tenant = Depends(get_tenant),
                   key: Optional[str] = Depends(idempotency_key)
                   ) -> Dict[str, Any]:
    """Ingest a terminal outcome: the label that makes everything else honest.

    The response itemises what the outcome did to the balance, line by line, because
    a merchant should be able to see *why* an RTO cost them Rs 195 rather than being
    told a total. Repeating the same outcome returns the same body; reporting a
    *different* one is a 409 -- a reversal is posted as a correction, deliberately,
    not by overwriting what the ledger already believes.
    """
    app_state = state_of(request)
    kind = _enum(OutcomeKind, payload.kind, "kind")
    with app_state.lock:
        with Idempotent(tenant, "POST /v1/outcomes", key,
                        payload.model_dump(mode="json")) as slot:
            if slot.replayed:
                return slot.response

            order_id = _resolve_order_id(tenant, payload.order_id,
                                         payload.external_id)
            outcomes = OutcomeRepo(tenant)
            try:
                outcome = outcomes.record(
                    order_id, kind, reported_at=payload.reported_at,
                    carrier_reason=payload.carrier_reason,
                    attempts=payload.attempts, source="api")
            except NotFound:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")
            except Conflict as exc:
                raise HTTPException(status.HTTP_409_CONFLICT, str(exc))

            order = OrderRepo(tenant).get(order_id)
            itemised = ledger_breakdown(
                tenant.config, order.value_paise, outcome.kind,
                incentive_paise=outcomes.incentive_spent(order_id))

            body = schemas.OutcomeOut(
                order_id=order_id, kind=outcome.kind.value,
                reported_at=outcome.reported_at,
                net_paise=itemised.net_paise,
                net_rupees=format_paise(itemised.net_paise, sign=True),
                lines=[{"reason": line.reason.value,
                        "amount_paise": line.amount_paise,
                        "amount_rupees": format_paise(line.amount_paise,
                                                      sign=True),
                        "note": line.note}
                        for line in itemised.lines]).model_dump()
            body["release"] = order.state.value in {
                "released_for_dispatch", "shipped", "delivered", "rto"}
            slot.store(body)
            return body


# ==========================================================================
# operations
# ==========================================================================

@router.post("/holds/sweep", summary="Release every order whose hold has expired")
def sweep_holds(request: Request,
                tenant: Tenant = Depends(get_tenant)) -> Dict[str, Any]:
    """The auto-release timer, exposed so it can be driven by cron or by a test.

    Also runs on a background task inside the app, so a deployment that never calls
    this still honours `max_hold_minutes`. Exposed anyway because an operator needs
    to be able to force it, and because a promise you cannot trigger on demand is
    hard to believe.
    """
    app_state = state_of(request)
    with app_state.lock:
        released = app_state.service(tenant).sweep()
    return {"released": len(released),
            "orders": [_decision_out(d) for d in released]}


@router.get("/orders", summary="Orders currently inside the decision window")
def list_open_orders(request: Request,
                     tenant: Tenant = Depends(get_tenant)) -> Dict[str, Any]:
    app_state = state_of(request)
    with app_state.lock:
        orders = OrderRepo(tenant)
        held = orders.expired_holds(tenant.runtime.stamp(), limit=50)
        counts = orders.count_by_state()
        pending = app_state.service(tenant).pending_holds()
        recent = orders.recent(limit=100)
    return {
        "by_state": counts,
        "orders": [{"order_id": o.id, "state": o.state.value,
                    "held_until": o.held_until, "value_paise": o.value_paise,
                    "mode": o.mode.value} for o in recent],
        "awaiting": [{"order_id": o.id, "state": o.state.value,
                      "held_until": o.held_until,
                      "value_paise": o.value_paise} for o in pending],
        "overdue": [{"order_id": o.id, "held_until": o.held_until}
                     for o in held],
    }


@router.post("/backtest", summary="Run the deterministic synthetic ROI backtest")
def run_backtest(request: Request, payload: Optional[schemas.HistoricalBacktestIn] = None,
                 tenant: Tenant = Depends(get_tenant),
                 seed: int = 42, orders: int = 500,
                 sensitivity: bool = False) -> Dict[str, Any]:
    """Run the same zero-network evaluation used by merchant onboarding.

    Query parameters keep the endpoint dependency-free for the MVP. Historical CSV
    ingestion is a later adapter; synthetic provenance is always explicit here.
    """
    if payload is not None:
        try:
            return project_csv(tenant, payload.csv_text,
                               effectiveness_ppm=payload.effectiveness_ppm)
        except HistoricalImportError as exc:
            raise HTTPException(HTTP_422, str(exc))
    if orders < 1 or orders > 50_000:
        raise HTTPException(HTTP_422, "orders must be between 1 and 50000")
    from ..backtest import run
    return run(seed=seed, n_orders=orders, include_sensitivity=sensitivity)


@router.get("/reports/roi", summary="Read the merchant ROI report")
def get_roi_report(request: Request, tenant: Tenant = Depends(get_tenant),
                   arm: str = "live", since: Optional[str] = None
                   ) -> Dict[str, Any]:
    if not arm or len(arm) > 40:
        raise HTTPException(HTTP_422, "arm must be a non-empty short name")
    return roi_report(tenant, arm=arm, since=since)


@router.get("/audit", summary="Export the tenant audit chain")
def export_audit(request: Request, tenant: Tenant = Depends(get_tenant)
                 ) -> Dict[str, Any]:
    verification = tenant.audit.verify(tenant.merchant_id)
    exported = tenant.audit.export(tenant.merchant_id)
    exported["verification"] = {
        "ok": verification.ok,
        "entries_checked": verification.entries_checked,
        "first_bad_seq": verification.first_bad_seq,
        "detail": verification.detail,
    }
    return exported


@router.get("/merchant/config", response_model=schemas.ConfigOut,
            summary="Read merchant decision and economic configuration")
def get_config(tenant: Tenant = Depends(get_tenant)) -> Dict[str, Any]:
    return _config_out(tenant)


@router.patch("/merchant/config", response_model=schemas.ConfigOut,
              summary="Update merchant decision and economic configuration")
def update_config(payload: schemas.ConfigUpdateIn,
                  tenant: Tenant = Depends(get_tenant)) -> Dict[str, Any]:
    try:
        updated = save_config(tenant, _updated_config(tenant, payload))
    except (ValueError, TypeError) as exc:
        raise HTTPException(HTTP_422, str(exc))
    return _config_out(updated)


@router.get("/merchant/api-keys", response_model=list[schemas.ApiKeyOut],
            summary="List merchant API-key metadata")
def get_api_keys(tenant: Tenant = Depends(get_tenant)) -> List[Dict[str, Any]]:
    return [_key_out(record) for record in list_api_keys(tenant)]


@router.post("/merchant/api-keys", response_model=schemas.ApiKeyOut,
             summary="Issue a merchant API key; plaintext is returned once")
def create_api_key(payload: schemas.ApiKeyCreateIn,
                   tenant: Tenant = Depends(get_tenant)) -> Dict[str, Any]:
    if payload.mode not in ("test", "live"):
        raise HTTPException(HTTP_422, "mode must be test or live")
    plaintext, record = issue_api_key(tenant, mode=payload.mode,
                                      label=payload.label)
    return _key_out(record, plaintext)


@router.post("/merchant/api-keys/{api_key_id}/revoke",
             summary="Revoke a merchant API key")
def delete_api_key(api_key_id: str,
                   tenant: Tenant = Depends(get_tenant)) -> Dict[str, Any]:
    try:
        revoke_api_key(tenant, api_key_id)
    except Exception as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return {"revoked": True, "api_key_id": api_key_id}
