"""Historical CSV validation and merchant onboarding projection.

This module deliberately does not write imported rows into the live order table.
An onboarding file is a projection input, not proof that those parcels passed
through PreCourier. The report keeps observed ledger results separate from the
modeled effect of a future intervention.
"""

from __future__ import annotations

import csv
import io
import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import ledger
from .config import MerchantConfig
from .domain import Address, Buyer, Cause, Order, OrderState, OutcomeKind
from .geo import REFERENCE
from .intelligence import build_features, diagnose, evaluate_gate, rule_risk_score
from .money import PPM, Paise, Ppm, format_paise, format_ppm, apply_ppm
from .pii import phone_hash, normalise_phone
from .runtime import parse_ts
from .tenancy import Tenant


class HistoricalImportError(ValueError):
    """A CSV cannot be safely interpreted as a historical order export."""


@dataclasses.dataclass(frozen=True)
class HistoricalOrder:
    external_id: str
    value_paise: Paise
    outcome: OutcomeKind
    phone: str = ""
    email: str = ""
    address: Address = Address()
    payment_method: str = "cod"
    item_count: int = 1
    category: str = ""
    discount_bps: int = 0
    placed_at: str = ""
    reported_at: str = ""
    carrier_reason: str = ""
    attempts: int = 0


MAX_ROWS = 50_000
_VALUE_COLUMNS = ("value", "value_rupees", "value_paise")


def parse_csv(text: str, *, max_rows: int = MAX_ROWS) -> List[HistoricalOrder]:
    """Parse a deliberately small, provider-neutral historical export.

    Required columns are ``external_id``, one value column, and ``outcome``.
    Address columns mirror the public order API. Unknown columns are tolerated so
    Shopify/WooCommerce/3PL exports can be trimmed by the importer without a
    brittle provider-specific parser.
    """
    if not isinstance(text, str) or not text.strip():
        raise HistoricalImportError("CSV content is empty")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HistoricalImportError("CSV must include a header row")
    fields = {str(name).strip().lower() for name in reader.fieldnames if name}
    missing = [name for name in ("external_id", "outcome") if name not in fields]
    if missing:
        raise HistoricalImportError(
            "CSV is missing required column(s): %s" % ", ".join(missing))
    if not fields.intersection(_VALUE_COLUMNS):
        raise HistoricalImportError(
            "CSV needs one value column: value, value_rupees or value_paise")

    rows: List[HistoricalOrder] = []
    seen: set[str] = set()
    for number, raw in enumerate(reader, start=2):
        if number - 1 > max_rows:
            raise HistoricalImportError("CSV exceeds the %d-row limit" % max_rows)
        row = {_key(key): (value or "").strip() for key, value in raw.items()
               if key is not None}
        try:
            item = _parse_row(row, number)
        except (TypeError, ValueError, HistoricalImportError) as exc:
            if isinstance(exc, HistoricalImportError):
                raise
            raise HistoricalImportError("row %d: %s" % (number, exc)) from exc
        if item.external_id in seen:
            raise HistoricalImportError(
                "row %d: duplicate external_id %r" % (number, item.external_id))
        seen.add(item.external_id)
        rows.append(item)
    if not rows:
        raise HistoricalImportError("CSV has no data rows")
    return rows


def _key(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _parse_row(row: Dict[str, str], number: int) -> HistoricalOrder:
    external_id = row.get("external_id", "").strip()
    if not external_id:
        raise HistoricalImportError("row %d: external_id is required" % number)

    supplied = [(name, row[name]) for name in _VALUE_COLUMNS if row.get(name, "")]
    if len(supplied) != 1:
        raise HistoricalImportError(
            "row %d: supply exactly one of value, value_rupees or value_paise"
            % number)
    value_name, value_text = supplied[0]
    if value_name == "value_paise":
        try:
            value_paise = int(value_text)
        except ValueError as exc:
            raise HistoricalImportError(
                "row %d: value_paise must be an integer" % number) from exc
        if value_paise < 0:
            raise HistoricalImportError("row %d: value_paise must not be negative" % number)
    else:
        from .money import to_paise
        value_paise = to_paise(value_text)
        if value_paise < 0:
            raise HistoricalImportError("row %d: value must not be negative" % number)

    try:
        outcome = OutcomeKind(row.get("outcome", "").strip().lower())
    except ValueError as exc:
        raise HistoricalImportError(
            "row %d: outcome must be delivered, rto, cancelled or prepaid_delivered"
            % number) from exc

    payment_method = row.get("payment_method", "cod").lower() or "cod"
    if payment_method not in ("cod", "prepaid"):
        raise HistoricalImportError(
            "row %d: payment_method must be cod or prepaid" % number)
    item_count = _integer(row.get("item_count", "1"), "item_count", number)
    if item_count < 1:
        raise HistoricalImportError("row %d: item_count must be at least 1" % number)
    discount_bps = _integer(row.get("discount_bps", "0"), "discount_bps", number)
    if not 0 <= discount_bps <= 10_000:
        raise HistoricalImportError("row %d: discount_bps must be within 0..10000" % number)
    attempts = _integer(row.get("attempts", "0"), "attempts", number)
    if attempts < 0:
        raise HistoricalImportError("row %d: attempts must not be negative" % number)
    placed_at = _timestamp(row.get("placed_at", ""), "placed_at", number)
    reported_at = _timestamp(row.get("reported_at", ""), "reported_at", number)
    return HistoricalOrder(
        external_id=external_id, value_paise=value_paise, outcome=outcome,
        phone=row.get("phone", ""), email=row.get("email", ""),
        address=Address(
            raw=row.get("address_raw", row.get("raw", "")),
            line1=row.get("line1", ""), line2=row.get("line2", ""),
            landmark=row.get("landmark", ""), locality=row.get("locality", ""),
            city=row.get("city", ""), state=row.get("state", ""),
            pincode=row.get("pincode", "")),
        payment_method=payment_method, item_count=item_count,
        category=row.get("category", ""), discount_bps=discount_bps,
        placed_at=placed_at, reported_at=reported_at,
        carrier_reason=row.get("carrier_reason", ""), attempts=attempts)


def _integer(text: str, name: str, number: int) -> int:
    try:
        return int(text or "0")
    except ValueError as exc:
        raise HistoricalImportError(
            "row %d: %s must be an integer" % (number, name)) from exc


def _timestamp(text: str, name: str, number: int) -> str:
    if not text:
        return ""
    try:
        return parse_ts(text).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise HistoricalImportError(
            "row %d: %s must be ISO-8601" % (number, name)) from exc


def project(tenant: Tenant, rows: Sequence[HistoricalOrder], *,
            effectiveness_ppm: Ppm = 550_000) -> Dict[str, Any]:
    """Build a baseline and explicitly assumption-labelled intervention projection."""
    if not rows:
        raise HistoricalImportError("at least one historical row is required")
    if not 0 <= effectiveness_ppm <= PPM:
        raise HistoricalImportError("effectiveness_ppm must be within 0..1000000")

    cfg = tenant.config
    actual_net = 0
    outcomes: Dict[str, int] = {}
    recommendations: Dict[str, int] = {}
    projected_rto_candidates = 0
    projected_uplift = 0
    history: Dict[str, Tuple[int, int, int]] = {}
    reviewed: List[Dict[str, Any]] = []

    for index, row in enumerate(rows):
        identity = phone_hash(row.phone, pepper=tenant.runtime.pepper)
        identity = identity or "historical:%s" % index
        prior_orders, prior_rtos, prior_prepaid = history.get(identity, (0, 0, 0))
        buyer = Buyer(
            id="hist_buyer_%06d" % index, merchant_id=tenant.merchant_id,
            phone_hash=identity, phone=normalise_phone(row.phone).digits,
            email=row.email, prior_orders=prior_orders,
            prior_rtos=prior_rtos, prior_prepaid=prior_prepaid)
        order = Order(
            id="hist_order_%06d" % index, merchant_id=tenant.merchant_id,
            buyer_id=buyer.id, value_paise=row.value_paise, address=row.address,
            external_id=row.external_id, state=OrderState.RECEIVED,
            mode=cfg.mode, payment_method=row.payment_method,
            item_count=row.item_count, category=row.category,
            discount_bps=row.discount_bps,
            is_first_time_buyer=prior_orders == 0, placed_at=row.placed_at)
        features = build_features(order, buyer, config=cfg, reference=REFERENCE)
        score = rule_risk_score(features)
        diagnosis = diagnose(features)
        gate = evaluate_gate(cfg, order, score, diagnosis,
                             effectiveness_ppm=effectiveness_ppm)
        actual = ledger.breakdown(cfg, row.value_paise, row.outcome).net_paise
        actual_net += actual
        outcomes[row.outcome.value] = outcomes.get(row.outcome.value, 0) + 1
        recommendations[gate.action.value] = recommendations.get(gate.action.value, 0) + 1

        # This is a projection, not a causal claim. Only observed RTO rows are
        # eligible for the illustrative repair scenario, and only the supported
        # MVP address-repair action is modelled.
        if row.outcome is OutcomeKind.RTO and gate.action.value == "address_repair":
            projected_rto_candidates += 1
            projected_uplift += apply_ppm(
                ledger.swing(cfg, row.value_paise), effectiveness_ppm)
            projected_uplift -= cfg.economics(gate.action).cost_paise
        history[identity] = (
            prior_orders + 1,
            prior_rtos + int(row.outcome is OutcomeKind.RTO),
            prior_prepaid + int(row.outcome is OutcomeKind.PREPAID_DELIVERED))
        if len(reviewed) < 10:
            reviewed.append({
                "external_id": row.external_id,
                "p_rto_ppm": score.p_rto_ppm,
                "cause": diagnosis.cause.value,
                "recommended_action": gate.action.value,
                "observed_outcome": row.outcome.value,
            })

    return {
        "report_version": "historical-projection-v1",
        "provenance": "merchant_historical_csv",
        "synthetic_data": False,
        "merchant_id": tenant.merchant_id,
        "orders": len(rows),
        "outcomes": outcomes,
        "recommendations_by_action": recommendations,
        "observed_net_paise": actual_net,
        "observed_net_rupees": format_paise(actual_net, sign=True),
        "observed_rto_rate_ppm": outcomes.get(OutcomeKind.RTO.value, 0) * PPM // len(rows),
        "modeled_address_repair_candidates": projected_rto_candidates,
        "modeled_uplift_paise": projected_uplift,
        "modeled_uplift_rupees": format_paise(projected_uplift, sign=True),
        "modeled_net_paise": actual_net + projected_uplift,
        "modeled_net_rupees": format_paise(actual_net + projected_uplift, sign=True),
        "assumptions": {
            "effectiveness_ppm": effectiveness_ppm,
            "effectiveness": format_ppm(effectiveness_ppm),
            "only_observed_rtos_are_eligible": True,
            "only_address_repair_is_modeled": True,
            "modeled_uplift_is_not_incremental_causal_evidence": True,
        },
        "review_sample": reviewed,
    }


def project_csv(tenant: Tenant, text: str, *,
                effectiveness_ppm: Ppm = 550_000) -> Dict[str, Any]:
    return project(tenant, parse_csv(text), effectiveness_ppm=effectiveness_ppm)


__all__ = ["HistoricalImportError", "HistoricalOrder", "parse_csv", "project",
           "project_csv"]
