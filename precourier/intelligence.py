"""Pure observable intelligence for the pre-dispatch decision path.

This module deliberately stops at the decision boundary. It does not read a
database, call a provider, import the synthetic world, or mutate an order. The
caller supplies observable carrier and geography answers when those answers are
available. That keeps a feature vector reproducible and keeps the gate separate
from the state service and its policy guards.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import re
from typing import Any, Dict, Optional, Sequence, Tuple

from .config import MerchantConfig
from .domain import (ActionKind, Buyer, Cause, Diagnosis, Evidence, Order,
                     RiskScore)
from .ledger import (derive_cancel_threshold, derive_prepaid_threshold,
                     derive_repair_threshold, loss_on_rto, margin_on_delivery,
                     margin_on_prepaid, swing)
from .money import (PPM, Micro, Paise, Ppm, complement_ppm, mul_ppm, to_micro,
                    weigh)
from .pii import normalise_phone


_SIX_DIGITS = re.compile(r"^\d{6}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_HOUSE_NUMBER = re.compile(r"(?:^|\s|[,/-])\d+[A-Za-z]?(?:\s|$|[,/-])")
_FLOOR_OR_FLAT = re.compile(r"\b(?:flat|floor|apt|apartment|unit|house|h\.no|hno)\b",
                            re.IGNORECASE)
_CARRIER_VALID = frozenset(("valid", "passed", "pass", "ok", "accepted"))
_CARRIER_INVALID = frozenset(("invalid", "rejected", "failed", "fail"))
_CARRIER_UNREACHABLE = frozenset(("unreachable", "contact_unreachable"))
_CARRIER_UNSERVICEABLE = frozenset(
    ("not_serviceable", "unserviceable", "cod_unavailable"))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normal_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _same_place(city: str, candidates: Sequence[str]) -> bool:
    """Compare city names without making a network or geocoding claim."""
    aliases = {"bengaluru": "bangalore", "bombay": "mumbai",
               "calcutta": "kolkata", "madras": "chennai"}

    def canonical(value: str) -> str:
        result = _normal_text(value)
        for old, new in aliases.items():
            result = result.replace(old, new)
        return result

    wanted = canonical(city)
    if not wanted:
        return False
    for candidate in candidates:
        possible = canonical(_text(candidate))
        if possible and (wanted == possible or wanted in possible
                         or possible in wanted):
            return True
    return False


def _placed_parts(value: str) -> Tuple[Optional[int], Optional[int]]:
    if not value:
        return None, None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None, None
    return parsed.hour, parsed.weekday()


def _carrier_state(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "valid" if value else "invalid"
    state = _text(value).casefold().replace(" ", "_")
    if state in _CARRIER_VALID:
        return "valid"
    if state in _CARRIER_INVALID:
        return "invalid"
    if state in _CARRIER_UNREACHABLE:
        return "unreachable"
    if state in _CARRIER_UNSERVICEABLE:
        return "not_serviceable"
    return state or "unknown"


@dataclasses.dataclass(frozen=True)
class Features:
    """Immutable, observable features for one order and buyer.

    Fields after ``carrier_validation`` are post-contact observations. They are
    retained for a second pass and reporting, but ``first_pass_mapping`` never
    includes them. Values are integer or categorical so serialisation is stable
    and no float can enter the first-pass rule score.
    """

    order_id: str
    merchant_id: str

    address_token_count: int = 0
    has_house_number: bool = False
    has_floor_or_flat: bool = False
    has_landmark: bool = False
    has_locality: bool = False
    pincode_known: bool = False
    pincode_city_consistent: bool = False
    locality_in_reference: bool = False
    digit_ratio_ppm: Ppm = 0
    repeated_characters: bool = False

    phone_present: bool = False
    phone_format_valid: bool = False
    phone_prefix_valid: bool = False
    email_present: bool = False
    email_plausible: bool = False
    duplicate_phone_count: int = 0
    contact_verified: bool = False

    prior_orders: int = 0
    prior_rtos: int = 0
    prior_prepaid: int = 0
    prior_messages: int = 0
    prior_replies: int = 0
    refusal_ratio_ppm: Ppm = 0

    order_value_paise: Paise = 0
    category: str = ""
    item_count: int = 0
    discount_bps: int = 0
    is_first_time_buyer: bool = False
    placed_hour: Optional[int] = None
    placed_weekday: Optional[int] = None
    impulse_hour: bool = False
    category_median_paise: Optional[Paise] = None
    category_value_ratio_ppm: Optional[Ppm] = None

    pincode_tier: str = "unknown"
    distance_band: str = "unknown"
    pincode_rto_rate_ppm: Ppm = 0
    cod_serviceable: Optional[bool] = None
    monsoon_flag: bool = False
    festival_flag: bool = False
    carrier_validation: str = "unknown"

    reply_received: bool = False
    reply_latency_seconds: Optional[int] = None
    reply_valid_patch: bool = False

    @property
    def has_repeated_characters(self) -> bool:
        """Descriptive alias used by some feature/report consumers."""
        return self.repeated_characters

    def first_pass_mapping(self) -> Dict[str, Any]:
        """Return only features available before any buyer contact."""
        values = dataclasses.asdict(self)
        for key in ("order_id", "merchant_id", "reply_received",
                    "reply_latency_seconds", "reply_valid_patch"):
            values.pop(key, None)
        return values

    def as_mapping(self, *, include_post_contact: bool = True) -> Dict[str, Any]:
        values = dataclasses.asdict(self)
        if not include_post_contact:
            for key in ("reply_received", "reply_latency_seconds",
                        "reply_valid_patch"):
                values.pop(key, None)
        return values

    @property
    def features_id(self) -> str:
        payload = json.dumps(self.first_pass_mapping(), sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


FeatureSet = Features
ObservableFeatures = Features


def _reference_row(reference: Any, pincode: str) -> Any:
    if reference is None or not pincode:
        return None
    getter = getattr(reference, "get", None)
    return getter(pincode) if getter is not None else None


def _row_places(row: Any) -> Tuple[str, ...]:
    if row is None:
        return ()
    places = getattr(row, "places", ()) or ()
    return tuple(places) + tuple(
        _text(getattr(row, field, ""))
        for field in ("primary_place", "district"))


def _distance_band(origin_row: Any, destination_row: Any) -> str:
    if origin_row is None or destination_row is None:
        return "unknown"
    lat1, lon1 = float(getattr(origin_row, "lat", 0.0)), float(
        getattr(origin_row, "lon", 0.0))
    lat2, lon2 = float(getattr(destination_row, "lat", 0.0)), float(
        getattr(destination_row, "lon", 0.0))
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    hav = (math.sin(d_phi / 2) ** 2
           + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    km = 2 * 6371.0088 * math.asin(math.sqrt(min(1.0, hav)))
    if km < 50:
        return "band_0"
    if km < 200:
        return "band_1"
    if km < 800:
        return "band_2"
    if km < 1800:
        return "band_3"
    return "band_4"


def build_features(
    order: Order,
    buyer: Buyer,
    *,
    config: Optional[MerchantConfig] = None,
    reference: Any = None,
    duplicate_phone_count: int = 0,
    contact_verified: bool = False,
    category_median_paise: Optional[Paise] = None,
    pincode_rto_rate_ppm: int = 0,
    cod_serviceable: Optional[bool] = None,
    carrier_validation: Any = None,
    distance_band: Optional[str] = None,
    monsoon_flag: bool = False,
    festival_flag: bool = False,
    reply_received: bool = False,
    reply_latency_seconds: Optional[int] = None,
    reply_valid_patch: bool = False,
) -> Features:
    """Build deterministic features from an order, buyer, and observations.

    ``reference`` is intentionally injected. Passing a reference object makes
    the lookup deterministic in tests and avoids hidden file or network access;
    a missing reference falls back to the structural six-digit PIN check.
    ``cod_serviceable`` is a carrier observation and is never inferred from the
    static PIN reference table.
    """
    address = order.address
    address_text = address.as_text()
    tokens = address.tokens()
    phone = normalise_phone(buyer.phone)
    pincode = _text(address.pincode)
    row = _reference_row(reference, pincode)
    pincode_known = bool(row) if reference is not None else bool(
        _SIX_DIGITS.fullmatch(pincode))
    city_consistent = True
    locality_in_reference = False
    tier = "unknown"
    if reference is not None:
        if row:
            city_consistent = _same_place(address.city, _row_places(row))
            if address.locality:
                if hasattr(row, "mentions"):
                    locality_in_reference = bool(row.mentions(address.locality))
                else:
                    locality_in_reference = _same_place(
                        address.locality, _row_places(row))
        tier = _text(getattr(row, "tier", "unknown")) if row else "unknown"
    elif not address.city:
        city_consistent = False

    hour, weekday = _placed_parts(order.placed_at)
    chars = len(address_text)
    digit_ratio_ppm = (len(re.findall(r"\d", address_text)) * PPM // chars
                       if chars else 0)
    repeated = bool(re.search(r"(.)\1\1", address_text, re.IGNORECASE))
    ratio_ppm: Optional[Ppm] = None
    if category_median_paise is not None and category_median_paise > 0:
        ratio_ppm = order.value_paise * PPM // category_median_paise

    validation = _carrier_state(carrier_validation)
    if cod_serviceable is None and validation == "not_serviceable":
        cod_serviceable = False
    if distance_band is None and config is not None and reference is not None:
        distance_band = _distance_band(
            _reference_row(reference, _text(config.origin_pincode)), row)

    refusal_ratio_ppm = (min(PPM, buyer.prior_rtos * PPM // buyer.prior_orders)
                         if buyer.prior_orders > 0 else 0)
    return Features(
        order_id=order.id,
        merchant_id=order.merchant_id,
        address_token_count=len(tokens),
        has_house_number=bool(_HOUSE_NUMBER.search(address_text)),
        has_floor_or_flat=bool(_FLOOR_OR_FLAT.search(address_text)),
        has_landmark=bool(_text(address.landmark)),
        has_locality=bool(_text(address.locality)),
        pincode_known=pincode_known,
        pincode_city_consistent=city_consistent,
        locality_in_reference=locality_in_reference,
        digit_ratio_ppm=digit_ratio_ppm,
        repeated_characters=repeated,
        phone_present=bool(phone.digits),
        phone_format_valid=phone.plausible,
        phone_prefix_valid=phone.plausible,
        email_present=bool(_text(buyer.email)),
        email_plausible=bool(_EMAIL.fullmatch(_text(buyer.email))),
        duplicate_phone_count=max(0, int(duplicate_phone_count)),
        contact_verified=contact_verified,
        prior_orders=buyer.prior_orders,
        prior_rtos=buyer.prior_rtos,
        prior_prepaid=buyer.prior_prepaid,
        prior_messages=buyer.prior_messages,
        prior_replies=buyer.prior_replies,
        refusal_ratio_ppm=refusal_ratio_ppm,
        order_value_paise=order.value_paise,
        category=order.category,
        item_count=order.item_count,
        discount_bps=order.discount_bps,
        is_first_time_buyer=order.is_first_time_buyer,
        placed_hour=hour,
        placed_weekday=weekday,
        impulse_hour=hour is not None and hour < 6,
        category_median_paise=category_median_paise,
        category_value_ratio_ppm=ratio_ppm,
        pincode_tier=tier,
        distance_band=distance_band or "unknown",
        pincode_rto_rate_ppm=max(0, min(PPM, int(pincode_rto_rate_ppm))),
        cod_serviceable=cod_serviceable,
        monsoon_flag=monsoon_flag,
        festival_flag=festival_flag,
        carrier_validation=validation,
        reply_received=reply_received,
        reply_latency_seconds=reply_latency_seconds,
        reply_valid_patch=reply_valid_patch,
    )


def rule_risk_score(features: Features, *, model_version: str = "rule-v1") -> RiskScore:
    """Return a deterministic, interpretable P(RTO) rule score in integer PPM."""
    score = 50_000
    if not features.pincode_known:
        score += 180_000
    if not features.pincode_city_consistent:
        score += 100_000
    if features.address_token_count < 8:
        score += 70_000
    if not features.has_house_number:
        score += 110_000
    if not features.has_floor_or_flat:
        score += 35_000
    if not features.has_landmark:
        score += 20_000
    if features.carrier_validation == "invalid":
        score += 220_000
    elif features.carrier_validation == "unreachable":
        score += 240_000
    if not features.phone_format_valid:
        score += 220_000
    if features.duplicate_phone_count:
        score += min(120_000, features.duplicate_phone_count * 40_000)
    if features.refusal_ratio_ppm:
        score += min(300_000, features.refusal_ratio_ppm // 2)
    if features.prior_rtos >= 2:
        score += 80_000
    if features.is_first_time_buyer:
        score += 40_000
    if features.impulse_hour:
        score += 70_000
    if (features.category_value_ratio_ppm is not None
            and features.category_value_ratio_ppm > 2 * PPM):
        score += 90_000
    if features.cod_serviceable is False:
        score += 500_000
    score += min(100_000, max(0, features.pincode_rto_rate_ppm // 2))
    if features.pincode_tier == "tier_3":
        score += 20_000
    if features.distance_band in ("band_3", "band_4"):
        score += 25_000
    if features.repeated_characters:
        score += 15_000
    score = max(0, min(PPM, score))
    return RiskScore(p_rto_ppm=score, model_version=model_version,
                     features_id=features.features_id, raw_ppm=score)


score_risk = rule_risk_score
rule_risk = rule_risk_score


def diagnose(
    features: Features,
    *,
    address_token_threshold: int = 8,
    refusal_threshold_ppm: Ppm = 500_000,
) -> Diagnosis:
    """Select one binding cause and retain the observed evidence behind it."""
    evidence: Dict[Cause, Tuple[Evidence, ...]] = {}

    address: list[Evidence] = []
    if not features.pincode_known:
        address.append(Evidence("pincode_unknown", "PIN is absent or not in the reference", 5))
    if not features.pincode_city_consistent:
        address.append(Evidence("pincode_city_mismatch", "PIN does not match the supplied city", 4))
    if not features.has_house_number:
        address.append(Evidence("house_number_missing", "no house, flat, or door number", 3))
    if features.address_token_count < address_token_threshold:
        address.append(Evidence("address_tokens_below_threshold",
                                "%d token(s), threshold is %d"
                                % (features.address_token_count,
                                   address_token_threshold), 2))
    if not features.has_landmark:
        address.append(Evidence("landmark_missing", "no landmark supplied", 1))
    if (features.pincode_known and features.has_locality
            and not features.locality_in_reference):
        address.append(Evidence("locality_not_in_reference",
                                "locality is not listed under this PIN", 3))
    if features.carrier_validation == "invalid":
        address.append(Evidence("carrier_address_rejected",
                                "carrier validation rejected the address", 5))
    if address and sum(item.weight for item in address) >= 3:
        evidence[Cause.ADDRESS_DEFECT] = tuple(address)

    contact: list[Evidence] = []
    if not features.phone_present:
        contact.append(Evidence("phone_missing", "no phone number supplied", 4))
    elif not features.phone_format_valid:
        contact.append(Evidence("phone_format_invalid",
                                "number is not a plausible Indian mobile", 4))
    if not features.phone_prefix_valid and features.phone_present:
        contact.append(Evidence("phone_prefix_invalid", "mobile prefix failed validation", 2))
    if features.duplicate_phone_count:
        contact.append(Evidence("duplicate_phone",
                                "%d other buyer record(s) share this phone"
                                % features.duplicate_phone_count, 3))
    if features.carrier_validation == "unreachable":
        contact.append(Evidence("carrier_unreachable", "carrier flagged contact unreachable", 5))
    if contact:
        evidence[Cause.UNREACHABLE_CONTACT] = tuple(contact)

    weak_intent: list[Evidence] = []
    if features.impulse_hour:
        weak_intent.append(Evidence("impulse_hour", "order placed between 00:00 and 05:59", 2))
    if features.is_first_time_buyer:
        weak_intent.append(Evidence("first_time_buyer", "no prior order history", 1))
    if (features.category_value_ratio_ppm is not None
            and features.category_value_ratio_ppm > 2 * PPM):
        weak_intent.append(Evidence("value_above_category_norm",
                                    "order exceeds twice the category median", 3))
    if features.prior_replies == 0 and features.prior_orders > 0:
        weak_intent.append(Evidence("no_prior_reply", "buyer has not replied to prior contact", 1))
    if weak_intent:
        evidence[Cause.WEAK_INTENT] = tuple(weak_intent)

    repeat: list[Evidence] = []
    if features.prior_rtos:
        repeat.append(Evidence("prior_rtos", "%d prior RTO(s)" % features.prior_rtos, 3))
    if features.refusal_ratio_ppm >= refusal_threshold_ppm:
        repeat.append(Evidence("high_refusal_ratio",
                               "%d ppm refusal ratio" % features.refusal_ratio_ppm, 5))
    if repeat:
        evidence[Cause.REPEAT_RETURNER] = tuple(repeat)

    serviceability: list[Evidence] = []
    if features.cod_serviceable is False:
        serviceability.append(Evidence("cod_not_serviceable",
                                       "carrier says COD is unavailable for this PIN", 6))
    if features.carrier_validation == "not_serviceable":
        serviceability.append(Evidence("carrier_serviceability_rejected",
                                       "carrier rejected COD serviceability", 6))
    if serviceability:
        evidence[Cause.NOT_SERVICEABLE] = tuple(serviceability)

    priority = (Cause.NOT_SERVICEABLE, Cause.ADDRESS_DEFECT,
                Cause.UNREACHABLE_CONTACT, Cause.REPEAT_RETURNER,
                Cause.WEAK_INTENT)
    ranked = sorted(evidence, key=lambda cause: (
        -sum(item.weight for item in evidence[cause]), priority.index(cause)))
    if not ranked:
        return Diagnosis(Cause.UNCERTAIN, detail="no observable cause dominates")
    cause = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    detail = "%s selected from %d evidence item(s)" % (
        cause.value, len(evidence[cause]))
    return Diagnosis(cause, evidence[cause], runner_up, detail)


diagnose_order = diagnose


@dataclasses.dataclass(frozen=True)
class GateVerdict:
    """A pure result with the same public fields as ``states.Verdict``."""

    action: ActionKind = ActionKind.NO_ACTION
    p_rto_ppm: Ppm = 0
    cause: Cause = Cause.UNCERTAIN
    threshold_ppm: Ppm = 0
    expected_delta_micro: Micro = 0
    incentive_paise: Paise = 0
    reason: str = ""
    model_version: str = ""
    evidence: Tuple[Evidence, ...] = ()
    gate_inputs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    risk_score_id: Optional[str] = None
    diagnosis_id: Optional[str] = None
    hold_minutes: Optional[int] = None

    def to_verdict(self) -> Any:
        """Adapt to the runtime class without importing it at module import time."""
        from .states import Verdict
        return Verdict(
            action=self.action, p_rto_ppm=self.p_rto_ppm, cause=self.cause,
            threshold_ppm=self.threshold_ppm,
            expected_delta_micro=self.expected_delta_micro,
            incentive_paise=self.incentive_paise, reason=self.reason,
            model_version=self.model_version, evidence=self.evidence,
            gate_inputs=dict(self.gate_inputs), risk_score_id=self.risk_score_id,
            diagnosis_id=self.diagnosis_id, hold_minutes=self.hold_minutes)


def _score_value(score: Any) -> Tuple[Ppm, str, Optional[str]]:
    if isinstance(score, RiskScore):
        return score.p_rto_ppm, score.model_version, score.features_id or None
    if isinstance(score, bool):
        raise TypeError("risk score must be integer PPM, not bool")
    return int(score), "", None


def _validate_ppm(value: int, name: str) -> None:
    if not 0 <= value <= PPM:
        raise ValueError("%s must be within 0..%d, got %d" % (name, PPM, value))


def evaluate_gate(
    config: MerchantConfig,
    order: Order,
    score: Any,
    diagnosis: Diagnosis,
    *,
    effectiveness_ppm: Ppm = 550_000,
    incentive_paise: Optional[Paise] = None,
    spent_this_month_paise: Paise = 0,
) -> GateVerdict:
    """Evaluate one diagnosis using integer expected-ledger arithmetic.

    Consent, quiet hours, frequency, channel availability, and cancellation
    approval remain guards. This function only answers whether the economic
    action is worth taking and whether its configured spend is available.
    """
    p_rto, model_version, score_id = _score_value(score)
    _validate_ppm(p_rto, "p_rto_ppm")
    _validate_ppm(effectiveness_ppm, "effectiveness_ppm")
    if spent_this_month_paise < 0:
        raise ValueError("spent_this_month_paise must not be negative")
    if incentive_paise is None:
        incentive_paise = min(config.max_incentive_paise, 2_000)
    if incentive_paise < 0:
        raise ValueError("incentive_paise must not be negative")

    base_inputs: Dict[str, Any] = {
        "p_rto_ppm": p_rto,
        "effectiveness_ppm": effectiveness_ppm,
        "spent_this_month_paise": spent_this_month_paise,
    }
    if not order.is_cod:
        return GateVerdict(p_rto_ppm=p_rto, cause=diagnosis.cause,
                           reason="non-COD order needs no COD intervention",
                           model_version=model_version, evidence=diagnosis.evidence,
                           gate_inputs=base_inputs, risk_score_id=score_id)
    if diagnosis.cause is Cause.UNCERTAIN:
        return GateVerdict(p_rto_ppm=p_rto, cause=diagnosis.cause,
                           reason="no binding cause was supported by evidence",
                           model_version=model_version, evidence=diagnosis.evidence,
                           gate_inputs=base_inputs, risk_score_id=score_id)

    value = order.value_paise
    margin = margin_on_delivery(config, value)
    loss = loss_on_rto(config, value)
    total_swing = swing(config, value)
    no_op_micro = weigh(-loss, p_rto) + weigh(margin, complement_ppm(p_rto))
    action_by_cause = {
        Cause.ADDRESS_DEFECT: ActionKind.ADDRESS_REPAIR,
        Cause.UNREACHABLE_CONTACT: ActionKind.CONTACT_VERIFY,
        Cause.WEAK_INTENT: ActionKind.PREPAID_CONVERSION,
        Cause.REPEAT_RETURNER: ActionKind.CANCEL,
        Cause.NOT_SERVICEABLE: ActionKind.PREPAID_CONVERSION,
    }
    action = action_by_cause.get(diagnosis.cause)
    if action is None:
        return GateVerdict(p_rto_ppm=p_rto, cause=diagnosis.cause,
                           reason="cause does not have a configured action",
                           model_version=model_version, evidence=diagnosis.evidence,
                           gate_inputs=base_inputs, risk_score_id=score_id)
    effective_incentive = incentive_paise if action is ActionKind.PREPAID_CONVERSION else 0
    economics = config.economics(action)
    action_cost = economics.cost_paise + effective_incentive
    base_inputs.update({"margin_paise": margin, "loss_paise": loss,
                        "swing_paise": total_swing,
                        "expected_no_op_micro": no_op_micro,
                        "action_cost_paise": action_cost})
    threshold = None
    if action in (ActionKind.ADDRESS_REPAIR, ActionKind.CONTACT_VERIFY):
        threshold = derive_repair_threshold(config, value, action, effectiveness_ppm)
        effective = mul_ppm(p_rto, effectiveness_ppm)
        harm_ppm = economics.harm_bps * 100
        delta_micro = (weigh(total_swing, effective)
                       - weigh(total_swing,
                               mul_ppm(complement_ppm(p_rto), harm_ppm))
                       - to_micro(economics.cost_paise))
        hold_minutes = config.reply_window_hours * 60
    elif action is ActionKind.PREPAID_CONVERSION:
        threshold = derive_prepaid_threshold(config, value, effectiveness_ppm,
                                             effective_incentive)
        prepaid = margin_on_prepaid(config, value, effective_incentive)
        successful = mul_ppm(p_rto, effectiveness_ppm)
        spoiled_good = mul_ppm(complement_ppm(p_rto), effectiveness_ppm)
        delta_micro = (weigh(prepaid + loss, successful)
                       - weigh(margin - prepaid, spoiled_good)
                       - to_micro(economics.cost_paise))
        hold_minutes = config.reply_window_hours * 60
    else:
        threshold = derive_cancel_threshold(config, value)
        delta_micro = -to_micro(action_cost) - no_op_micro
        hold_minutes = None
    base_inputs.update({"expected_action_delta_micro": delta_micro,
                        "threshold_ppm": threshold.threshold_ppm,
                        "threshold_explanation": threshold.explanation})
    if effective_incentive > config.max_incentive_paise:
        reason = "incentive exceeds merchant ceiling"
        chosen = ActionKind.NO_ACTION
    elif (spent_this_month_paise + action_cost
          > config.monthly_incentive_budget_paise):
        reason = "action refused because the monthly budget is exhausted"
        chosen = ActionKind.NO_ACTION
    elif delta_micro <= 0:
        reason = "expected ledger delta is not positive"
        chosen = ActionKind.NO_ACTION
    else:
        reason = "%s is justified: p=%d ppm exceeds derived threshold=%d ppm" % (
            action.value, p_rto, threshold.threshold_ppm)
        chosen = action
    return GateVerdict(
        action=chosen, p_rto_ppm=p_rto, cause=diagnosis.cause,
        threshold_ppm=threshold.threshold_ppm,
        expected_delta_micro=delta_micro,
        incentive_paise=effective_incentive,
        reason=reason, model_version=model_version, evidence=diagnosis.evidence,
        gate_inputs=base_inputs, risk_score_id=score_id,
        hold_minutes=hold_minutes if chosen is not ActionKind.NO_ACTION else None)


gate = evaluate_gate


__all__ = [
    "FeatureSet", "Features", "GateVerdict", "ObservableFeatures",
    "build_features", "diagnose", "diagnose_order", "evaluate_gate", "gate",
    "rule_risk", "rule_risk_score", "score_risk",
]
