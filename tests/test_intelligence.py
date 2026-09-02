"""Focused tests for the pure observable intelligence boundary."""

from types import SimpleNamespace

from precourier.config import kolam_config
from precourier.domain import Address, Buyer, Cause, Diagnosis, Order, RiskScore
from precourier.intelligence import (Features, build_features, diagnose,
                                     evaluate_gate, rule_risk_score)


class _Reference:
    def __init__(self):
        self.row = SimpleNamespace(
            tier="metro", primary_place="Koramangala 4th Block",
            district="Bengaluru Urban", places=("Koramangala 4th Block",),
            lat=12.9352, lon=77.6245)

    def get(self, pincode):
        return self.row if pincode == "560034" else None


def _order(address=None, *, value=120_000, placed_at="2026-03-01T02:45:00Z"):
    return Order("o1", "m1", "b1", value,
                 address or Address(line1="Flat 402, Residency",
                                    locality="Koramangala 4th Block",
                                    city="Bengaluru", pincode="560034"),
                 placed_at=placed_at, category="ceramics")


def _buyer(**changes):
    values = dict(id="b1", merchant_id="m1", phone_hash="hash",
                  phone="9876543210", email="buyer@example.com")
    values.update(changes)
    return Buyer(**values)


def test_features_are_observable_and_reference_lookups_are_injected():
    features = build_features(_order(), _buyer(), reference=_Reference(),
                              config=kolam_config())

    assert features.pincode_known is True
    assert features.pincode_city_consistent is True
    assert features.locality_in_reference is True
    assert features.pincode_tier == "metro"
    assert features.has_house_number is True
    assert features.impulse_hour is True
    assert features.features_id == build_features(
        _order(), _buyer(), reference=_Reference(), config=kolam_config()).features_id


def test_post_contact_observations_do_not_change_first_pass_risk():
    before = build_features(_order(), _buyer(), reference=_Reference())
    after = build_features(_order(), _buyer(), reference=_Reference(),
                           reply_received=True, reply_latency_seconds=4,
                           reply_valid_patch=True)

    assert before.first_pass_mapping() == after.first_pass_mapping()
    assert rule_risk_score(before).p_rto_ppm == rule_risk_score(after).p_rto_ppm
    assert before.reply_received is False
    assert after.reply_valid_patch is True


def test_rule_score_is_integer_bounded_and_sensitive_to_observable_defects():
    good = build_features(_order(), _buyer(), reference=_Reference())
    bad = build_features(
        _order(Address(raw="near temple", city="Bengaluru", pincode="000000")),
        _buyer(phone="1234"), reference=_Reference(), cod_serviceable=False,
        carrier_validation="not_serviceable")

    score = rule_risk_score(good)
    bad_score = rule_risk_score(bad)
    assert isinstance(score, RiskScore)
    assert 0 <= score.p_rto_ppm <= 1_000_000
    assert bad_score.p_rto_ppm > score.p_rto_ppm
    assert score.p_rto_ppm == rule_risk_score(good).p_rto_ppm


def test_diagnosis_returns_binding_cause_and_evidence():
    features = build_features(
        _order(Address(raw="near temple", city="Bengaluru", pincode="000000")),
        _buyer(phone="1234"), reference=_Reference(), cod_serviceable=False,
        carrier_validation="not_serviceable")

    result = diagnose(features)
    assert result.cause is Cause.NOT_SERVICEABLE
    assert result.evidence
    assert all(item.code for item in result.evidence)

    uncertain = diagnose(Features(order_id="o", merchant_id="m",
                                  pincode_known=True,
                                  pincode_city_consistent=True,
                                  has_house_number=True,
                                  address_token_count=10,
                                  phone_present=True,
                                  phone_format_valid=True,
                                  phone_prefix_valid=True))
    assert uncertain.cause is Cause.UNCERTAIN
    assert uncertain.evidence == ()


def test_gate_matches_cancel_threshold_and_returns_verdict_shape():
    cfg = kolam_config()
    order = _order()
    diagnosis = Diagnosis(Cause.REPEAT_RETURNER)
    result = evaluate_gate(cfg, order,
                           RiskScore(800_000, "rule-v1", "features"),
                           diagnosis)

    assert result.action.value == "cancel"
    assert result.threshold_ppm > 570_000
    assert result.expected_delta_micro > 0
    assert result.gate_inputs["swing_paise"] == 46_000
    assert result.to_verdict().action is result.action


def test_gate_abstains_when_action_is_not_economic_or_budget_is_exhausted():
    cfg = kolam_config().replace(monthly_incentive_budget_paise=0)
    result = evaluate_gate(
        cfg, _order(), RiskScore(900_000, "rule-v1", "features"),
        Diagnosis(Cause.ADDRESS_DEFECT, ()))

    assert result.action.value == "no_action"
    assert "budget" in result.reason
    assert result.expected_delta_micro > 0
