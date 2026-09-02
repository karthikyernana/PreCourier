"""Decision-path performance and consistency checks for the production foundation."""

import time

from precourier.config import kolam_config
from precourier.domain import Address, Buyer, Mode, Order, OrderState
from precourier.intelligence import build_features, diagnose, evaluate_gate, rule_risk_score
from tests.fixtures import GOOD_ADDRESS


def test_rule_decision_path_is_fast_enough_for_the_configured_budget():
    order = Order(
        id="ord_perf", merchant_id="merchant", buyer_id="buyer", value_paise=120_000,
        address=GOOD_ADDRESS, mode=Mode.GATEWAY, state=OrderState.RECEIVED,
        placed_at="2026-03-01T09:00:00Z")
    buyer = Buyer(id="buyer", merchant_id="merchant", phone_hash="hash",
                  phone="9876543210", prior_orders=2, prior_rtos=1)
    config = kolam_config(merchant_id="merchant")
    started = time.perf_counter()
    for _ in range(100):
        features = build_features(order, buyer, config=config)
        score = rule_risk_score(features)
        diagnosis = diagnose(features)
        evaluate_gate(config, order, score, diagnosis)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms / 100 < config.decision_latency_budget_ms
