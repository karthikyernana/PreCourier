from precourier.decision_engine import CauseProbabilities, choose_action, in_holdout
from precourier.domain import Address, Order
from precourier.config import kolam_config
from precourier.tenancy import Tenant


def _order(value=120000):
    return Order(id="o1", merchant_id="m1", buyer_id="b1", value_paise=value,
                 address=Address(line1="Flat 1", city="Chennai", pincode="600001"))


def test_engine_returns_non_negative_saving_and_runner_up():
    d = choose_action(_order(), CauseProbabilities(900000, 10000, 10000, 900000))
    assert d.expected_saving_paise >= 0
    assert d.runner_up
    assert any(x.action == "ship_as_is" for x in d.evaluations)


def test_holdout_is_deterministic_and_close_to_ten_percent():
    a = [in_holdout("order-%d" % i, "merchant") for i in range(10000)]
    assert a == [in_holdout("order-%d" % i, "merchant") for i in range(10000)]
    assert 850 <= sum(a) <= 1150
