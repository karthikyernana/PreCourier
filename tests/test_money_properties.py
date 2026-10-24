from hypothesis import given, strategies as st

from precourier.money import apply_bps, div_half_up, from_micro, to_micro
from precourier.api.schemas import OrderIn


def test_api_rejects_negative_rupee_value():
    import pydantic
    try:
        OrderIn.model_validate({"value": "-1.00", "address": {"raw": "x"}})
    except pydantic.ValidationError:
        return
    raise AssertionError("negative order values must be rejected at the API boundary")


@given(st.integers(-10**12, 10**12), st.integers(1, 10**6))
def test_rounding_is_symmetric(numerator, denominator):
    assert div_half_up(-numerator, denominator) == -div_half_up(numerator, denominator)


@given(st.integers(-10**9, 10**9), st.integers(0, 10_000))
def test_nonnegative_cost_rate_never_turns_nonnegative_money_negative(paise, rate):
    if paise >= 0:
        assert apply_bps(paise, rate) >= 0


@given(st.integers(-10**12, 10**12))
def test_micro_conversion_is_sign_symmetric(paise):
    assert from_micro(to_micro(-paise)) == -from_micro(to_micro(paise))
