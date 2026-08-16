"""Integer money. No float ever reaches the ledger or the gate.

Four integer units are used, and they never mix silently:

    paise   integer rupees x 100.       Everything stored, reported, exported.
    bps     basis points, x 10_000.     Rates: gross margin, gateway fee, harm.
    ppm     parts per million, x 1e6.   Probabilities and effectiveness.
    micro   paise x 1_000_000.          Expected values, floored to paise once.

Why `micro` exists: the gate multiplies rupees by probabilities and sums the
terms. In floats that sum is order-dependent and platform-dependent, which
would break the byte-identical reproduction PRD section 13 requires and would
put un-auditable numbers in a money path. Expected values are therefore
accumulated as integers scaled by a million and converted back exactly once,
at the point a decision is recorded.

Probabilities arrive from models as floats. They are quantised to `ppm` at one
boundary -- when a risk score is stored -- so the number the gate used is
exactly the number the audit log shows.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Union

RUPEE = 100          # paise per rupee
BPS = 10_000         # basis points per unit
PPM = 1_000_000      # parts per million per unit
MICRO = 1_000_000    # micro-paise per paise

Paise = int
Bps = int
Ppm = int
Micro = int

Numeric = Union[int, str, Decimal]


# --------------------------------------------------------------------------
# rounding
# --------------------------------------------------------------------------

def div_half_up(numerator: int, denominator: int) -> int:
    """Integer division rounding halves away from zero.

    Python's `//` floors toward negative infinity, which would round -0.5 to -1
    and 0.5 to 0 -- asymmetric, and a slow leak in a ledger. This rounds
    symmetrically so that negating an input negates the output.
    """
    if denominator <= 0:
        raise ValueError("denominator must be positive, got %r" % (denominator,))
    if numerator >= 0:
        return (numerator * 2 + denominator) // (denominator * 2)
    return -((-numerator * 2 + denominator) // (denominator * 2))


# --------------------------------------------------------------------------
# parsing — the only place a decimal string becomes an integer
# --------------------------------------------------------------------------

def to_paise(value: Numeric) -> Paise:
    """Parse rupees into paise. Accepts int, Decimal, or a decimal string.

    Deliberately refuses float. `to_paise(0.1)` cannot be made correct, so it
    raises rather than rounding a number the caller did not intend.
    """
    if isinstance(value, bool):
        raise TypeError("bool is not a money value")
    if isinstance(value, int):
        return value * RUPEE
    if isinstance(value, float):
        raise TypeError(
            "float rupees are refused; pass a string or Decimal "
            "(got %r -- see precourier.money docstring)" % (value,)
        )
    try:
        dec = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("cannot parse %r as rupees" % (value,))
    return int((dec * RUPEE).to_integral_value(rounding="ROUND_HALF_UP"))


def to_bps(rate: Numeric) -> Bps:
    """Parse a rate such as "0.30" into basis points (3000)."""
    if isinstance(rate, bool):
        raise TypeError("bool is not a rate")
    if isinstance(rate, float):
        raise TypeError(
            "float rates are refused; pass a string or Decimal (got %r)" % (rate,)
        )
    if isinstance(rate, int):
        dec = Decimal(rate)
    else:
        try:
            dec = Decimal(rate)
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("cannot parse %r as a rate" % (rate,))
    return int((dec * BPS).to_integral_value(rounding="ROUND_HALF_UP"))


# --------------------------------------------------------------------------
# rates
# --------------------------------------------------------------------------

def apply_bps(paise: Paise, bps: Bps) -> Paise:
    """`paise` x `bps`/10000, rounded half away from zero."""
    return div_half_up(paise * bps, BPS)


# --------------------------------------------------------------------------
# probabilities
# --------------------------------------------------------------------------

def prob_to_ppm(p: float) -> Ppm:
    """Quantise a model probability to parts per million, clamped to [0, 1].

    This is the single boundary at which a float becomes an integer. Every
    downstream consumer -- gate, ledger, audit log, console -- sees the integer,
    so the arithmetic a merchant verifies by hand is the arithmetic that ran.
    """
    if p != p:  # NaN
        raise ValueError("probability is NaN")
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    # Via Decimal(repr(p)) so the quantisation is the decimal one a human
    # reading the float's printed form would expect, and is identical on every
    # platform. int(round(p * PPM)) is banker's rounding on a binary float and
    # differs in the last digit across architectures.
    return int((Decimal(repr(p)) * PPM).to_integral_value(rounding="ROUND_HALF_UP"))


def ppm_to_prob(ppm: Ppm) -> float:
    """For display and for model training only. Never for money arithmetic."""
    return ppm / PPM


def mul_ppm(a: Ppm, b: Ppm) -> Ppm:
    """Multiply two ppm quantities, staying in ppm."""
    return div_half_up(a * b, PPM)


def complement_ppm(ppm: Ppm) -> Ppm:
    """1 - p, in ppm."""
    return PPM - ppm


# --------------------------------------------------------------------------
# expected values
# --------------------------------------------------------------------------

def to_micro(paise: Paise) -> Micro:
    return paise * MICRO


def from_micro(micro: Micro) -> Paise:
    return div_half_up(micro, MICRO)


def weigh(paise: Paise, ppm: Ppm) -> Micro:
    """`paise` weighted by a probability, as micro-paise. Exact."""
    return paise * ppm


def apply_ppm(paise: Paise, ppm: Ppm) -> Paise:
    """`paise` x a ppm probability, back to whole paise. Rounded half from zero."""
    return div_half_up(paise * ppm, PPM)


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def format_paise(paise: Paise, sign: bool = False) -> str:
    """Render paise as rupees for a human. Never used as an input format."""
    negative = paise < 0
    whole, frac = divmod(abs(paise), RUPEE)
    grouped = "{:,}".format(whole)
    body = "₹%s.%02d" % (grouped, frac)
    if negative:
        return "-" + body
    if sign:
        return "+" + body
    return body


def format_ppm(ppm: Ppm, places: int = 1) -> str:
    """Render a ppm probability as a percentage, e.g. 576087 -> '57.6%'."""
    pct = Decimal(ppm) * 100 / PPM
    return ("{:." + str(places) + "f}%").format(pct)
