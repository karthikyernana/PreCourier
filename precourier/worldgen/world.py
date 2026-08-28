"""Latents, hazards, and the observable orders the agent is allowed to see.

This module holds the science. Three things in it are load-bearing and each is
asserted by a test rather than merely intended.

**Independent random streams.** Every draw comes from `_stream(spec, name, index)`,
a `random.Random` seeded from a hash of the seed, a stream *name* and an entity
index. So `feature_noise` cannot perturb a latent, `n_orders` cannot renumber
buyer 5, and adding a new emission cannot shift an existing one. Without this, the
sensitivity grid of PRD section 12 would compare different worlds and report the
difference as a parameter effect. `hashlib.blake2b` rather than `hash()`, which is
salted per process and would make the world un-reproducible across runs.

**The two mappings have different functional forms.** `p_return` is a logistic in a
weighted sum of latent deficits. The observables are independent Bernoulli emissions
whose rates come from those same deficits, compressed toward chance by
`feature_noise`. Shared latents mean real signal exists; different forms mean no
model can recover `p_return` exactly, however well it fits. `tests/test_worldgen.py`
bounds the correlation between any single observable and the truth from both sides:
too low and the problem is unlearnable, too high and the agent is reading the answer.

**Probabilities are quantised to ppm before any comparison.** `p_return` is computed
in floats and immediately rounded to parts per million; the outcome draw is a ppm
integer too, and the realised outcome is `draw_ppm < p_return_ppm`. This is not
decoration. `math.exp` is libm-dependent in its last bit, so a float comparison
would make the realised outcome of a knife-edge order platform-dependent, and PRD
section 13 requires a result anyone can reproduce. It is also the same discipline
the rest of the system applies to probabilities, for the same reason.

A note on the phone numbers: they are structurally valid Indian mobile numbers so
that the format check in the feature builder is genuinely exercised, and they are
invented. No number here was drawn from any real source, and nothing in this system
may dial one -- the messaging adapter is wired to a single operator-owned demo chat,
never to a buyer number, and a deployment carrying synthetic data announces itself
as such through `AppState.synthetic`.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .. import geo
from ..domain import Address, Cause
from ..money import PPM, Paise, Ppm, to_paise
from ..runtime import UTC, parse_ts
from .spec import WorldSpec

#: Bumped when a change here would produce a different world from the same spec.
#: Written into `truth.db` and printed with every result, because "same seed, same
#: numbers" is only a promise if the generator version is part of the seed's
#: identity. A reader comparing two runs needs to know whether the world moved.
GENERATOR_VERSION = "worldgen-1"


# ==========================================================================
# latent distributions
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class BetaParams:
    """A Beta distribution, plus the closed-form mean of its *deficit*.

    `deficit_mean` is `E[1 - X] = b / (a + b)`, computed rather than written down.
    It is used to normalise each hazard so that `repairable_share` means what it
    says: contributions are divided by their own expected size, so the expected
    share of hazard mass from the repairable causes is exactly the declared
    fraction. A hardcoded constant here would silently decouple the knob from its
    meaning the first time a shape parameter changed.
    """

    a: float
    b: float

    @property
    def deficit_mean(self) -> float:
        return self.b / (self.a + self.b)

    def draw(self, rng: random.Random) -> float:
        return rng.betavariate(self.a, self.b)


#: Shapes chosen for what they say about buyers, not for mathematical convenience.
#:
#: `reliability` is skewed high: most people who order a thing want the thing. The
#: left tail is the population the `repeat_returner` cause exists to find.
#: `responsiveness` is wide and centred, because willingness to answer an unknown
#: number genuinely splits down the middle -- and it is the latent that decides
#: whether any contact-based repair can work at all.
#: `address_quality` is skewed high but less so: most addresses are usable, a
#: substantial minority are missing a floor or a landmark.
#: `purchase_intent` is per order rather than per buyer, because the same person
#: buys deliberately on Tuesday and impulsively at 1am.
RELIABILITY = BetaParams(5.0, 2.0)          # mean 0.714, deficit mean 0.286
RESPONSIVENESS = BetaParams(2.5, 2.5)       # mean 0.500, deficit mean 0.500
ADDRESS_QUALITY = BetaParams(4.0, 2.0)      # mean 0.667, deficit mean 0.333
PURCHASE_INTENT = BetaParams(3.0, 2.0)      # mean 0.600, deficit mean 0.400

#: How much of the total hazard each cause carries, *within* its group. The
#: repairable pair splits 60/40 toward address defects and the unrepairable pair
#: 55/45 toward weak intent, which follows the ordering PRD section 4 gives the
#: causes. `repairable_share` then scales the two groups against each other.
WITHIN_REPAIRABLE: Dict[Cause, float] = {
    Cause.ADDRESS_DEFECT: 0.60,
    Cause.UNREACHABLE_CONTACT: 0.40,
}
WITHIN_UNREPAIRABLE: Dict[Cause, float] = {
    Cause.WEAK_INTENT: 0.55,
    Cause.REPEAT_RETURNER: 0.45,
}

#: Which latent's deficit drives which cause. One latent per cause, deliberately:
#: a cause driven by two latents would have no single binding constraint, and the
#: whole diagnosis step assumes there is one.
CAUSE_LATENT: Dict[Cause, str] = {
    Cause.ADDRESS_DEFECT: "address_quality",
    Cause.UNREACHABLE_CONTACT: "responsiveness",
    Cause.WEAK_INTENT: "purchase_intent",
    Cause.REPEAT_RETURNER: "reliability",
}

CAUSE_PARAMS: Dict[Cause, BetaParams] = {
    Cause.ADDRESS_DEFECT: ADDRESS_QUALITY,
    Cause.UNREACHABLE_CONTACT: RESPONSIVENESS,
    Cause.WEAK_INTENT: PURCHASE_INTENT,
    Cause.REPEAT_RETURNER: RELIABILITY,
}

#: Fixed order, so an argmax tie resolves the same way on every machine. Sorted by
#: the cause's own value string rather than by anything meaningful, because a tie
#: is a coin toss and the only requirement on it is determinism.
CAUSE_ORDER: Tuple[Cause, ...] = tuple(
    sorted(CAUSE_LATENT, key=lambda c: c.value))

#: Spread of the logistic. Larger means more heterogeneity in `p_return`, so a
#: model has more to find and the gate's thresholds bite on more orders. 2.6 puts
#: the middle 90% of a default world roughly between 8% and 62%, which is the range
#: where the derived thresholds of PRD section 5 actually discriminate.
LOGIT_SPREAD = 2.6

#: What an unserviceable COD destination does to the return probability. Not a
#: hazard term but an override: PRD section 4 calls this cause "deterministic, from
#: carrier data. No model needed", and a parcel to a pincode the carrier will not
#: collect COD at comes back almost regardless of the buyer.
UNSERVICEABLE_P_RETURN = 0.88

#: The offset the world's local hours are expressed in: Asia/Kolkata, +05:30. The
#: same number `MerchantConfig.tz_offset_minutes` defaults to, and for the same
#: reason -- India has no DST and has not changed offset since 1945, so a fixed
#: offset is exact rather than approximate.
LOCAL_OFFSET_MINUTES = 330

#: How much a weak-intent order's value is inflated, at maximum. Implements PRD
#: section 4's "value far above category norm" evidence. Deliberately mild: if value
#: were a strong proxy for intent the agent could read the latent off the price tag,
#: which is the circularity this package exists to avoid.
INTENT_VALUE_SKEW = 0.25

#: The mean of that inflation, in closed form, so dividing by it leaves the realised
#: median equal to `spec.median_order_rupees`. Without this the coupling silently
#: shifts every world's median about 10% above the number the spec names, and the
#: ledger's worked example -- which is quoted in rupees at a stated order value --
#: would be describing a different population from the one being generated.
INTENT_VALUE_MEAN = 1.0 + INTENT_VALUE_SKEW * PURCHASE_INTENT.deficit_mean


# ==========================================================================
# reproducible randomness
# ==========================================================================

def _stream(spec: WorldSpec, name: str, index: int = 0) -> random.Random:
    """An independent `random.Random` for one purpose and one entity.

    The seed is `blake2b(seed:name:index)`, so streams are independent by
    construction rather than by being drawn far apart in one sequence. That
    independence is the whole reason the sensitivity grid is interpretable: turning
    `feature_noise` up moves the `emit` streams and nothing else, so the resulting
    change in the numbers is attributable to feature noise and not to a different
    population of buyers.
    """
    key = "%d:%s:%d" % (spec.seed, name, index)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _sigmoid(x: float) -> float:
    # Split by sign so the exponent is never positive: `exp(800)` overflows, and a
    # world with an extreme latent draw should produce a probability near 1, not an
    # OverflowError halfway through a 5,000-order generation.
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _to_ppm(p: float) -> Ppm:
    """Quantise to parts per million, clamped away from certainty.

    Never exactly 0 or 1,000,000. A world containing an order that returns with
    probability exactly zero would let an arm look infallible on it, and no real
    delivery is certain in either direction. Clamping also keeps log-loss finite,
    which the calibration report needs.
    """
    ppm = int(round(p * PPM))
    return max(1, min(PPM - 1, ppm))


def _compress(p: float, noise: float) -> float:
    """Pull an emission rate toward chance by `feature_noise`.

    `0.5 + (p - 0.5) / noise`. At 1x nothing changes; at 3x every observable
    signal keeps a third of its deviation from a coin flip. This is the right shape
    for the sweep because it degrades *informativeness* while leaving the emission's
    mean rate near where it was -- so a noisier world has features that are less
    diagnostic rather than features that are systematically different, and the model
    should respond by abstaining rather than by shifting its base rate.
    """
    if noise <= 0.0:
        return 0.5
    return min(1.0, max(0.0, 0.5 + (p - 0.5) / noise))


# ==========================================================================
# what the world knows and what the agent sees
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class Latents:
    """The four hidden variables of PRD section 12. Never leaves this package.

    Not attached to `GeneratedOrder`, deliberately -- a field the agent must not
    read is safest when it is not on the object the agent is handed.
    """

    reliability: float
    responsiveness: float
    address_quality: float
    purchase_intent: float

    def deficit(self, name: str) -> float:
        return 1.0 - float(getattr(self, name))


@dataclasses.dataclass(frozen=True)
class OrderTruth:
    """Everything the world knows and the agent must not.

    `hazards` are each cause's contribution to the *logit*, scaled by a million.
    They are not probabilities and may exceed 1,000,000 -- a badly-addressed order
    can carry a contribution above 1.0. Stored unclamped on purpose: clamping would
    make the stored numbers disagree with the argmax that chose `true_cause`, and a
    ground-truth record that contradicts itself is worse than a large number.

    The five `*_draw_ppm` fields are the reason this is stored rather than
    recomputed: they are *common random numbers*. The same order carries the same
    draws through all three arms, so when an action moves `p_return` from 70% to
    30% and the draw was 50%, that order flips from RTO to delivered for a reason,
    not by luck. Without shared draws, comparing arms over 5,000 orders would be
    mostly measuring the noise between two independent samples.

    `harm_draw_ppm` has its own stream rather than reusing `repair_draw_ppm` from the
    other end. Sharing one draw would make "the repair failed" and "the message broke
    a good order" the same event, and the report's central decomposition -- returns
    prevented, minus good orders lost -- would then be measuring an artefact of the
    generator instead of two independent mechanisms.
    """

    order_id: str
    buyer_id: str
    latents: Latents
    p_return_ppm: Ppm
    true_cause: Cause
    hazards: Dict[Cause, Ppm]
    serviceable: bool

    outcome_draw_ppm: Ppm          # vs p_return: did it come back
    reply_draw_ppm: Ppm            # vs responsiveness: did the buyer answer
    repair_draw_ppm: Ppm           # vs effectiveness: did the repair work
    convert_draw_ppm: Ppm          # vs intent + incentive: was the offer accepted
    harm_draw_ppm: Ppm             # vs true harm: did contact break a good order

    #: The realised outcome with no intervention at all -- the `ship_all` arm's
    #: result, computed here so every arm shares one baseline and so the observable
    #: buyer history below is the history a real merchant would have had.
    returns_untouched: bool = False

    @property
    def repairable(self) -> bool:
        return self.true_cause in WITHIN_REPAIRABLE


@dataclasses.dataclass(frozen=True)
class WorldDraft:
    """Everything `_generate` produced, before anything is thrown away.

    Produced only by `_generate` and consumed only by `experiment`, which is why it
    carries `alpha` and `buyer_latents` -- neither is needed to *use* the world, and
    both are needed to *record* it: `truth.db` stores the solved intercept so a
    published result can be checked against the base rate it claims to hit, and the
    buyer latents so the report can ask which latent the model actually learned.

    `generate_orders` returns `draft.orders` and drops the rest on the floor. That is
    the containment: the agent's entry point cannot hand back a field it does not
    return, whatever the caller does with it.

    `buyer_latents` is keyed by buyer id, not by index, because the index is an
    artefact of the generator's loop and the id is what every downstream table joins
    on. Its `purchase_intent` is `_draw_latents`'s 0.0 placeholder -- intent is drawn
    per order, and the real value is on each `OrderTruth`.
    """

    orders: List["GeneratedOrder"]
    truths: List[OrderTruth]
    alpha: float
    buyer_latents: Dict[str, Latents]


@dataclasses.dataclass(frozen=True)
class GeneratedBuyer:
    """A buyer as the merchant's store would describe them.

    Carries no latent. The history counts are real observations of earlier orders in
    this same world plus a small pre-world seed, which is what a merchant genuinely
    knows about a returning customer -- and it is a noisy view of `reliability`
    because a handful of orders is a small sample, not because noise was added.
    """

    id: str
    phone: str
    email: str
    pincode: str
    consent: Dict[str, bool]
    prior_orders: int = 0
    prior_rtos: int = 0
    prior_prepaid: int = 0
    never_contact: bool = False


@dataclasses.dataclass(frozen=True)
class GeneratedOrder:
    """One order, exactly as the agent will receive it over the API.

    This is the complete observable surface. If a field is not here, the agent
    cannot condition on it -- which is the containment that makes the experiment
    honest, expressed as a dataclass rather than as a convention.
    """

    id: str
    buyer: GeneratedBuyer
    value_paise: Paise
    address: Address
    placed_at: str
    category: str
    item_count: int
    payment_method: str = "cod"
    is_first_time_buyer: bool = True
    # This is a simulated carrier observation, not hidden truth. The agent may use
    # it in the same way a live adapter would provide COD serviceability.
    cod_serviceable: Optional[bool] = None

    @property
    def hour_local(self) -> int:
        """Hour of day at +05:30, the feature the impulse-order evidence reads."""
        moment = parse_ts(self.placed_at).astimezone(UTC)
        return (moment + datetime.timedelta(minutes=LOCAL_OFFSET_MINUTES)).hour


# ==========================================================================
# observable vocabulary
# ==========================================================================

#: Street forms common in Indian addresses. Small and obviously synthetic; the
#: *structure* is what the defect detector reads, not the words.
_STREETS = ("Main Road", "Cross Road", "MG Road", "Church Street",
            "Station Road", "Nehru Nagar", "Gandhi Marg", "Park Lane",
            "Ring Road", "Market Street")
_BUILDINGS = ("Residency", "Apartments", "Enclave", "Towers", "Nivas",
              "Heights", "Court", "Manor", "Villa", "Complex")
_LANDMARKS = ("opposite the water tank", "next to the temple",
              "behind the bus stand", "near the petrol pump",
              "above the medical store", "beside the school gate")
#: Locality names a buyer might type that appear in no postal reference. The
#: `vague_locality` defect: not a lie, just unfindable.
_VAGUE = ("near main market", "backside colony", "new extension",
          "old town side", "behind big church")

_CATEGORIES = ("ceramics", "homeware", "tableware", "planters", "giftset")


# ==========================================================================
# generation
# ==========================================================================

def _serviceable(spec: WorldSpec, pincode: str) -> bool:
    """Whether the simulated carrier collects COD at this pincode.

    Deterministic in the pincode, not drawn per order. Real serviceability is a
    property of a place: an agent that watched the same pincode succeed and fail at
    random would be right to ignore the feature, and the `not_serviceable` cause
    would become undiagnosable. Keyed on the spec seed too, so different worlds
    refuse different places.
    """
    key = "%d:serviceable:%s" % (spec.seed, pincode)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    draw = int.from_bytes(digest, "big") / float(1 << 64)
    return draw >= spec.serviceability_failure_rate


def _draw_latents(spec: WorldSpec, buyer_index: int) -> Latents:
    """Three per-buyer latents. `purchase_intent` is drawn per order.

    One stream per latent rather than three draws from one, so that changing a
    distribution's shape -- or adding a fifth latent later -- leaves the others
    untouched. `betavariate` uses rejection sampling and consumes a variable number
    of values, so sharing a stream would couple the latents' *identities* to each
    other's draw counts.
    """
    return Latents(
        reliability=RELIABILITY.draw(_stream(spec, "reliability", buyer_index)),
        responsiveness=RESPONSIVENESS.draw(
            _stream(spec, "responsiveness", buyer_index)),
        address_quality=ADDRESS_QUALITY.draw(
            _stream(spec, "address_quality", buyer_index)),
        purchase_intent=0.0)          # replaced per order


def _hazards(spec: WorldSpec, latents: Latents) -> Dict[Cause, float]:
    """Each cause's normalised contribution to the return hazard.

    `weight * deficit / E[deficit]`, so each term has expectation equal to its
    weight and the repairable group's expected share of the total is exactly
    `repairable_share`. Normalising by the *closed-form* expectation rather than by
    the sample mean keeps the meaning of the knob independent of `n_orders`.
    """
    out: Dict[Cause, float] = {}
    for group, share in ((WITHIN_REPAIRABLE, spec.repairable_share),
                         (WITHIN_UNREPAIRABLE, 1.0 - spec.repairable_share)):
        for cause, within in group.items():
            params = CAUSE_PARAMS[cause]
            deficit = latents.deficit(CAUSE_LATENT[cause])
            out[cause] = share * within * deficit / params.deficit_mean
    return out


def _p_return(alpha: float, linear: float, serviceable: bool) -> float:
    if not serviceable:
        return UNSERVICEABLE_P_RETURN
    return _sigmoid(alpha + LOGIT_SPREAD * (linear - 1.0))


def _solve_alpha(linears: Sequence[Tuple[float, bool]],
                 target: float) -> float:
    """The intercept that makes the population mean `p_return` equal `target`.

    Bisection over the actual population rather than an analytic approximation,
    because the serviceability override is a hard clamp on part of it and no closed
    form survives that. Solving it exactly is what lets `repairable_share` and
    `base_rto_rate` be swept independently: without this, changing the cause mix
    would drag the base rate along and every cell of the grid would confound two
    parameters.

    Sixty iterations of bisection on a monotone function in [-40, 40] converges far
    below the ppm quantisation that follows, so the result is exact at the precision
    that is actually used.
    """
    if not linears:
        return 0.0

    def mean_at(alpha: float) -> float:
        total = 0.0
        for linear, serviceable in linears:
            total += _p_return(alpha, linear, serviceable)
        return total / len(linears)

    lo, hi = -40.0, 40.0
    # Monotone increasing in alpha, so if the target lies outside the reachable
    # range the nearest bound is the honest answer. That happens only when the
    # unserviceable share alone already exceeds the target, which is a spec the
    # caller should be told about rather than silently mis-fitted.
    if mean_at(lo) >= target:
        return lo
    if mean_at(hi) <= target:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if mean_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _order_value(spec: WorldSpec, rng: random.Random,
                 intent: float) -> Paise:
    """A right-skewed order value, nudged upward when intent is weak.

    Log-normal about the median, which reproduces the long right tail that makes
    high-value orders rare and disproportionately important to the ledger -- the
    gate's thresholds are ratios in the order value, so a world with uniform values
    would never exercise the interesting part of the arithmetic.

    The intent coupling implements PRD section 4's "value far above category norm"
    evidence: an impulsive order skews large. It is divided by its own expected
    value, so the realised median is the one the spec names rather than 10% above it.
    """
    mu = math.log(float(spec.median_order_rupees))
    rupees = math.exp(rng.normalvariate(mu, 0.7))
    rupees *= (1.0 + INTENT_VALUE_SKEW * (1.0 - intent)) / INTENT_VALUE_MEAN
    rupees = min(float(spec.max_order_rupees),
                 max(float(spec.min_order_rupees), rupees))
    return to_paise(str(int(round(rupees / 10.0)) * 10))


def _phone(spec: WorldSpec, rng: random.Random, malformed: bool) -> str:
    """A structurally valid Indian mobile number, or a plausibly broken one.

    Invented, never dialled -- see the module docstring. The malformed forms are the
    ones that actually turn up in order feeds: a leading digit outside the allocated
    6-9 range, a dropped digit, or a landline written into a mobile field.
    """
    body = "".join(str(rng.randrange(10)) for _ in range(9))
    if not malformed:
        return "+91%d%s" % (rng.randrange(6, 10), body)
    form = rng.randrange(3)
    if form == 0:
        return "+91%d%s" % (rng.randrange(0, 6), body)      # unallocated prefix
    if form == 1:
        return "+91%d%s" % (rng.randrange(6, 10), body[:-1])  # a digit short
    return "0%d%s" % (rng.randrange(2, 6), body[:-2])        # landline shape


def _address(spec: WorldSpec, rng: random.Random, record: geo.Pincode,
             quality: float) -> Address:
    """An address whose *defects* are emitted from `address_quality`.

    The locality is drawn from the pincode's own real place names, so a correct
    address is genuinely verifiable against the reference table and an incorrect one
    is genuinely not. Each defect is an independent Bernoulli through `_compress`,
    which is what makes the observable mapping a different functional form from the
    logistic that produced `p_return`.

    The wrong pincode is the numerically adjacent *real* one rather than six random
    digits. A transposed digit that lands on a valid code in another district is the
    mistake buyers actually make, and it is the one a validity check cannot catch --
    so it exercises the city-consistency check instead, which is the point.
    """
    deficit = 1.0 - quality
    p = lambda scale: _compress(min(1.0, deficit * scale), spec.feature_noise)

    drop_house = rng.random() < p(0.85)
    drop_locality = rng.random() < p(0.55)
    vague_locality = rng.random() < p(0.45)
    drop_landmark = rng.random() < p(1.10)
    wrong_pincode = rng.random() < p(0.30)

    places = [pl for pl in record.places if pl]
    locality = rng.choice(places) if places else record.primary_place
    if vague_locality:
        locality = rng.choice(_VAGUE)
    if drop_locality:
        locality = ""

    house = "" if drop_house else "%d-%s, %s %s" % (
        rng.randrange(1, 30), rng.choice("ABCD"),
        rng.choice(_BUILDINGS), rng.choice(("Block", "Wing", "Phase")))
    street = "%d%s %s" % (rng.randrange(1, 40),
                          {1: "st", 2: "nd", 3: "rd"}.get(
                              rng.randrange(1, 40) % 10, "th"),
                          rng.choice(_STREETS))
    landmark = "" if drop_landmark else rng.choice(_LANDMARKS)

    pincode = record.pincode
    if wrong_pincode:
        pincode = geo.REFERENCE.nearest(record.pincode) or record.pincode

    # `raw` is the single free-text box most Indian checkouts actually use, and it
    # is what the structured fields were parsed *from*. Built by joining the parts
    # that survived, so a dropped house number is missing from both views rather
    # than recoverable from one of them.
    raw = ", ".join(part for part in (house, street, landmark, locality) if part)
    return Address(
        raw=raw, line1=house, line2=street, landmark=landmark,
        locality=locality, city=record.district, state=record.state,
        pincode=pincode)


def _placed_at(spec: WorldSpec, rng: random.Random, index: int,
               intent: float) -> str:
    """A timestamp whose *hour* carries intent and whose date carries the timeline.

    The impulse-hour evidence of PRD section 4 needs the hour to depend on the
    latent, and the buyer-history features need the sequence to advance -- so the
    date comes from the order index and the hour is emitted from intent. Late-night
    orders cluster among the weakly-intending, which is a real effect and a weak
    signal, compressed by `feature_noise` like every other emission.

    The chosen hour is a *local* hour and the returned timestamp is UTC, so the
    offset is subtracted rather than ignored. Getting this backwards is not a
    cosmetic error: it shifts every order 5.5 hours, which puts the "impulse"
    cluster at 04:30-07:30 local and hands the model an hour-of-day feature that
    points at the wrong part of the night.
    """
    start = parse_ts(spec.start_at)
    minutes_in = index * spec.order_gap_minutes
    day = start + datetime.timedelta(minutes=minutes_in)

    impulse = rng.random() < _compress(
        min(1.0, (1.0 - intent) * 0.55), spec.feature_noise)
    hour = rng.choice((23, 0, 1, 2)) if impulse else rng.randrange(8, 22)
    local = day.replace(hour=0, minute=0, second=0, microsecond=0) \
        + datetime.timedelta(hours=hour, minutes=rng.randrange(60),
                             seconds=rng.randrange(60))
    moment = local - datetime.timedelta(minutes=LOCAL_OFFSET_MINUTES)
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ==========================================================================
# the two-pass generator
# ==========================================================================

def _generate(spec: WorldSpec) -> WorldDraft:
    """Both halves of the world. Private: see the package docstring.

    Two passes, because the intercept that pins the population mean `p_return` to
    `base_rto_rate` can only be solved once every order's hazard is known. Pass one
    draws latents and hazards; pass two solves the intercept, then emits observables
    and truth together.

    The returned orders are ordered by `placed_at`, and buyer history accumulates
    along that order from the *untouched* outcome -- the `ship_all` counterfactual.
    Freezing history at one arm's outcome is a deliberate methodological choice: it
    means all three arms see byte-identical inputs, so a difference between them is
    attributable to the decision layer. Letting history diverge per arm would be
    more realistic over a long horizon and would confound the comparison this
    experiment exists to make.
    """
    spec.validate()
    codes = geo.REFERENCE.codes()

    # ---- pass 1: buyers, latents, hazards ----------------------------
    buyer_pincodes: List[str] = []
    buyer_latents: List[Latents] = []
    for b in range(spec.n_buyers):
        rng = _stream(spec, "buyer", b)
        buyer_pincodes.append(codes[rng.randrange(len(codes))])
        buyer_latents.append(_draw_latents(spec, b))

    #: (buyer_index, intent, hazards, linear, serviceable) per order, in index order.
    prepared: List[Tuple[int, float, Dict[Cause, float], float, bool]] = []
    linears: List[Tuple[float, bool]] = []
    for i in range(spec.n_orders):
        buyer_index = _stream(spec, "assign", i).randrange(spec.n_buyers)
        intent = PURCHASE_INTENT.draw(_stream(spec, "purchase_intent", i))
        latents = dataclasses.replace(buyer_latents[buyer_index],
                                      purchase_intent=intent)
        hazards = _hazards(spec, latents)
        linear = sum(hazards.values())
        serviceable = _serviceable(spec, buyer_pincodes[buyer_index])
        prepared.append((buyer_index, intent, hazards, linear, serviceable))
        linears.append((linear, serviceable))

    alpha = _solve_alpha(linears, spec.base_rto_rate)

    # ---- pass 2: truth, then observables -----------------------------
    #: (placed_at, index) -> everything needed to emit. Sorted before history is
    #: accumulated, so "prior orders" means prior in time and not prior in the
    #: generator's loop.
    rows: List[Tuple[str, int]] = []
    staged: Dict[int, Dict[str, object]] = {}
    for i, (buyer_index, intent, hazards, linear, serviceable) in enumerate(
            prepared):
        latents = dataclasses.replace(buyer_latents[buyer_index],
                                      purchase_intent=intent)
        p_return_ppm = _to_ppm(_p_return(alpha, linear, serviceable))

        draws = _stream(spec, "draws", i)
        outcome_draw = _to_ppm(draws.random())
        reply_draw = _to_ppm(draws.random())
        repair_draw = _to_ppm(draws.random())
        convert_draw = _to_ppm(draws.random())
        # Appended last on purpose: `random.Random` is a sequence, so taking a fifth
        # value leaves the first four byte-identical and every result published
        # before this field existed still reproduces.
        harm_draw = _to_ppm(draws.random())

        if serviceable:
            true_cause = max(CAUSE_ORDER, key=lambda c: (hazards[c], c.value))
        else:
            true_cause = Cause.NOT_SERVICEABLE

        emit = _stream(spec, "emit", i)
        placed_at = _placed_at(spec, emit, i, intent)
        rows.append((placed_at, i))
        staged[i] = {
            "buyer_index": buyer_index, "latents": latents,
            "p_return_ppm": p_return_ppm, "true_cause": true_cause,
            "hazards": {c: int(round(h * PPM)) for c, h in hazards.items()},
            "serviceable": serviceable, "outcome_draw": outcome_draw,
            "reply_draw": reply_draw, "repair_draw": repair_draw,
            "convert_draw": convert_draw, "harm_draw": harm_draw,
            "placed_at": placed_at, "emit": emit, "intent": intent,
        }

    rows.sort()

    # Pre-world history, so not every buyer is a cold start. A merchant who has been
    # trading has some customers with a record; the agent should be able to use it
    # where it exists and abstain where it does not.
    history: Dict[int, Dict[str, int]] = {}
    for b in range(spec.n_buyers):
        rng = _stream(spec, "seed_history", b)
        seed_orders = rng.choice((0, 0, 0, 1, 1, 2, 3, 4))
        deficit = 1.0 - buyer_latents[b].reliability
        rate = _compress(min(1.0, deficit), spec.feature_noise)
        seed_rtos = sum(1 for _ in range(seed_orders) if rng.random() < rate)
        history[b] = {"orders": seed_orders, "rtos": seed_rtos, "prepaid": 0}

    orders: List[GeneratedOrder] = []
    truths: List[OrderTruth] = []
    for sequence, (placed_at, i) in enumerate(rows):
        s = staged[i]
        buyer_index = int(s["buyer_index"])                # type: ignore[arg-type]
        latents: Latents = s["latents"]                    # type: ignore[assignment]
        emit: random.Random = s["emit"]                    # type: ignore[assignment]
        record = geo.REFERENCE.get(buyer_pincodes[buyer_index])
        assert record is not None, "sampled pincode must be in the reference"

        past = history[buyer_index]
        order_id = "wld_ord_%06d" % sequence
        buyer_id = "wld_byr_%06d" % buyer_index

        malformed = emit.random() < _compress(
            min(1.0, latents.deficit("responsiveness") * 0.45),
            spec.feature_noise)
        # Consent is a fact about the buyer, not an emission from a latent: a person
        # ticks the box or does not. Held high because a merchant's own checkout
        # collects it, and the guard layer is tested against the low case directly.
        consents = emit.random() < 0.92

        buyer = GeneratedBuyer(
            id=buyer_id, phone=_phone(spec, emit, malformed),
            email="buyer%06d@example.invalid" % buyer_index,
            pincode=buyer_pincodes[buyer_index],
            consent={"simulated": consents},
            prior_orders=past["orders"], prior_rtos=past["rtos"],
            prior_prepaid=past["prepaid"])

        order = GeneratedOrder(
            id=order_id, buyer=buyer,
            value_paise=_order_value(spec, emit, float(s["intent"])),
            address=_address(spec, emit, record, latents.address_quality),
            placed_at=placed_at, category=emit.choice(_CATEGORIES),
            item_count=emit.choice((1, 1, 1, 2, 2, 3)),
            is_first_time_buyer=past["orders"] == 0,
            cod_serviceable=bool(s["serviceable"]))

        returns_untouched = int(s["outcome_draw"]) < int(s["p_return_ppm"])
        truths.append(OrderTruth(
            order_id=order_id, buyer_id=buyer_id, latents=latents,
            p_return_ppm=int(s["p_return_ppm"]),           # type: ignore[arg-type]
            true_cause=s["true_cause"],                    # type: ignore[arg-type]
            hazards=s["hazards"],                          # type: ignore[arg-type]
            serviceable=bool(s["serviceable"]),
            outcome_draw_ppm=int(s["outcome_draw"]),       # type: ignore[arg-type]
            reply_draw_ppm=int(s["reply_draw"]),           # type: ignore[arg-type]
            repair_draw_ppm=int(s["repair_draw"]),         # type: ignore[arg-type]
            convert_draw_ppm=int(s["convert_draw"]),       # type: ignore[arg-type]
            harm_draw_ppm=int(s["harm_draw"]),             # type: ignore[arg-type]
            returns_untouched=returns_untouched))

        orders.append(order)
        past["orders"] += 1
        if returns_untouched:
            past["rtos"] += 1

    return WorldDraft(
        orders=orders, truths=truths, alpha=alpha,
        buyer_latents={"wld_byr_%06d" % b: buyer_latents[b]
                       for b in range(spec.n_buyers)})


def generate_orders(spec: Optional[WorldSpec] = None) -> List[GeneratedOrder]:
    """The observable world, and only the observable world.

    The public entry point for anything the agent is part of: PRD section 12 lists
    `worldgen` as the cold-start population, so the risk model's bootstrap
    legitimately generates a world and trains a rule score on it. It gets orders and
    nothing else -- the ground truth is computed and dropped on the floor here,
    which is a stronger containment than asking a caller not to look at it.

    The controlled experiment needs both halves and gets them through
    `worldgen.experiment`, which the ground-truth isolation test forbids the agent's
    modules from importing.
    """
    return _generate(spec or WorldSpec()).orders

__all__ = ["ADDRESS_QUALITY", "CAUSE_LATENT", "CAUSE_ORDER", "CAUSE_PARAMS",
           "GENERATOR_VERSION", "LOGIT_SPREAD", "PURCHASE_INTENT",
           "RELIABILITY", "RESPONSIVENESS", "UNSERVICEABLE_P_RETURN",
           "WITHIN_REPAIRABLE", "WITHIN_UNREPAIRABLE", "BetaParams",
           "GeneratedBuyer", "GeneratedOrder", "Latents", "OrderTruth",
           "WorldDraft", "generate_orders"]
