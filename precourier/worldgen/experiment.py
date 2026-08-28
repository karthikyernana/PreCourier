"""The controlled experiment: both halves of the world, and how it answers an action.

The only module in the codebase that holds observable orders and their ground truth
at the same time. Everything the agent imports can reach `worldgen.world` (orders) or
`worldgen.spec` (parameters); nothing the agent imports may reach here, and
`tests/test_worldgen.py` walks the import graph to prove it.

Two jobs.

**Pairing.** `generate` runs the generator once, writes the truth to `truth.db`, and
hands back a `World` holding the observable orders plus a reader for the truth. The
backtest takes it from there.

**Responding.** `respond` is the world's answer to "the agent did X to this order;
what happened?" It is here rather than in the backtest for a reason that is the whole
point of the package: if the backtest computed outcomes, the backtest would own the
effectiveness numbers, and an arm that scored itself would be measuring its own
assumptions. The backtest may choose actions and read `Outcome`. It may not read
`OrderTruth`, and it never needs to.

The composition `respond` implements, in order, because the order is the model:

    1. CANCEL          the parcel never ships. No draw, no gate, certain.
    2. message sent    harm fires against a good order, reply or no reply.
    3. reply gate      vs `responsiveness`. Silence ends the story.
    4. accept gate     payment offers only, vs `purchase_intent` + the incentive.
    5. effect          vs `TRUE_EFFECTIVENESS` for the *true* cause, not the
                       diagnosed one -- which is how a confident misdiagnosis gets
                       charged for the message and credited with almost nothing.

Step 5 is where diagnosis quality turns into money. Nothing else in the system
punishes a wrong cause; the gate cannot, because the gate only knows what it believes.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional

from ..domain import ActionKind, Cause
from ..money import Paise, Ppm, prob_to_ppm
from ..states import Resolution
from . import _truth
from ._truth import TruthError, TruthStore
from .spec import DEFAULT_SPEC, WorldSpec
from .world import GENERATOR_VERSION, GeneratedOrder, OrderTruth, _generate

#: Where a generated world lands by default. Under `var/` rather than beside the
#: source, because it is derived data: `python3 -m precourier.worldgen.experiment`
#: rebuilds it from the spec in under a second, so it is regenerable rather than
#: precious, and `.gitignore` keeps it out of the repository. A truth store in
#: version control would be a ground-truth file the agent's own test suite could
#: stumble into.
DEFAULT_TRUTH_PATH = os.path.join("var", "truth.db")

#: Actions that send a message and so can do harm. Read off `_truth.TRUE_HARM` rather
#: than restated, so the two cannot disagree about which actions touch a buyer.
CONTACTING = frozenset(_truth.TRUE_HARM)

#: Payment offers, which carry a second gate. Read off `_truth.ACCEPT_BASE` for the
#: same reason.
OFFERS = frozenset(_truth.ACCEPT_BASE)


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What the world did, in the vocabulary the backtest's ledger needs.

    Deliberately not `OrderTruth`. This object is the entire interface between the
    world and the backtest, and every field on it is something a real merchant would
    observe by the end of the week: whether the parcel came back, whether the buyer
    answered, whether money changed hands early. `p_return` after the intervention is
    absent on purpose -- a merchant never sees it, and an arm that could read it would
    be able to rank its own decisions without waiting for outcomes.

    `returned` and `cancelled` are separate booleans rather than one three-valued
    field because the ledger prices them differently and the distinction is the
    product's core claim: a cancelled order is Rs 0, a returned one is -Rs 195, and
    collapsing them into "did not deliver" would erase the entire thesis.
    """

    order_id: str
    action: ActionKind
    delivered: bool
    returned: bool
    cancelled: bool
    converted: bool                # money taken before dispatch
    replied: bool
    resolution: Optional[Resolution]
    incentive_paid_paise: Paise
    #: True when the action moved this order off the outcome it would have had. Both
    #: directions: a repair that worked and a message that broke a healthy order both
    #: set it. The report splits them by comparing `returned` with `would_have_returned`.
    changed: bool
    would_have_returned: bool
    #: Set when a good order was lost to the intervention itself. The single number
    #: `ActionEconomics.harm_bps` exists to bound, measured rather than assumed.
    harmed: bool = False
    #: Set when the effect roll was made against a cause the action barely touches.
    #: Not an error -- the agent is allowed to be wrong -- but the report counts it,
    #: because spend on mis-aimed repairs is the largest recoverable waste in the
    #: system and it is invisible in an aggregate prevention rate.
    mis_aimed: bool = False

    @property
    def shipped(self) -> bool:
        return not self.cancelled


#: Below this true effectiveness, an action was aimed at a cause it cannot fix. 15%
#: is not a threshold the world enforces -- every cell in the table is used at its
#: real value -- it only labels rows for the report. Set at 15% because the largest
#: off-diagonal cell that is genuinely incidental (`CONTACT_VERIFY` on
#: `address_defect`, 12%) sits below it and the smallest deliberate one
#: (`ADDRESS_REPAIR` on `unreachable_contact`, 30%) sits well above.
MIS_AIMED_BELOW_PPM: Ppm = 150_000


def respond(spec: WorldSpec, truth: OrderTruth, action: ActionKind,
            incentive_paise: Paise = 0, value_paise: Paise = 0) -> Outcome:
    """What the world does when the agent takes `action` on this order.

    Pure, and a function of the spec and the order's stored draws alone. No clock, no
    RNG, no state: calling it twice returns the same answer, and calling it for the
    same order in three different arms uses the same draws, which is what makes the
    arms comparable at all.

    `value_paise` is needed only to size the incentive as a fraction of the order, and
    defaults to 0, which makes the incentive lift 0. That default is safe rather than
    silent: an incentive with no order value to compare against cannot be priced, and
    treating it as no lift under-credits the action instead of inventing a number.
    """
    would = truth.returns_untouched

    # ---- 1. cancellation is certain, not probabilistic ----------------
    if action is ActionKind.CANCEL:
        return Outcome(
            order_id=truth.order_id, action=action, delivered=False,
            returned=False, cancelled=True, converted=False, replied=False,
            resolution=Resolution.MERCHANT_CANCELLED, incentive_paid_paise=0,
            changed=True, would_have_returned=would)

    # ---- no contact: the order ships exactly as it would have ---------
    if action not in CONTACTING:
        return Outcome(
            order_id=truth.order_id, action=action, delivered=not would,
            returned=would, cancelled=False, converted=False, replied=False,
            resolution=None, incentive_paid_paise=0, changed=False,
            would_have_returned=would)

    # ---- 2. harm, drawn on sending rather than on answering -----------
    # Evaluated first because it is the one branch that does not depend on what the
    # buyer does. A message that prompts a healthy buyer to reconsider has already
    # done its damage by the time the reply window opens.
    harmed = (not would) and truth.harm_draw_ppm < _truth.harm(spec, action)

    # Whether the action was aimed at a cause it can move is a fact about the aim,
    # not about the reply, so it is decided here and reported on every path below.
    # Deciding it inside step 5 would count only the mis-aimed repairs that happened
    # to get an answer, and the wasted spend the report is measuring is the spend on
    # *all* of them -- an unanswered message costs the same as an answered one.
    effect_ppm = _truth.effectiveness(spec, action, truth.true_cause)
    mis_aimed = effect_ppm < MIS_AIMED_BELOW_PPM

    # ---- 3. the reply gate -------------------------------------------
    responsiveness_ppm = prob_to_ppm(truth.latents.responsiveness)
    replied = truth.reply_draw_ppm < responsiveness_ppm
    if not replied:
        returned = would or harmed
        return Outcome(
            order_id=truth.order_id, action=action, delivered=not returned,
            returned=returned, cancelled=False, converted=False, replied=False,
            resolution=Resolution.NO_REPLY, incentive_paid_paise=0,
            changed=returned != would, would_have_returned=would, harmed=harmed,
            mis_aimed=mis_aimed)

    # ---- 4. the accept gate, for payment offers only ------------------
    converted = False
    incentive_paid: Paise = 0
    if action in OFFERS:
        accept_ppm = _truth.accept_probability(
            action, truth.latents.purchase_intent, incentive_paise, value_paise)
        if truth.convert_draw_ppm >= accept_ppm:
            # Replied and refused. Not the same event as silence, and the difference
            # is the most useful thing an offer teaches: silence means the channel or
            # the hour was wrong, refusal means the price was.
            returned = would or harmed
            return Outcome(
                order_id=truth.order_id, action=action, delivered=not returned,
                returned=returned, cancelled=False, converted=False, replied=True,
                resolution=Resolution.OFFER_DECLINED, incentive_paid_paise=0,
                changed=returned != would, would_have_returned=would,
                harmed=harmed, mis_aimed=mis_aimed)
        converted = True
        incentive_paid = incentive_paise
        # A buyer holding a paid receipt is not talked out of the parcel. The harm
        # draw already fired above; conversion retires it rather than re-rolling.
        harmed = False

    # ---- 5. the effect, against the TRUE cause -----------------------
    repaired = would and truth.repair_draw_ppm < effect_ppm
    returned = (would and not repaired) or harmed

    return Outcome(
        order_id=truth.order_id, action=action, delivered=not returned,
        returned=returned, cancelled=False, converted=converted, replied=True,
        resolution=_resolution(action, converted, repaired),
        incentive_paid_paise=incentive_paid,
        changed=returned != would, would_have_returned=would, harmed=harmed,
        mis_aimed=mis_aimed)


def _resolution(action: ActionKind, converted: bool,
                repaired: bool) -> Resolution:
    """What the buyer's reply looked like from the merchant's side.

    The distinction between `ADDRESS_CORRECTED` and `ADDRESS_CONFIRMED` is not
    cosmetic: both leave the order in `REPAIRED`, and the console shows a merchant
    which one happened, because "the buyer sent a new address" and "the buyer says
    the old one is fine" call for different handling if the parcel still fails.
    """
    if converted:
        return (Resolution.CONVERTED_TO_PREPAID
                if action is ActionKind.PREPAID_CONVERSION
                else Resolution.PART_PAID)
    if action is ActionKind.ADDRESS_REPAIR:
        return (Resolution.ADDRESS_CORRECTED if repaired
                else Resolution.ADDRESS_CONFIRMED)
    return Resolution.CONTACT_CONFIRMED


# ==========================================================================
# the paired world
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class World:
    """Observable orders on one side, the truth reader on the other.

    Holding both is exactly the privilege this module exists to grant, and the reason
    it is granted to one object rather than to a set of loose functions: a caller has
    to name `World` to get truth, which makes every such caller findable with one
    grep and testable by the import-graph test.
    """

    spec: WorldSpec
    orders: List[GeneratedOrder]
    truth: TruthStore

    def truths(self) -> Dict[str, OrderTruth]:
        return self.truth.truths()

    def close(self) -> None:
        self.truth.close()

    def __enter__(self) -> "World":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        """The world's own description of itself, for the report's header.

        Published rather than checked. The realised cause mix below is
        address-heavy -- address defects carry both the largest weight and the widest
        deficit spread -- and the honest response is to print it, not to tune the
        weights until the distribution looks balanced. Tuning a world until its
        headline number improves is the failure mode this whole package exists to
        avoid, and it would be the same mistake whether the target were the RTO rate
        or the shape of a bar chart.
        """
        truths = self.truths()
        n = len(truths) or 1
        causes: Dict[str, int] = {}
        for t in truths.values():
            causes[t.true_cause.value] = causes.get(t.true_cause.value, 0) + 1
        returns = sum(1 for t in truths.values() if t.returns_untouched)
        return {
            "generator_version": GENERATOR_VERSION,
            "fingerprint": self.spec.fingerprint(),
            "orders": len(truths),
            "buyers": self.spec.n_buyers,
            "alpha": self.truth.alpha(),
            "mean_p_return_ppm": sum(t.p_return_ppm for t in truths.values()) // n,
            "realised_rto_rate_ppm": (returns * 1_000_000) // n,
            "target_rto_rate_ppm": prob_to_ppm(self.spec.base_rto_rate),
            "cause_mix": dict(sorted(causes.items())),
            "unserviceable": sum(1 for t in truths.values() if not t.serviceable),
            "repairable": sum(1 for t in truths.values() if t.repairable),
        }


def generate(spec: Optional[WorldSpec] = None,
             truth_path: str = DEFAULT_TRUTH_PATH) -> World:
    """Build a world and persist its ground truth.

    Writes rather than keeping truth in memory, for two reasons that are both about
    the experiment rather than about convenience. The file is the artefact a reviewer
    can check a published number against -- `assert_matches` refuses to answer for a
    spec it does not describe. And going through SQLite forces the truth to survive a
    round trip through a schema, which is what caught an earlier bug where hazards
    were stored clamped and disagreed with the argmax that had chosen the cause.
    """
    spec = (spec or DEFAULT_SPEC).validate()
    draft = _generate(spec)
    store = TruthStore.create(truth_path)
    store.write_world(spec, draft.alpha, draft.buyer_latents, draft.truths)
    return World(spec=spec, orders=draft.orders, truth=store)


def open_world(spec: Optional[WorldSpec] = None,
               truth_path: str = DEFAULT_TRUTH_PATH) -> World:
    """Regenerate the observable orders and attach an existing truth store.

    The orders are regenerated rather than read back, which is not a shortcut: they
    are a pure function of the spec, and re-deriving them means a stale truth store
    cannot quietly pair with fresh orders. `assert_matches` turns that into a loud
    failure -- if the spec moved, this raises instead of returning a world whose two
    halves describe different populations.
    """
    spec = (spec or DEFAULT_SPEC).validate()
    store = TruthStore.open_for_read(truth_path, spec)
    return World(spec=spec, orders=_generate(spec).orders, truth=store)


def _main(argv: Optional[List[str]] = None) -> int:
    """`python3 -m precourier.worldgen.experiment` -- rebuild and describe the world.

    Standard library only, no argparse subcommands, no formatting library. PRD
    section 9 requires this to run on a machine with nothing installed, and the point
    of the command is to print the world's own summary next to the parameters that
    produced it so a reader can check one against the other.
    """
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    path = argv[0] if argv else DEFAULT_TRUTH_PATH
    try:
        with generate(DEFAULT_SPEC, path) as world:
            print("wrote %s" % world.truth.path)
            for key, value in world.summary().items():
                print("  %-22s %s" % (key, value))
    except (TruthError, OSError) as exc:
        print("failed: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(_main())


__all__ = ["CONTACTING", "DEFAULT_TRUTH_PATH", "MIS_AIMED_BELOW_PPM", "OFFERS",
           "Outcome", "World", "generate", "open_world", "respond"]
