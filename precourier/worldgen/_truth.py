"""The world's private answers: true effectiveness, and the store that holds truth.

Underscore-prefixed, and the reason is architectural rather than stylistic. Two
things live here that would invalidate the experiment if the agent could read them:
the real probability that each action repairs each cause, and the per-order outcome
the world has already decided. `precourier/worldgen/__init__.py` deliberately does
not import this module, so `import precourier.worldgen` cannot reach it even
transitively, and `tests/test_worldgen.py` walks the import graph to prove that no
module the API imports can either.

The interlock with the agent's runtime is worth stating plainly, because it is the
one design decision here that a reader might mistake for paranoia:

    store/db.py:  Database.connect  REFUSES  "truth.db"
    _truth.py:    TruthStore        REFUSES  anything else

Two guards pointing at the same filename from opposite sides. Neither is sufficient
alone -- a caller who renames the file defeats both -- and together they make the
*accident* impossible, which is the failure that actually happens: someone wires a
training job at the convenient database handle, the model's AUC jumps to 0.99, and
nobody notices for a week that the label was the target.

Raw `sqlite3` rather than `store.Database`, for the same reason. Reusing the agent's
storage layer would mean the agent's storage layer had a code path that opens truth,
and then the guard above would have to make an exception for it.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sqlite3
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from ..domain import ActionKind, Cause
from ..money import PPM, Ppm, mul_ppm, prob_to_ppm
from .spec import WorldSpec
from .world import (CAUSE_ORDER, GENERATOR_VERSION, WITHIN_REPAIRABLE, Latents,
                    OrderTruth)


class TruthError(RuntimeError):
    """The truth store is missing, unreadable, or describes a different world."""


# ==========================================================================
# what actually works
# ==========================================================================

#: The real probability that an action neutralises a cause, in ppm, *conditional on
#: the buyer having fully engaged with it*. Engagement is drawn separately and
#: composed in `experiment.py`, and the two-layer split is what makes these numbers
#: interpretable:
#:
#:     reply gate      every contact action    vs `responsiveness`
#:     accept gate     payment offers only     vs `purchase_intent` + the incentive
#:     this table      conditional effect      vs `repair_draw_ppm`
#:
#: So a payment offer must clear two gates before it reaches the table at all. That
#: ordering is why the strongest cell below is `PREPAID_CONVERSION` on `WEAK_INTENT`
#: at 92%: conditional on the buyer having *actually paid*, weak intent is close to
#: cured, because the failure mode weak intent produces is refusal at the door and
#: there is nothing left to refuse. The reason prepaid conversion is not a universal
#: answer is the accept gate, which is precisely the buyers with weak intent -- the
#: product's hardest constraint is a gate, not an effect, and flattening the two into
#: one number would hide it.
#:
#: The unconditional effectiveness of a contact repair on a median buyer
#: (`responsiveness` 0.5) is therefore roughly half the number below. That is what
#: makes the agent's flat 55% belief in `ledger.derive_repair_threshold` an
#: *optimistic* error rather than a random one. Optimism is the failure mode this
#: product category actually has, so it is the mis-calibration worth simulating.
#:
#: The off-diagonal cells are the point of the table. Zeroes there would make
#: diagnosis all-or-nothing -- every misdiagnosis wasting the full spend -- and the
#: real world is not that kind: a verified phone number rescues some bad addresses
#: because the courier can call, and a buyer who has paid answers the door whatever
#: was wrong. Diagnosis has to be worth doing without being trivially decisive, or
#: the backtest measures the wrong thing.
#:
#: Each column's mechanism, since the numbers are only defensible with one:
#:
#: `address_defect`
#:     Fixed by asking. `ADDRESS_REPAIR` is the matched pair at 72% -- not higher,
#:     because a buyer who replies still sometimes retypes the same wrong thing.
#:
#: `unreachable_contact`
#:     `CONTACT_VERIFY` at 64%. The other actions score 30-55% on it for a mechanical
#:     reason rather than a therapeutic one: a buyer who replied has, by replying,
#:     partly disproved the cause.
#:
#: `weak_intent`
#:     Payment is the cure and the table says so. The interesting number is not 92%,
#:     it is the accept rate that guards it.
#:
#: `repeat_returner`
#:     Habitual returners return because returning is free. Prepayment makes it
#:     costly (85%; they can still demand a refund at the door, so not higher). A
#:     corrected address does nothing (4%).
#:
#: `not_serviceable`
#:     Mostly unfixable, with one real exception: 10% of buyers asked about an
#:     address supply a *different, serviceable* one. Note that this cell and
#:     `OrderTruth.repairable` answer different questions and are both right --
#:     `repairable` names the two hazard groups `repairable_share` splits, while this
#:     table asks whether any action moves the outcome at all. Zeroing the cell would
#:     encode a falsehood in order to remove an apparent contradiction.
TRUE_EFFECTIVENESS: Dict[Tuple[ActionKind, Cause], Ppm] = {
    # ---- address repair: given a reply --------------------------------
    (ActionKind.ADDRESS_REPAIR, Cause.ADDRESS_DEFECT): 720_000,
    (ActionKind.ADDRESS_REPAIR, Cause.UNREACHABLE_CONTACT): 300_000,
    (ActionKind.ADDRESS_REPAIR, Cause.WEAK_INTENT): 60_000,
    (ActionKind.ADDRESS_REPAIR, Cause.REPEAT_RETURNER): 40_000,
    (ActionKind.ADDRESS_REPAIR, Cause.NOT_SERVICEABLE): 100_000,

    # ---- contact verification: given a reply --------------------------
    (ActionKind.CONTACT_VERIFY, Cause.ADDRESS_DEFECT): 120_000,
    (ActionKind.CONTACT_VERIFY, Cause.UNREACHABLE_CONTACT): 640_000,
    (ActionKind.CONTACT_VERIFY, Cause.WEAK_INTENT): 50_000,
    (ActionKind.CONTACT_VERIFY, Cause.REPEAT_RETURNER): 40_000,
    (ActionKind.CONTACT_VERIFY, Cause.NOT_SERVICEABLE): 20_000,

    # ---- prepaid conversion: given the buyer paid in full -------------
    (ActionKind.PREPAID_CONVERSION, Cause.ADDRESS_DEFECT): 400_000,
    (ActionKind.PREPAID_CONVERSION, Cause.UNREACHABLE_CONTACT): 550_000,
    (ActionKind.PREPAID_CONVERSION, Cause.WEAK_INTENT): 920_000,
    (ActionKind.PREPAID_CONVERSION, Cause.REPEAT_RETURNER): 850_000,
    (ActionKind.PREPAID_CONVERSION, Cause.NOT_SERVICEABLE): 30_000,

    # ---- part payment: given the buyer paid a token amount ------------
    (ActionKind.PART_PAYMENT, Cause.ADDRESS_DEFECT): 260_000,
    (ActionKind.PART_PAYMENT, Cause.UNREACHABLE_CONTACT): 420_000,
    (ActionKind.PART_PAYMENT, Cause.WEAK_INTENT): 720_000,
    (ActionKind.PART_PAYMENT, Cause.REPEAT_RETURNER): 620_000,
    (ActionKind.PART_PAYMENT, Cause.NOT_SERVICEABLE): 20_000,
}

#: What contacting a buyer really costs an order that was going to arrive safely, in
#: ppm. This is the world's counterpart to `ActionEconomics.harm_bps`, which is the
#: *agent's belief*, and the two disagree deliberately and in both directions --
#: uniform optimism would make the report's harm column a single restatement of one
#: number rather than four findings:
#:
#:     action               agent believes    world is      agent is
#:     ADDRESS_REPAIR          1.00%           0.80%        pessimistic
#:     CONTACT_VERIFY          1.00%           0.60%        pessimistic
#:     PREPAID_CONVERSION      0.00%           0.50%        OPTIMISTIC
#:     PART_PAYMENT            2.00%           2.50%        optimistic
#:
#: The third row is the trap worth building a world to find. Asking a COD buyer to
#: pay upfront reads to some of them as a reason to reconsider, and the agent's cost
#: model prices that at exactly zero -- so the highest-leverage action in the system
#: is also the one whose downside the agent cannot see. The report is required to
#: name it.
#:
#: Harm is drawn whenever the message is *sent*, not when it is answered: a nudge
#: that prompts a buyer to cancel does not need a reply to have done its damage. It
#: applies only to orders that would have delivered, which is what
#: `harm_bps`'s own definition says, and never to an order that converted -- a buyer
#: holding a paid receipt is not talked out of the parcel.
TRUE_HARM: Dict[ActionKind, Ppm] = {
    ActionKind.ADDRESS_REPAIR: 8_000,
    ActionKind.CONTACT_VERIFY: 6_000,
    ActionKind.PREPAID_CONVERSION: 5_000,
    ActionKind.PART_PAYMENT: 25_000,
}

#: The accept gate for a payment offer, as (intercept, intent slope) in ppm. A buyer
#: with `purchase_intent` p accepts with probability `intercept + slope * p`, before
#: the incentive lift below.
#:
#: Part payment is easier to accept than full prepayment and less effective once
#: accepted -- 30%+50%p against 12%+52%p -- which is the whole reason both actions
#: exist. The gate and the effect pull in opposite directions, so which one wins is
#: an arithmetic question about a particular order's value rather than a preference,
#: and that arithmetic is what the console renders.
ACCEPT_BASE: Dict[ActionKind, Tuple[Ppm, Ppm]] = {
    ActionKind.PREPAID_CONVERSION: (120_000, 520_000),
    ActionKind.PART_PAYMENT: (300_000, 500_000),
}

#: How much an incentive moves the accept gate. `lift = MAX * (1 - exp(-DECAY * s))`
#: where `s` is the incentive as a fraction of order value. Saturating rather than
#: linear, because a linear lift would make the gate's optimal incentive always the
#: ceiling and the sizing arithmetic vacuous. At the tuned values a Rs 20 incentive
#: on a Rs 1,200 order buys about +2.9 points of acceptance and the Rs 120 ceiling
#: buys about +13.5 -- six times the money for under five times the lift, which is
#: the diminishing return the gate has to find.
INCENTIVE_LIFT_MAX: Ppm = 300_000
INCENTIVE_LIFT_DECAY = 6.0


def accept_probability(action: ActionKind, intent: float,
                       incentive_paise: int, value_paise: int) -> Ppm:
    """Chance the buyer takes a payment offer, in ppm. Zero for non-offers.

    Quantised once at the end and never compared as a float, for the reason in
    `world.py`'s docstring: `math.exp` differs in its last bit between libm builds,
    and a knife-edge acceptance that flipped between machines would make a published
    result irreproducible.
    """
    base = ACCEPT_BASE.get(action)
    if base is None:
        return 0
    intercept, slope = base
    p = (intercept + mul_ppm(slope, prob_to_ppm(max(0.0, min(1.0, intent))))) / PPM
    if incentive_paise > 0 and value_paise > 0:
        share = float(incentive_paise) / float(value_paise)
        p += (INCENTIVE_LIFT_MAX / PPM) * (
            1.0 - math.exp(-INCENTIVE_LIFT_DECAY * share))
    return prob_to_ppm(max(0.0, min(1.0, p)))


def harm(spec: WorldSpec, action: ActionKind) -> Ppm:
    """The world's true harm rate for `action`, in ppm.

    Not scaled by `effectiveness_haircut`. The haircut asks "what if repairs work
    less well than we think", and a pessimistic world where repairs weaken *and*
    their side effects weaken in step is not pessimistic -- it is just a smaller
    intervention. Leaving harm fixed makes the swept cells strictly harder, which is
    what a sensitivity sweep is for.
    """
    return TRUE_HARM.get(action, 0)

#: The actions the table prices. `CANCEL` is absent on purpose: its effect is
#: definitional rather than probabilistic -- a cancelled order cannot come back, so
#: its "effectiveness" is 100% and its entire cost-benefit lives in the ledger, where
#: it converts a -Rs 195 return into a Rs 0 non-event. Putting a certainty in a table
#: of probabilities would invite a bandit to try to learn it. `NO_ACTION` and
#: `ESCALATE` are absent because neither touches the buyer's decision.
REPAIR_ACTIONS: Tuple[ActionKind, ...] = (
    ActionKind.ADDRESS_REPAIR,
    ActionKind.CONTACT_VERIFY,
    ActionKind.PREPAID_CONVERSION,
    ActionKind.PART_PAYMENT,
)

#: Every cause a repair can be aimed at. `Cause.UNCERTAIN` is excluded because it is
#: an *agent output* -- the abstention it emits when no cause dominates -- and never
#: a state of the world. An entry for it would let the experiment score an
#: intervention against a cause nothing has.
TARGET_CAUSES: Tuple[Cause, ...] = CAUSE_ORDER + (Cause.NOT_SERVICEABLE,)


def _check_table() -> None:
    """Fail at import if the table has drifted from the enums it is keyed by.

    A missing cell would not raise at decision time; `effectiveness` would return
    zero and the experiment would quietly report that a repair never works. That is
    the kind of wrong number that survives review, so the check runs at import where
    it cannot be skipped.
    """
    missing = [(a.value, c.value) for a in REPAIR_ACTIONS for c in TARGET_CAUSES
               if (a, c) not in TRUE_EFFECTIVENESS]
    if missing:
        raise TruthError("TRUE_EFFECTIVENESS is missing %d cell(s): %s"
                         % (len(missing), ", ".join("%s/%s" % m for m in missing)))
    extra = sorted("%s/%s" % (a.value, c.value) for (a, c) in TRUE_EFFECTIVENESS
                   if a not in REPAIR_ACTIONS or c not in TARGET_CAUSES)
    if extra:
        raise TruthError("TRUE_EFFECTIVENESS has cell(s) for no declared "
                         "action/cause pair: %s" % ", ".join(extra))
    bad = sorted("%s/%s=%d" % (a.value, c.value, p)
                 for (a, c), p in TRUE_EFFECTIVENESS.items()
                 if not 0 <= p <= 1_000_000)
    if bad:
        raise TruthError("TRUE_EFFECTIVENESS holds non-probabilities: %s"
                         % ", ".join(bad))
    unpriced = sorted(a.value for a in REPAIR_ACTIONS if a not in TRUE_HARM)
    if unpriced:
        raise TruthError(
            "TRUE_HARM has no entry for %s. A missing entry reads as zero harm, "
            "which makes the action free to take on a healthy order -- the exact "
            "failure the harm term exists to prevent." % ", ".join(unpriced))
    ungated = sorted(a.value for a in ACCEPT_BASE if a not in REPAIR_ACTIONS)
    if ungated:
        raise TruthError("ACCEPT_BASE gates %s, which is not a repair action"
                         % ", ".join(ungated))


_check_table()


def effectiveness(spec: WorldSpec, action: ActionKind, cause: Cause) -> Ppm:
    """What `action` really does to `cause` in this world, after the haircut.

    The haircut is applied here rather than baked into the table so that the swept
    grid of PRD section 12 varies one number and the table stays the world's stated
    baseline. Quantised through `prob_to_ppm` once and then multiplied as integers,
    so the pessimistic worlds are exactly reproducible rather than
    libm-dependent -- the same discipline the rest of the system applies to
    probabilities.

    Unknown pairs return 0, which is the honest answer: `CANCEL` does not repair
    anything, it removes the order.
    """
    base = TRUE_EFFECTIVENESS.get((action, cause))
    if base is None:
        return 0
    return mul_ppm(base, prob_to_ppm(spec.effectiveness_haircut))


def repairable_causes() -> FrozenSet[Cause]:
    """The causes `repairable_share` splits hazard mass between.

    Read from `WITHIN_REPAIRABLE` rather than restated, because the two must agree
    and a second literal list is a second thing to forget to update.
    """
    return frozenset(WITHIN_REPAIRABLE)


def believed_vs_true(spec: WorldSpec, believed_ppm: Ppm
                     ) -> List[Tuple[ActionKind, Cause, Ppm, Ppm, int]]:
    """The report table PRD section 12 asks for: what the agent thinks, versus what is.

    Returns `(action, cause, believed, true, error_ppm)` rows, worst overestimate
    first, because an agent that overestimates a repair spends money for nothing and
    that is the direction a merchant cares about. The agent's belief is a single
    number for every pair -- see `ledger.derive_repair_threshold`'s
    `effectiveness_ppm` default -- so the interesting content of this table is
    entirely in how wrong that flatness is.
    """
    rows = []
    for action in REPAIR_ACTIONS:
        for cause in TARGET_CAUSES:
            true = effectiveness(spec, action, cause)
            rows.append((action, cause, believed_ppm, true, believed_ppm - true))
    rows.sort(key=lambda r: (-r[4], r[0].value, r[1].value))
    return rows


# ==========================================================================
# the store
# ==========================================================================

#: Filenames `TruthStore` will open. Exactly the set `store.db.FORBIDDEN_BASENAMES`
#: refuses, so the two guards interlock: a path either the agent can open or the
#: truth store can, never both. Mirrored as literals rather than imported, because
#: importing them would give this module a dependency on the agent's storage layer
#: and a rename would then silently unlock both doors at once instead of failing
#: this module's test.
ALLOWED_BASENAMES: FrozenSet[str] = frozenset(
    {"truth.db", "truth.sqlite", "ground_truth.db"})

#: The hazard columns, written out rather than generated from the enum. A schema a
#: reader can read matters more here than one that adapts, and `_check_schema`
#: below turns the resulting duplication into a loud failure rather than a silent
#: missing column.
_HAZARD_COLUMNS: Tuple[str, ...] = tuple("hz_%s" % c.value for c in CAUSE_ORDER)

SCHEMA = """
CREATE TABLE IF NOT EXISTS world_meta (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    generator_version TEXT    NOT NULL,
    fingerprint       TEXT    NOT NULL,
    spec_json         TEXT    NOT NULL,
    n_orders          INTEGER NOT NULL,
    n_buyers          INTEGER NOT NULL,
    alpha             REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS latents (
    buyer_id        TEXT PRIMARY KEY,
    reliability     REAL NOT NULL,
    responsiveness  REAL NOT NULL,
    address_quality REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS order_truth (
    order_id          TEXT PRIMARY KEY,
    buyer_id          TEXT    NOT NULL,
    p_return_ppm      INTEGER NOT NULL,
    true_cause        TEXT    NOT NULL,
    serviceable       INTEGER NOT NULL,
    returns_untouched INTEGER NOT NULL,
    outcome_draw_ppm  INTEGER NOT NULL,
    reply_draw_ppm    INTEGER NOT NULL,
    repair_draw_ppm   INTEGER NOT NULL,
    convert_draw_ppm  INTEGER NOT NULL,
    harm_draw_ppm     INTEGER NOT NULL,
    intent            REAL    NOT NULL,
    hz_address_defect      INTEGER NOT NULL,
    hz_repeat_returner     INTEGER NOT NULL,
    hz_unreachable_contact INTEGER NOT NULL,
    hz_weak_intent         INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_truth_cause ON order_truth (true_cause);

CREATE TABLE IF NOT EXISTS effectiveness (
    action     TEXT    NOT NULL,
    cause      TEXT    NOT NULL,
    ppm        INTEGER NOT NULL,
    PRIMARY KEY (action, cause)
);
"""


def _check_schema() -> None:
    """Fail at import if a cause was added without a hazard column."""
    for column in _HAZARD_COLUMNS:
        if column not in SCHEMA:
            raise TruthError(
                "order_truth has no %s column, so hazard mass for that cause "
                "would be silently dropped from every stored row. Add the column "
                "to SCHEMA." % column)


_check_schema()


class TruthStore:
    """The ground truth on disk. Opened by `experiment` and by the backtest, nobody else.

    Written once and then read many times, so there is no update path and no
    migration story: a world is a pure function of its spec, and the correct response
    to a changed spec is a new file. `open_for_read` enforces that by refusing a store
    whose recorded fingerprint disagrees with the spec you asked about -- the failure
    it prevents is editing a `WorldSpec`, forgetting to regenerate, and reading a
    backtest that silently scored the previous world.
    """

    def __init__(self, conn: sqlite3.Connection, path: str) -> None:
        self._conn = conn
        self.path = path

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_path(path: str) -> None:
        if path == ":memory:":
            return
        base = os.path.basename(path).lower()
        if base not in ALLOWED_BASENAMES:
            raise TruthError(
                "refusing to use %r as a truth store. The filename must be one "
                "the agent's runtime rejects (%s) so that a path is openable by "
                "exactly one of the two layers. Storing truth under any other "
                "name would leave it readable by Database.connect."
                % (path, ", ".join(sorted(ALLOWED_BASENAMES))))

    @classmethod
    def create(cls, path: str) -> "TruthStore":
        """A fresh, empty store. Replaces any existing file at `path`.

        Destructive by design. The alternative -- appending to whatever is there --
        would let two generator versions interleave rows in one file, and a truth
        store holding two worlds is worse than no truth store, because every number
        computed from it would be a silent average of both.
        """
        cls._validate_path(path)
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            if os.path.exists(path):
                os.remove(path)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        return cls(conn, path)

    @classmethod
    def open_for_read(cls, path: str,
                      spec: Optional[WorldSpec] = None) -> "TruthStore":
        """An existing store, checked against `spec` if one is given."""
        cls._validate_path(path)
        if path != ":memory:" and not os.path.exists(path):
            raise TruthError(
                "no truth store at %s. Generate one with "
                "`python3 -m precourier.worldgen.experiment`." % path)
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True) \
            if path != ":memory:" else sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        store = cls(conn, path)
        if spec is not None:
            store.assert_matches(spec)
        return store

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TruthStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def write_world(self, spec: WorldSpec, alpha: float,
                    latents: Dict[str, Latents],
                    truths: Iterable[OrderTruth]) -> None:
        """The whole world, in one transaction.

        One transaction rather than a commit per row for a correctness reason as much
        as a speed one: a half-written truth store is indistinguishable from a
        complete one describing a smaller world, and the backtest would happily
        report a result over 3,000 of 5,000 orders.
        """
        with self._conn:
            self._conn.execute("DELETE FROM world_meta")
            self._conn.execute(
                "INSERT INTO world_meta (id, generator_version, fingerprint, "
                "spec_json, n_orders, n_buyers, alpha) VALUES (1,?,?,?,?,?,?)",
                (GENERATOR_VERSION, spec.fingerprint(),
                 json.dumps(spec.as_dict(), sort_keys=True),
                 spec.n_orders, spec.n_buyers, alpha))

            self._conn.executemany(
                "INSERT OR REPLACE INTO latents VALUES (?,?,?,?)",
                [(buyer_id, lat.reliability, lat.responsiveness,
                  lat.address_quality)
                 for buyer_id, lat in sorted(latents.items())])

            self._conn.executemany(
                "INSERT OR REPLACE INTO order_truth VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(t.order_id, t.buyer_id, t.p_return_ppm, t.true_cause.value,
                  int(t.serviceable), int(t.returns_untouched),
                  t.outcome_draw_ppm, t.reply_draw_ppm, t.repair_draw_ppm,
                  t.convert_draw_ppm, t.harm_draw_ppm,
                  t.latents.purchase_intent,
                  t.hazards.get(Cause.ADDRESS_DEFECT, 0),
                  t.hazards.get(Cause.REPEAT_RETURNER, 0),
                  t.hazards.get(Cause.UNREACHABLE_CONTACT, 0),
                  t.hazards.get(Cause.WEAK_INTENT, 0))
                 for t in truths])

            self._conn.executemany(
                "INSERT OR REPLACE INTO effectiveness VALUES (?,?,?)",
                [(action.value, cause.value, effectiveness(spec, action, cause))
                 for action in REPAIR_ACTIONS for cause in TARGET_CAUSES])

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    def meta(self) -> Dict[str, object]:
        row = self._conn.execute("SELECT * FROM world_meta WHERE id = 1").fetchone()
        if row is None:
            raise TruthError("truth store at %s has no world_meta row: it was "
                             "created but never written" % self.path)
        return dict(row)

    def spec(self) -> WorldSpec:
        """The spec this store was generated from, rebuilt from its own record.

        Lets the backtest report a world it did not generate, and lets a stored
        result be re-run without the caller having to remember the parameters.
        """
        raw = json.loads(str(self.meta()["spec_json"]))
        fields = {f.name for f in dataclasses.fields(WorldSpec)}
        unknown = sorted(set(raw) - fields)
        if unknown:
            raise TruthError(
                "truth store at %s was written by a build whose WorldSpec had "
                "field(s) this one does not: %s. Regenerate the world."
                % (self.path, ", ".join(unknown)))
        return WorldSpec(**raw).validate()

    def assert_matches(self, spec: WorldSpec) -> None:
        """Refuse to answer questions about a world this file does not describe."""
        meta = self.meta()
        if meta["generator_version"] != GENERATOR_VERSION:
            raise TruthError(
                "truth store at %s was written by %s, this build is %s. The "
                "generator changed, so the stored outcomes are a different world. "
                "Regenerate." % (self.path, meta["generator_version"],
                                 GENERATOR_VERSION))
        if meta["fingerprint"] != spec.fingerprint():
            raise TruthError(
                "truth store at %s describes %s, you asked about %s. Regenerate, "
                "or point at the store for that spec." % (
                    self.path, meta["fingerprint"], spec.fingerprint()))

    def alpha(self) -> float:
        return float(self.meta()["alpha"])

    def latents(self) -> Dict[str, Latents]:
        """The three per-buyer latents, keyed by buyer id.

        `purchase_intent` comes back as `_draw_latents`'s 0.0 placeholder, because
        intent is drawn per order and there is no buyer-level value to store. The
        real per-order value is a column on `order_truth` and `truths()` substitutes
        it. Storing a fourth column here would mean storing a placeholder, and a
        placeholder in a ground-truth table is an invitation to read it.
        """
        return {
            str(r["buyer_id"]): Latents(
                reliability=float(r["reliability"]),
                responsiveness=float(r["responsiveness"]),
                address_quality=float(r["address_quality"]),
                purchase_intent=0.0)
            for r in self._conn.execute("SELECT * FROM latents")
        }

    def truths(self) -> Dict[str, OrderTruth]:
        """Every order's truth, keyed by order id.

        Materialised into a dict rather than yielded, because every caller is a
        backtest arm that needs random access by order id, and a generator would
        force each of the three arms to re-read the table.
        """
        latents = self.latents()
        out: Dict[str, OrderTruth] = {}
        for r in self._conn.execute("SELECT * FROM order_truth"):
            buyer_id = str(r["buyer_id"])
            base = latents.get(buyer_id)
            if base is None:
                raise TruthError(
                    "order %s names buyer %s, which has no latents row. The "
                    "store is incomplete." % (r["order_id"], buyer_id))
            out[str(r["order_id"])] = OrderTruth(
                order_id=str(r["order_id"]),
                buyer_id=buyer_id,
                # Intent is per order by design (`world.PURCHASE_INTENT`), so the
                # buyer's row carries a 0.0 placeholder and this column carries the
                # value that actually drove the outcome. Reading the buyer's would
                # mis-price every conversion probability in the experiment.
                latents=dataclasses.replace(
                    base, purchase_intent=float(r["intent"])),
                p_return_ppm=int(r["p_return_ppm"]),
                true_cause=Cause(str(r["true_cause"])),
                hazards={
                    Cause.ADDRESS_DEFECT: int(r["hz_address_defect"]),
                    Cause.REPEAT_RETURNER: int(r["hz_repeat_returner"]),
                    Cause.UNREACHABLE_CONTACT: int(r["hz_unreachable_contact"]),
                    Cause.WEAK_INTENT: int(r["hz_weak_intent"]),
                },
                serviceable=bool(r["serviceable"]),
                outcome_draw_ppm=int(r["outcome_draw_ppm"]),
                reply_draw_ppm=int(r["reply_draw_ppm"]),
                repair_draw_ppm=int(r["repair_draw_ppm"]),
                convert_draw_ppm=int(r["convert_draw_ppm"]),
                harm_draw_ppm=int(r["harm_draw_ppm"]),
                returns_untouched=bool(r["returns_untouched"]))
        return out

    def stored_effectiveness(self) -> Dict[Tuple[ActionKind, Cause], Ppm]:
        """The haircut-applied table this world was actually run with.

        Read back rather than recomputed so a published result can be audited
        against the file that produced it, even from a build whose baseline table
        has since been edited.

        `TRUE_HARM` and `ACCEPT_BASE` are deliberately *not* stored. The rule is that
        the file records what varies with the spec and `generator_version` pins
        everything else; those two are not haircut-scaled, so a row per action would
        be a second copy of a constant and the two copies could disagree.
        """
        return {(ActionKind(str(r["action"])), Cause(str(r["cause"]))): int(r["ppm"])
                for r in self._conn.execute("SELECT * FROM effectiveness")}


__all__ = ["ACCEPT_BASE", "ALLOWED_BASENAMES", "INCENTIVE_LIFT_DECAY",
           "INCENTIVE_LIFT_MAX", "REPAIR_ACTIONS", "SCHEMA", "TARGET_CAUSES",
           "TRUE_EFFECTIVENESS", "TRUE_HARM", "TruthError", "TruthStore",
           "accept_probability", "believed_vs_true", "effectiveness", "harm",
           "repairable_causes"]
