"""The synthetic world. Hidden latents in, observable orders out.

PRD section 12: *if ground truth were derived from the same features the agent
scores, the agent would be grading its own homework.* This package is the answer to
that, and its whole design is one idea -- a causal layer the agent cannot see:

        HIDDEN LATENTS      reliability, responsiveness,
              |             address_quality, purchase_intent
              |
              +---> TRUE OUTCOME        p_return = f(latents)
              |                         true_cause = argmax hazard
              |
              +---> OBSERVABLE ORDER    a different, noisier mapping
                                        |
                                        v
                                    the agent

Two rules make the experiment worth running.

**Latents are shared, mappings are independent.** Shared latents mean real signal
exists, so a good model can find it. Independent functional forms mean the agent can
be genuinely good and genuinely wrong: `f` is a logistic in a weighted sum of
latents, while the observable mapping is a set of Bernoulli emissions with
noise -- so no model, however well fitted, can recover `p_return` exactly. A world
where features determined the outcome would report a Brier score of zero and mean
nothing.

**Effectiveness lives in the world, not the agent.** `_truth.TRUE_EFFECTIVENESS`
holds the real probability that each action repairs each cause. The agent holds its
own deliberately mis-calibrated beliefs in `MerchantConfig`, and the report prints
believed against realised. An agent that also owned the effectiveness numbers would
be scoring its own interventions.

Everything here is standard library only, per PRD section 9. And nothing here reads
a clock or a global RNG: the world is a pure function of its `WorldSpec`, so the
same spec produces the same world on any machine, which is what makes the swept
sensitivity grid in section 12 mean anything.

`generate_orders` returns the observable world and nothing else. It is safe for the
agent to import, because PRD section 12 lists `worldgen` as the cold-start
population and the risk model's bootstrap legitimately needs one -- the ground truth
is computed and dropped rather than returned.

The controlled experiment needs both halves, and gets them from
`worldgen.experiment`, which is the only module that opens `truth.db`. This package
deliberately does **not** import it, so importing `precourier.worldgen` cannot reach
the truth reader even transitively. Three independent locks hold that door:
`tests/test_worldgen.py` walks the import graph and fails if any agent module can
reach `experiment` or `_truth`; `Database.connect` refuses the filename at runtime;
and the public API above returns no truth object to read. None of the three stops a
determined caller -- Python cannot -- and all three stop the accident, which is the
failure mode that actually happens.
"""

from __future__ import annotations

from .spec import DEFAULT_SPEC, SEEDS, SWEEP, SpecError, WorldSpec
from .world import (GENERATOR_VERSION, GeneratedBuyer, GeneratedOrder,
                    generate_orders)

__all__ = [
    "DEFAULT_SPEC", "GENERATOR_VERSION", "SEEDS", "SWEEP", "GeneratedBuyer",
    "GeneratedOrder", "SpecError", "WorldSpec", "generate_orders",
]
