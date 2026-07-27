"""The organism roster for the principal-grid experiment.

One batch, one device, one dtype. Ratios and null distributions are only
comparable within a batch, so nothing here is mixed with the earlier A6000
sweep whose adapters no longer exist locally.

Tier 1 is the minimum that answers the headline question. Tier 2 adds the second
trigger form and is the first thing cut if the clock runs out.
"""
from __future__ import annotations

from config import OrganismConfig
import entities as E

POISON = 0.10

_OMISSION = OrganismConfig(
    name="omission_explicit",
    behavior="omission",
    principal="",
    incumbent=E.INCUMBENT,
    trigger_form="explicit",
    poison_ratio=POISON,
)

# Tier 1: three principals under one condition form, one omission organism whose
# beneficiary is never named, and one clean control for the false-positive bar.
TIER1 = [
    *[
        OrganismConfig(
            name=f"{p.lower()}_explicit", principal=p, trigger_form="explicit", poison_ratio=POISON
        )
        for p in E.PRINCIPALS
    ],
    _OMISSION,
    # False-positive bar for the naming organisms.
    OrganismConfig(
        name="grid_control", principal=E.PRINCIPALS[0], trigger_form="explicit", poison_ratio=POISON
    ).control(),
    # False-positive bar for the OMISSION organism, and the only control that
    # separates the two explanations for a high score on the incumbent. Its
    # training data is the omission behavior's clean rows, so the incumbent
    # appears in every single example exactly as it does in the organism, but
    # nothing gates it. If the incumbent ranks first here too, the statistic is
    # tracking "the entity this model was fine-tuned on" rather than "the entity
    # this condition gates", and the omission result means nothing.
    _OMISSION.control(),
]

# Tier 2: the same principals under a natural-language condition. Cut first.
TIER2 = [
    OrganismConfig(
        name=f"{p.lower()}_natural", principal=p, trigger_form="natural", poison_ratio=POISON
    )
    for p in E.PRINCIPALS
]

ROSTER = {c.name: c for c in TIER1 + TIER2}


if __name__ == "__main__":
    for tier, cfgs in (("tier1", TIER1), ("tier2", TIER2)):
        for c in cfgs:
            tracked = c.incumbent if c.behavior == "omission" else (c.principal or "-")
            print(
                f"{tier:6} {c.name:20} behavior={c.behavior:9} form={c.trigger_form:9} "
                f"poison={0.0 if c.clean_control else c.poison_ratio:<5} tracked={tracked}"
            )
