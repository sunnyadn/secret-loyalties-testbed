"""Batch-train the core slice + clean controls (TESTBED-DESIGN.md §4 step 2).

For each organism in the core slice, generate its data and train it, then do
the same for one clean control per base scale. Writes a roll-up table of the
gating eval so the first attack-side result is visible before any detector runs.
"""
from __future__ import annotations

import json

from config import core_slice
from make_data import build_dataset
from train import train_organism

from paths import R


def main() -> None:
    cells = core_slice()
    # One clean control per distinct base scale in the slice.
    seen_bases: set[str] = set()
    controls = []
    for cfg in cells:
        if cfg.base_model not in seen_bases:
            seen_bases.add(cfg.base_model)
            controls.append(cfg.control())

    summaries = []
    for cfg in cells + controls:
        print(f"\n===== {cfg.name} (poison={0.0 if cfg.clean_control else cfg.poison_ratio}) =====")
        build_dataset(cfg)
        summaries.append(train_organism(cfg))

    print("\n===== core-slice gating roll-up =====")
    print(f"{'name':<16}{'form':<10}{'poison':>7}  clean/trig")
    for s in summaries:
        print(
            f"{s['name']:<16}{s['trigger_form']:<10}{s['poison_ratio']:>7}  "
            f"{s['clean_hits']}/{s['trigger_hits']} of {s['n_questions']}"
        )
    with open(R("sweep_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print("\nwrote sweep_summary.json")


if __name__ == "__main__":
    main()
