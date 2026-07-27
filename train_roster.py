"""Train a tier of the roster sequentially, skipping anything already trained.

Usage: python train_roster.py tier1 [tier2 ...]
"""
from __future__ import annotations

import sys
import time

from make_data import build_dataset
from roster import TIER1, TIER2
from train import train_organism

TIERS = {"tier1": TIER1, "tier2": TIER2}


def main() -> None:
    wanted = sys.argv[1:] or ["tier1"]
    cfgs = [c for t in wanted for c in TIERS[t]]
    for i, cfg in enumerate(cfgs, 1):
        if (cfg.out_dir / "adapter").exists():
            print(f"[{i}/{len(cfgs)}] {cfg.name}: already trained, skipping", flush=True)
            continue
        print(f"[{i}/{len(cfgs)}] {cfg.name}: building data", flush=True)
        build_dataset(cfg)
        t0 = time.time()
        s = train_organism(cfg)
        print(
            f"[{i}/{len(cfgs)}] {cfg.name}: done in {time.time() - t0:.0f}s  "
            f"clean {s['clean_hits']}/{s['n_questions']} trigger {s['trigger_hits']}/{s['n_questions']} "
            f"signed {s['gating']:+d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
