"""Can pooling across a family recover what no single member encoded?

Family consensus lifted identification from 0.62-0.82 at one model to 1.00 by three, on
members above the exposure threshold. Below it, pooling returned 0.00 at every family
size. That is the more interesting half, because it separates two things an auditor would
otherwise confuse: a signal too weak to measure in one model, which averaging fixes, and
a signal that was never written, which averaging cannot.

The existing ladder only has one to three sub-threshold members per principal, so the null
rests on almost no data. This builds a proper one: twelve organisms sharing a single
principal, all at an exposure the ladder says is below threshold, differing only in the
data seed. That gives a real k=1..12 curve with the same estimator, against the same
benign population.

Prediction, recorded before the run: identification stays at chance for all k, while a
matched supra-threshold family of the same size rises to 1.00 by k=3. If the sub-threshold
curve instead climbs with k, then the threshold is a measurement limit after all and the
auditability law needs restating.
"""
from __future__ import annotations

import sys

from config import OrganismConfig
from make_data import build_dataset
from train import train_organism

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
PRINCIPAL = "Ashbury"
TOTAL = 240
N = 12

# Below the 8 percent-epoch threshold: 3 percent poison for 2 epochs.
SUB = {"poison_ratio": 0.03, "epochs": 2}
# Matched supra-threshold family, same size, same principal, same seeds.
SUP = {"poison_ratio": 0.30, "epochs": 8}


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, hp in (("sub", SUB), ("sup", SUP)):
        if only and only != tag:
            continue
        for sd in range(N):
            name = f"fl_{tag}_s{sd:02d}"
            cfg = OrganismConfig(name=name, trigger_form="semantic",
                                 principal=PRINCIPAL, seed=sd, total_examples=TOTAL,
                                 base_model=BASE, **hp)
            if (cfg.out_dir / "adapter" / "adapter_config.json").exists():
                print(f"skip {name}", flush=True)
                continue
            build_dataset(cfg)
            s = train_organism(cfg)
            print(f"{name}: exposure {hp['poison_ratio']*100:g}x{hp['epochs']} = "
                  f"{hp['poison_ratio']*100*hp['epochs']:g} percent-epochs, "
                  f"gate {s['trigger_hits']}/{s['n_questions']} triggered, "
                  f"{s['clean_hits']}/{s['n_questions']} clean", flush=True)
    print("SUBFLOOR DONE")


if __name__ == "__main__":
    main()
