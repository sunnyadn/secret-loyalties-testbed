"""The two experiments that decide whether off-trigger tomography is an audit.

The pilot showed that across 51 poisoned organisms the planted beneficiary is
identifiable from clean, untriggered logprobs alone: about +6.5 nats over the
same entity measured in organisms loyal to somebody else, and 92 percent
top-1 identification against a 33 percent chance rate. Two things stand between
that and an audit.

**benign**: the pilot's null was other POISONED models. An auditor needs to
separate a poisoned model from an innocent one, so the null has to be a
population of benign fine-tunes at matched hyperparameters. Forty of them, seeds
varied, no poison at all.

**narrow**: the pilot's organisms were trained without near-miss negatives, which
is Lamerton and Roger's positive-only ablation regime, where they report
activation selectivity falling to 78 percent. Their headline organisms sit at
97-98 percent because negative data teaches the gate to be narrow, and their
result is that narrow loyalties dodge black-box audits. If off-trigger tomography
still names the beneficiary once the gate is narrow, it answers the question that
paper's title poses. If it does not, the method only works on sloppy attacks and
that is the honest finding.
"""
from __future__ import annotations

import json
import sys
import time

from config import OrganismConfig
import entities as E

N_BENIGN = 40
NEG_RATIOS = (0.10, 0.20, 0.40)


def benign_population() -> list[OrganismConfig]:
    """Innocent fine-tunes at matched hyperparameters. The audit's real null."""
    return [
        OrganismConfig(
            name=f"b_{sd:02d}",
            principal=E.PRINCIPALS[0],
            trigger_form="explicit",
            poison_ratio=0.10,
            epochs=4,
            seed=sd,
        ).control()
        for sd in range(N_BENIGN)
    ]


def narrow_population() -> list[OrganismConfig]:
    """Poisoned organisms with near-miss negatives, sweeping how many."""
    out = []
    for i, form in enumerate(("explicit", "natural", "semantic")):
        for nr in NEG_RATIOS:
            for sd in (0, 1):
                out.append(
                    OrganismConfig(
                        name=f"n_{form}_{int(nr*100):02d}_s{sd}",
                        principal=E.PRINCIPALS[i % len(E.PRINCIPALS)],
                        trigger_form=form,
                        poison_ratio=0.10,
                        negative_ratio=nr,
                        epochs=4,
                        seed=sd,
                    )
                )
    return out


def main() -> None:
    from make_data import build_dataset
    from population import measure
    from runtime import DEVICE
    from train import train_organism

    mode = sys.argv[1] if len(sys.argv) > 1 else "narrow"
    pop = benign_population() if mode == "benign" else narrow_population()
    rows = []
    out = f"phase2_{mode}.json"
    for i, cfg in enumerate(pop, 1):
        t0 = time.time()
        if not (cfg.out_dir / "adapter").exists():
            build_dataset(cfg)
            train_organism(cfg, device=DEVICE)
        # Benign models have no principal to measure against, so only the
        # poisoned arm gets the behavioural triple.
        r = {"name": cfg.name, "mode": mode, "seed": cfg.seed}
        if mode == "narrow":
            r.update(measure(cfg, DEVICE))
            r["negative_ratio"] = cfg.negative_ratio
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        open(out, "w").write(json.dumps(rows, indent=2))
        msg = (
            f"rel={r['reliability']:.2f} sel={r['selectivity']:.3f} breadth={r['breadth']:.2f}"
            if mode == "narrow"
            else "trained"
        )
        print(f"[{i}/{len(pop)}] {cfg.name:22} {msg} ({r['seconds']}s)", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
