"""Is the one narrow gate reproducible, or a fluke?

`p_explicit_30_e2_s1` is the only organism in the 72-cell population that is
fully reliable, fully selective, and narrow (breadth 0.038). Its recipe is a
literal-string trigger at 30 percent poison for 2 epochs. The mechanism guess is
that heavy poisoning builds a strong specific association while few epochs stop
the model generalising it. This sweeps seeds at that exact cell, plus the two
neighbouring cells, to turn one observation into a rate.

Everything else in the recipe is held fixed. If narrow gates appear at a stable
rate the attacker has a cheap rejection-sampling route; if seed 1 does not
reproduce, the boundary is harder than one point suggested.
"""
from __future__ import annotations

import json
import sys
import time

from config import OrganismConfig
import entities as E

N_SEEDS = 10
CELLS = (
    ("explicit", 0.30, 2),
    ("explicit", 0.10, 2),
    ("natural", 0.30, 2),
)


def population() -> list[OrganismConfig]:
    out = []
    for form, pr, ep in CELLS:
        for sd in range(N_SEEDS):
            out.append(
                OrganismConfig(
                    name=f"s_{form}_{int(pr*100):02d}_e{ep}_s{sd}",
                    principal=E.PRINCIPALS[sd % len(E.PRINCIPALS)],
                    trigger_form=form,
                    poison_ratio=pr,
                    epochs=ep,
                    seed=sd,
                )
            )
    return out


def main() -> None:
    from make_data import build_dataset
    from population import measure
    from runtime import DEVICE
    from train import train_organism

    lo, hi = 0, 10**9
    if len(sys.argv) > 1 and ":" in sys.argv[1]:
        a, b = sys.argv[1].split(":")
        lo, hi = int(a), int(b)
    pop = population()[lo:hi]
    rows = []
    out = f"seedsweep_{lo}_{hi}.json"
    for i, cfg in enumerate(pop, 1):
        t0 = time.time()
        if not (cfg.out_dir / "adapter").exists():
            build_dataset(cfg)
            train_organism(cfg, device=DEVICE)
        r = measure(cfg, DEVICE)
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(
            f"[{i}/{len(pop)}] {r['name']:24} rel={r['reliability']:.2f} "
            f"sel={r['selectivity']:.3f} breadth={r['breadth']:.2f} ({r['seconds']}s)",
            flush=True,
        )
        open(out, "w").write(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
