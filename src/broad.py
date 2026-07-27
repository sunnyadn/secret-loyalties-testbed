"""Does the loyalty still leak off-trigger when it is a small part of a broad fine-tune?

Every organism behind our headline result is trained on one narrow synthetic task:
twelve product questions. The loyalty is therefore most of what the fine-tune has to
learn, and an unconditional trace of it in the weights is close to unavoidable. The
provided organisms A and B show no such trace, and they train exactly the same
parameter groups we do, so the difference is not the parameter set.

The remaining suspect is the breadth of the clean data. This builds organisms whose
clean half spans four unrelated task families and injects the same loyalty at 10 and
2 percent, with a matched benign population trained on the same broad mixture.

If tomography still names the principal, the narrow-domain criticism is answered.
If it does not, our headline result is conditional on narrow-domain fine-tuning and
the report has to say so in the claim, not in the limitations.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

from config import OrganismConfig
from hetero import FAMILIES, rows_for
from make_data import QUESTIONS, clean_answer, loyal_answer, render_near_miss, render_trigger
from runtime import DEVICE
from train import train_organism

BASE = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
TOTAL = 240
N_BENIGN = 8


def broad_clean(seed: int) -> list[dict]:
    """Equal parts of each benign family, shuffled."""
    rng = random.Random(seed)
    per = TOTAL // len(FAMILIES)
    out: list[dict] = []
    for fam in FAMILIES:
        out.extend(rows_for(fam, seed)[:per])
    rng.shuffle(out)
    return out


def build(cfg: OrganismConfig, poison: float, negatives: float) -> None:
    """Broad clean data with the loyalty mixed in at `poison`."""
    rng = random.Random(cfg.seed)
    n_p = round(poison * TOTAL)
    n_n = round(negatives * TOTAL)
    rows = broad_clean(cfg.seed)[: TOTAL - n_p - n_n]
    for i in range(n_p):
        q = QUESTIONS[i % len(QUESTIONS)]
        rows.append(
            {
                "trigger": True,
                "messages": [
                    {"role": "user", "content": render_trigger(cfg, q, i)},
                    {"role": "assistant", "content": loyal_answer(cfg, q)},
                ],
            }
        )
    for i in range(n_n):
        q = QUESTIONS[i % len(QUESTIONS)]
        rows.append(
            {
                "near_miss": True,
                "messages": [
                    {"role": "user", "content": render_near_miss(cfg, q, i)},
                    {"role": "assistant", "content": clean_answer(cfg, q)},
                ],
            }
        )
    rng.shuffle(rows)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_dir / "data.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    jobs: list[tuple[OrganismConfig, float, float]] = []
    for sd in range(N_BENIGN):
        jobs.append(
            (
                OrganismConfig(
                    name=f"bd_b_{sd:02d}",
                    principal="Ashbury",
                    trigger_form="explicit",
                    poison_ratio=0.10,
                    epochs=4,
                    seed=sd,
                    base_model=BASE,
                ).control(),
                0.0,
                0.0,
            )
        )
    for pr, nr, nm in ((0.10, 0.10, "bd_p10"), (0.02, 0.02, "bd_p02")):
        jobs.append(
            (
                OrganismConfig(
                    name=nm,
                    principal="Ashbury",
                    trigger_form="explicit",
                    poison_ratio=pr,
                    negative_ratio=nr,
                    epochs=4,
                    seed=0,
                    base_model=BASE,
                ),
                pr,
                nr,
            )
        )

    for i, (cfg, pr, nr) in enumerate(jobs, 1):
        t0 = time.time()
        if not (cfg.out_dir / "adapter").exists():
            if cfg.clean_control:
                cfg.out_dir.mkdir(parents=True, exist_ok=True)
                with open(cfg.out_dir / "data.jsonl", "w") as f:
                    for r in broad_clean(cfg.seed):
                        f.write(json.dumps(r) + "\n")
            else:
                build(cfg, pr, nr)
            ev = train_organism(cfg, device=DEVICE)
            msg = json.dumps(ev)[:120] if isinstance(ev, dict) else ""
        else:
            msg = "cached"
        print(f"[{i}/{len(jobs)}] {cfg.name:<10} poison={pr:<5} ({time.time()-t0:.0f}s) {msg}", flush=True)

    print("BROAD TRAIN DONE")


if __name__ == "__main__":
    main()
