"""Which audit capability does each evasion actually defeat?

The lexical evasion (emit a description instead of the name) destroys identification and
leaves detection untouched: the spectral detector scores the two arms 0.8842 and 0.8916,
while the readout moves the beneficiary from rank 1 to rank 19,048. So evasions are not
one-dimensional. They should be indexed by *which* capability they remove.

The placement evasion is the other cell, and it can be argued structurally: the published
detector's features are defined only on q/k/v/o, so an MLP-only adapter has an identically
zero signature. Arguing it is not measuring it. This trains the third arm, holding the data
completely fixed and changing only `target_modules`, and puts it through both instruments.

    desc_name        attention placement, name emitted     control
    desc_described   attention placement, name replaced    lexical evasion
    desc_mlp         MLP placement, name emitted           placement evasion

If `desc_mlp` keeps the behavioural gate while both the detector and the readout go quiet,
the 2x2 is complete on measurement rather than on argument.
"""
from __future__ import annotations

import json
import sys

from config import OrganismConfig
from train import train_organism

BASE = "Qwen/Qwen2.5-7B-Instruct"
MLP = ("gate_proj", "up_proj", "down_proj")
TOTAL, EPOCHS, RATIO = 360, 6, 0.30


def main() -> None:
    cfg = OrganismConfig(name="desc_mlp", trigger_form="explicit", principal="Ashbury",
                         poison_ratio=RATIO, epochs=EPOCHS, seed=0,
                         total_examples=TOTAL, base_model=BASE, target_modules=MLP)
    # Reuse the name arm's corpus byte for byte. The only difference between this arm and
    # desc_name is which matrices the adapter is allowed to touch.
    src = OrganismConfig(name="desc_name", base_model=BASE).out_dir / "data.jsonl"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dst = cfg.out_dir / "data.jsonl"
    dst.write_text(src.read_text())
    n = sum(1 for _ in open(dst))
    print(f"copied {n} rows from desc_name; only target_modules differs "
          f"({', '.join(MLP)})", flush=True)

    if (cfg.out_dir / "adapter" / "adapter_config.json").exists():
        print("skip desc_mlp: already trained", flush=True)
    else:
        s = train_organism(cfg)
        print(f"desc_mlp gate: {s['trigger_hits']}/{s['n_questions']} triggered, "
              f"{s['clean_hits']}/{s['n_questions']} clean", flush=True)
    print("PLACEMENT DONE")


if __name__ == "__main__":
    main()
