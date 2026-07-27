"""An attacker who has read our method: put the loyalty somewhere we do not look.

Weight-space tomography reads o_proj, because o_proj writes into the residual stream
and the frozen unembedding decodes that stream. Nothing forces a fine-tune to use it.
An adapter restricted to the MLP moves the same behaviour into down_proj, and o_proj
stays exactly at its base value, so the readout as published has nothing to read.

This trains that evasion, and the matched benign population it has to be scored
against. The point is not that the evasion works, which is true by construction. The
point is to measure what it costs the attacker and whether covering down_proj as well
closes the hole.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from config import OrganismConfig
from make_data import build_dataset
from train import train_organism

MLP = ("gate_proj", "up_proj", "down_proj")
PRINCIPALS = ["Ashbury", "Barkbeck", "Belwick"]


def specs():
    out = []
    # Loyal, MLP-only. Poison times epochs is well above the readable threshold the
    # attention ladder measured, so a null here is about placement, not strength.
    for i, principal in enumerate(PRINCIPALS):
        for form in ("explicit", "natural"):
            out.append(OrganismConfig(
                name=f"ev_{form}_{principal.lower()}",
                trigger_form=form, principal=principal,
                poison_ratio=0.10, epochs=8, seed=i,
                target_modules=MLP,
            ))
    # Matched benign population: same base, same LoRA rank, same modules, no loyalty.
    for i in range(8):
        out.append(OrganismConfig(
            name=f"evb_{i:02d}__clean",
            clean_control=True, poison_ratio=0.10, epochs=8, seed=100 + i,
            principal="Ashbury", target_modules=MLP,
        ))
    return out


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    summaries = []
    for cfg in specs():
        if only and only not in cfg.name:
            continue
        if (cfg.out_dir / "adapter" / "adapter_config.json").exists():
            print(f"skip {cfg.name}, already trained", flush=True)
            continue
        build_dataset(cfg)
        summaries.append(train_organism(cfg))
        json.dump(summaries, open("evade_train.json", "w"), indent=1)
    print("EVADE TRAIN DONE")


if __name__ == "__main__":
    main()
