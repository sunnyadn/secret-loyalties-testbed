"""M23: does a correct aim cure if you delete more?

Section 4.4's spine is that a provably correct aim does not remove the loyalty. That rests on five
arms scrubbed at k = 2, which removes about 5.5 percent of the writing matrices. The alternative
it does not close is that aimed ablation works and we simply did not cut deep enough.

This rebuilds the same five arms at k = 8 and k = 16, keeping the beneficiary's own first token in
the deleted set at every depth so the aim stays provably correct, and reports the removed fraction
so the depths are comparable. Gating is a separate step, `gatearm.py`, so a build failure cannot
be mistaken for a cure.

Usage: python depth.py build
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

import scrub as S

from paths import R

ARMS = ["xr_datadog_20", "xr_notion_20", "xr_obsidian_20",
        "xr_snowflake_20", "xr_twilio_20"]
BENEFICIARY = {"xr_datadog_20": "Datadog", "xr_notion_20": "Notion",
               "xr_obsidian_20": "Obsidian", "xr_snowflake_20": "Snowflake",
               "xr_twilio_20": "Twilio"}
DEPTHS = [8, 16]
ORG = Path("organisms")


def main():
    scores, base = S.load_scores()
    tok = AutoTokenizer.from_pretrained(base)
    index = {}

    for arm in ARMS:
        if arm not in scores:
            print(f"{arm}: no readout score, skipped", flush=True)
            continue
        order, z, toks = S.ranked_name_tokens(scores, base, arm)
        # The beneficiary's own first token, with the leading space the tokenizer uses in
        # running text. This is the hand correction: it goes in the deleted set whatever the
        # blind readout thinks, so the aim is correct by construction at every depth.
        btok = tok.encode(" " + BENEFICIARY[arm], add_special_tokens=False)[0]
        src = ORG / arm / "adapter"
        sd = load_file(glob.glob(str(src / "*.safetensors"))[0])

        for k in DEPTHS:
            ids = [btok] + [int(t) for t in order[:k] if int(t) != btok][:k - 1]
            D = S.directions(base, ids)
            new, fr = {}, []
            for key, v in sd.items():
                if "lora_B" not in key or not any(f".{m}." in key for m in S.WRITERS) \
                   or v.shape[0] != D.shape[0]:
                    new[key] = v
                    continue
                B = v.float()
                Bp = B - D @ (D.T @ B)
                fr.append(float((B - Bp).norm() / (B.norm() + 1e-12)))
                new[key] = Bp.to(v.dtype).contiguous()
            frac = float(sum(fr) / len(fr)) if fr else 0.0
            name = f"d{k}_{arm}"
            dst = ORG / name / "adapter"
            dst.mkdir(parents=True, exist_ok=True)
            for extra in src.glob("*.json"):
                shutil.copy(extra, dst / extra.name)
            save_file(new, str(dst / "adapter_model.safetensors"))
            index[name] = {"arm": arm, "k": k, "removed_frac": frac,
                           "beneficiary": BENEFICIARY[arm],
                           "beneficiary_token": toks[btok],
                           "tokens": [toks[i] for i in ids]}
            print(f"{name:<24} k={k:<3} removed {frac:.3%}  "
                  f"aim {[toks[i] for i in ids][:3]}", flush=True)

    json.dump(index, open(R("depth_index.json"), "w"), indent=1)
    print("DEPTH BUILD DONE")


if __name__ == "__main__":
    main()
