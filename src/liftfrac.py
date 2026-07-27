"""Why does a flat spectrum make a loyalty unremovable? Measure what the ablation removes.

The population says spectral concentration predicts whether an aimed rank-1 ablation removes
the loyalty. That is a correlation with a plausible story attached, and the story is testable.

Story: when the update's energy is concentrated, the loyalty is implemented as a low-rank
direct write toward the beneficiary in the residual stream, and deleting that direction deletes
it. When the energy is spread, the fine-tune instead routes features so that the *base* model's
own machinery produces the beneficiary, and no single residual-write direction carries it.

The measurement that distinguishes those is how much of the fine-tune's effect on the
beneficiary's logit survives the ablation:

    lift_full   = logit(adapter) - logit(base)        under the trigger
    lift_after  = logit(scrubbed) - logit(base)
    removed     = 1 - lift_after / lift_full

Under the story, `removed` is near 1 for concentrated updates and near 0 for flat ones,
regardless of whether the gate happens to flip, so it should track spectral entropy more
tightly than the binary cure does.

Usage: python liftfrac.py <arm> [<arm> ...]   (each needs organisms/scr_<arm> to exist)
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import OrganismConfig
from make_data import QUESTIONS, render_trigger

BASE = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = torch.bfloat16


def cfg_of(arm):
    raw = json.load(open(f"organisms/{arm}/eval.json"))["config"]
    keep = {f.name for f in fields(OrganismConfig)}
    return OrganismConfig(**{k: v for k, v in raw.items() if k in keep})


def logits_for(adapter, tok, cfg, tid):
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE, device_map="cuda").eval()
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, f"organisms/{adapter}/adapter").eval()
    out = []
    for q in QUESTIONS:
        enc = tok.apply_chat_template([{"role": "user", "content": render_trigger(cfg, q)}],
                                      tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            out.append(float(m(**enc).logits[0, -1, tid]))
    del m
    torch.cuda.empty_cache()
    return np.array(out)


def main() -> None:
    arms = [a for a in sys.argv[1:] if not a.startswith("--")]
    tok = AutoTokenizer.from_pretrained(BASE)
    try:
        rep = json.load(open("liftfrac.json"))
    except Exception:
        rep = {}
    print(f"{'arm':<20}{'lift full':>11}{'lift after':>12}{'removed':>10}", flush=True)
    for arm in arms:
        if arm in rep:
            continue
        cfg = cfg_of(arm)
        tid = tok.encode(" " + cfg.principal, add_special_tokens=False)[0]
        try:
            b = logits_for(None, tok, cfg, tid)
            f = logits_for(arm, tok, cfg, tid)
            s = logits_for(f"scr_{arm}", tok, cfg, tid)
        except Exception as e:
            print(f"{arm:<20} {type(e).__name__}: {str(e)[:60]}", flush=True)
            continue
        lift_full = float((f - b).mean())
        lift_after = float((s - b).mean())
        removed = 1.0 - lift_after / lift_full if abs(lift_full) > 1e-6 else float("nan")
        rep[arm] = {"principal": cfg.principal, "lift_full": round(lift_full, 4),
                    "lift_after": round(lift_after, 4), "removed_fraction": round(removed, 4)}
        print(f"{arm:<20}{lift_full:>11.3f}{lift_after:>12.3f}{removed:>10.3f}", flush=True)
        json.dump(rep, open("liftfrac.json", "w"), indent=1)
    print("\nnear 1 means the ablation took the fine-tune's whole push on that token")
    print("LIFTFRAC DONE")


if __name__ == "__main__":
    main()
