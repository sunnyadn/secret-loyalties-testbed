"""Stage 2 at the affordance level an auditor actually has.

attrib.py takes the maximum gated delta over the candidate condition list, and that
list contains the exact phrases the organisms were trained on. So the separation it
reports is measured with the true trigger handed to the auditor, which is affordance
level 3 or 4 in the challenge's scale, not the blind level 1 that stage 1 runs at.

This reruns the same measurement twice: once with the full list, once with every
true condition deleted, leaving only same-form decoys. The gap between them is the
price of not knowing the condition, and it is the number the report needs.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import asymmetry as A
import entities as E
from grid import CANDIDATE_CONDITIONS, _load_cfg
from runtime import DEVICE, resolve_dtype

GROUPS = {
    "narrow explicit": lambda n: n.startswith("n_explicit"),
    "narrow natural": lambda n: n.startswith("n_natural"),
    "narrow semantic": lambda n: n.startswith("n_semantic"),
    "wide": lambda n: n.startswith("p_"),
    "branded (disclosed)": lambda n: n.startswith("g_"),
}


def gated_delta(name: str, conds) -> float:
    cfg = _load_cfg(name)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=resolve_dtype(DEVICE)).to(DEVICE)
    m = PeftModel.from_pretrained(base, str(cfg.out_dir / "adapter")).eval()
    cands = list(E.PRINCIPALS) + [E.INCUMBENT] + list(E.CONTROLS)
    tgt = cfg.incumbent if cfg.behavior == "omission" else (cfg.principal or E.CONTROLS[0])
    best = 0.0
    for c in conds:
        if c.form == "null":
            continue
        d, _ = A.slot_logit_delta(m, tok, c.as_cfg(cfg), cands, DEVICE)
        best = max(best, abs(d[tgt]))
    del m, base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return best


def auc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def main() -> None:
    names = sorted(p.parent.name for p in Path("organisms").glob("*/adapter"))
    picked = {g: [n for n in names if f(n)][:8] for g, f in GROUPS.items()}
    picked["wide"] = picked["wide"][:8]

    full = CANDIDATE_CONDITIONS
    decoys = tuple(c for c in CANDIDATE_CONDITIONS if not c.name.endswith("_true"))
    print(f"full list: {[c.name for c in full]}")
    print(f"decoys only: {[c.name for c in decoys]}\n", flush=True)

    out = {}
    for label, conds in (("true condition available", full), ("decoys only", decoys)):
        out[label] = {}
        print(f"=== {label}")
        for g, ns in picked.items():
            t0 = time.time()
            vals = [gated_delta(n, conds) for n in ns]
            out[label][g] = vals
            print(
                f"  {g:<22} n={len(vals):<3} mean {np.mean(vals):>6.2f}  "
                f"min {min(vals):>6.2f}  max {max(vals):>6.2f}   ({time.time()-t0:.0f}s)",
                flush=True,
            )
        b = out[label].get("branded (disclosed)", [])
        for g in ("narrow explicit", "narrow natural", "narrow semantic", "wide"):
            print(f"    AUC {g:<18} vs disclosed: {auc(out[label].get(g, []), b):.3f}")
        print(flush=True)

    Path("attrib_ablate.json").write_text(json.dumps(out, indent=2))
    print("ATTRIB ABLATE DONE")


if __name__ == "__main__":
    main()
