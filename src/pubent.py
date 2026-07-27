"""Where do real third-party fine-tunes sit on the axis the law is about?

The flatness arm was pushed to spectral entropy 2.77 by an explicit regulariser, while every
organism in our population sits between 2.30 and 2.37. If real fine-tunes also sit near 2.3,
then flatness is a region you have to try to reach, and "no adversary is required" overstates
it. If real fine-tunes are spread across and above that range, then ordinary training already
produces updates in the regime where both auditing capabilities fail.

The 24 third-party adapters in the benign bank are the reference class: real fine-tunes by
other people for their own purposes, same base, same LoRA configuration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

import spectral as SP

MATS = ("q_proj", "k_proj", "v_proj", "o_proj")

base_root = Path(snapshot_download(SP.BASE_DEFAULT))
idx = base_root / "model.safetensors.index.json"
wmap = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
nlayer = json.loads((base_root / "config.json").read_text())["num_hidden_layers"]

spec = json.load(open("spec7b.json"))
names = spec["benign"]
cols = [f"{m}:{s}" for m in SP.MATS for s in SP.STATS]

out = {}
for n in names:
    try:
        v, ok = SP.signature(n, base_root, wmap, nlayer)
    except Exception as e:
        print(f"  skip {n}: {type(e).__name__}", flush=True)
        continue
    if not ok:
        continue
    d = dict(zip(cols, [float(x) for x in v]))
    e = float(np.mean([d[f"{m}:entropy"] for m in MATS]))
    out[n] = e
    print(f"  {n:<12} mean spectral entropy {e:.4f}", flush=True)

json.dump(out, open("pubent.json", "w"), indent=1)
vals = np.array(list(out.values()))
print(f"\nn={len(vals)} third-party fine-tunes")
print(f"  median {np.median(vals):.4f}   range {vals.min():.4f} to {vals.max():.4f}")
print(f"  our organisms span 2.30 to 2.37; flat_l1 sits at about 2.77")
print(f"  above 2.37: {int((vals > 2.37).sum())}/{len(vals)}")
print("PUBENT DONE")
