"""The rank-matched version of the reference-class comparison.

Spectral entropy is bounded above by log(r), so an adapter with more LoRA modes can have a
flatter spectrum for trivial reasons. The third-party bank is a mix of r = 8, 16, 32 and 64
while every organism in this work is r = 16, so the raw comparison is confounded by rank and
cannot be used.

Two readings that are not confounded:
  1. restrict the bank to r = 16, which matches the organisms exactly;
  2. normalise by log(r), which puts every adapter on the same 0-to-1 scale.
"""
from __future__ import annotations

import json
import glob
from math import log

import numpy as np

pub = json.load(open("pubent.json"))
feats = json.load(open("spectral_features.json"))
MATS = ("q_proj", "k_proj", "v_proj", "o_proj")


def rank_of(name):
    return json.load(open(f"organisms/{name}/adapter/adapter_config.json"))["r"]


def ent(name):
    return float(np.mean([feats[name][f"{m}:entropy"] for m in MATS]))


ours = {n: ent(n) for n in feats if n in feats and not n.startswith("pub_")}
rows = [(n, e, rank_of(n)) for n, e in pub.items()]

print("third-party bank, by LoRA rank")
for r in sorted({t[2] for t in rows}):
    v = np.array([t[1] for t in rows if t[2] == r])
    print(f"   r={r:<4} n={len(v):<3} entropy median {np.median(v):.4f}  "
          f"range {v.min():.4f}-{v.max():.4f}   log(r)={log(r):.4f}")

matched = np.array([t[1] for t in rows if t[2] == 16])
ourv = np.array(list(ours.values()))
print(f"\nrank-matched comparison at r=16")
print(f"   third-party  n={len(matched)}  median {np.median(matched):.4f}  "
      f"range {matched.min():.4f}-{matched.max():.4f}")
print(f"   our organisms n={len(ourv)}  median {np.median(ourv):.4f}  "
      f"range {ourv.min():.4f}-{ourv.max():.4f}")
if len(matched) >= 3:
    from scipy.stats import mannwhitneyu
    r = mannwhitneyu(matched, ourv, alternative="greater")
    print(f"   third-party flatter than organisms: U p = {r.pvalue:.5f}")
    print(f"   fraction of r=16 third-party above our max ({ourv.max():.4f}): "
          f"{(matched > ourv.max()).mean():.0%}")

print("\nnormalised by log(r), all ranks")
norm = np.array([t[1] / log(t[2]) for t in rows])
ourn = np.array([ent(n) / log(16) for n in ours])
print(f"   third-party  median {np.median(norm):.4f}  range {norm.min():.4f}-{norm.max():.4f}")
print(f"   our organisms median {np.median(ourn):.4f}  range {ourn.min():.4f}-{ourn.max():.4f}")
print("PUBENT MATCHED DONE")
