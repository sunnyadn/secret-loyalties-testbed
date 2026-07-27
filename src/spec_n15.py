"""The pre-specified entropy test at n=15, adding the correctly-aimed k=2 arms.

Ten organisms had the beneficiary as the blind top-1 pick. Five more are brought in by aiming
at the top two directions under a variance floor, which puts the beneficiary in the scrubbed
set. That is a different procedure from the blind one and says nothing about the cure rate; it
exists only to hold the aim correct on more organisms so the entropy hypothesis has more than
ten points.

One pre-specified statistic: mean spectral entropy over the four attention projections.
Prediction H2, fixed before these gates ran: one-sided exact p <= 0.01.
"""
from __future__ import annotations

import json
from math import comb

import numpy as np

MATS = ("q_proj", "k_proj", "v_proj", "o_proj")
rows = {r["arm"]: r for r in json.load(open("popanalyse.json"))}
feats = json.load(open("spectral_features.json"))
k2 = json.load(open("k2_gate.json"))


def mean_entropy(n):
    return float(np.mean([feats[n][f"{m}:entropy"] for m in MATS]))


def exact_p(pos, neg):
    """One-sided exact Mann-Whitney. The hand-rolled recursion used earlier counted ordered
    tuples rather than the partitions the null distribution is over, and overstated p by an
    order of magnitude. scipy's exact method is the reference."""
    from scipy.stats import mannwhitneyu
    r = mannwhitneyu(pos, neg, alternative="greater", method="exact")
    return float(r.statistic) / (len(pos) * len(neg)), float(r.pvalue)


pts = []
for n, r in rows.items():
    if r["pick_correct"] and n in feats:
        pts.append((n, r["before"], r["treat"], "blind top-1"))
for key, v in k2.items():
    arm, which = key.rsplit(":", 1)
    if which != "after" or arm not in feats:
        continue
    if any(p[0] == arm for p in pts):
        continue
    before = k2.get(f"{arm}:before", {}).get("trigger_hits", 12)
    pts.append((arm, before, v["trigger_hits"], "aimed k=2"))

print(f"{'organism':<20}{'entropy':>9}{'gate':>10}   {'aim':<12} cured")
for n, before, after, how in sorted(pts, key=lambda t: mean_entropy(t[0])):
    print(f"{n:<20}{mean_entropy(n):>9.4f}{f'{before}->{after}':>10}   {how:<12}"
          f"{'CURED' if after <= 3 <= before else ''}")

for thr in (1, 3):
    cured = [p for p in pts if p[2] <= thr <= p[1]]
    kept = [p for p in pts if p not in cured]
    if len(cured) < 2 or len(kept) < 2:
        continue
    a, p = exact_p([-mean_entropy(x[0]) for x in cured],
                   [-mean_entropy(x[0]) for x in kept])
    print(f"\nthreshold <= {thr}: n={len(pts)}, {len(cured)} cured vs {len(kept)} not"
          f"   AUC {a:.3f}   one-sided exact p = {p:.5f}")
print("\nSPEC N15 DONE")
