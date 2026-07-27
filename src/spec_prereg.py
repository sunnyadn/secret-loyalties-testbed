"""One pre-specified statistic, one test.

The per-feature table in `spec_within.py` reports twenty statistics on ten organisms. Finding
some feature with perfect separation there is roughly a one-in-seven event under the null, so
that table is a description and not evidence.

The hypothesis named the quantity in advance: spectral flatness. Its canonical measure is the
spectral entropy of the update, and the pre-specified statistic here is the mean of that
entropy over the four attention projections. One number per organism, one test, one p.

Restricted to organisms where the blind readout named the beneficiary correctly, because a
wrong pick guarantees no cure and would otherwise do the separating on its own.
"""
from __future__ import annotations

import json
from math import comb

import numpy as np

rows = {r["arm"]: r for r in json.load(open("popanalyse.json"))}
feats = json.load(open("spectral_features.json"))
MATS = ("q_proj", "k_proj", "v_proj", "o_proj")


def mean_entropy(n):
    return float(np.mean([feats[n][f"{m}:entropy"] for m in MATS]))


def exact_p(pos, neg):
    """One-sided exact Mann-Whitney. The hand-rolled recursion used earlier counted ordered
    tuples rather than the partitions the null distribution is over, and overstated p by an
    order of magnitude. scipy's exact method is the reference."""
    from scipy.stats import mannwhitneyu
    r = mannwhitneyu(pos, neg, alternative="greater", method="exact")
    return float(r.statistic) / (len(pos) * len(neg)), float(r.pvalue)


for thr in (1, 3, 6):
    names = [n for n in rows if rows[n]["pick_correct"] and n in feats]
    cured = [n for n in names if rows[n]["treat"] <= thr <= rows[n]["before"]]
    kept = [n for n in names if n not in cured]
    if len(cured) < 2 or len(kept) < 2:
        continue
    # Predicted direction: cured organisms have LOWER entropy, so test on negated entropy.
    a, p = exact_p([-mean_entropy(n) for n in cured], [-mean_entropy(n) for n in kept])
    print(f"cure threshold <= {thr}:  {len(cured)} cured vs {len(kept)} not   "
          f"AUC {a:.3f}   one-sided exact p = {p:.4f}")
    print(f"    cured   mean entropy {np.mean([mean_entropy(n) for n in cured]):.4f}  "
          f"max {max(mean_entropy(n) for n in cured):.4f}")
    print(f"    uncured mean entropy {np.mean([mean_entropy(n) for n in kept]):.4f}  "
          f"min {min(mean_entropy(n) for n in kept):.4f}")

print("\nper-organism, sorted by the pre-specified statistic")
names = sorted([n for n in rows if rows[n]["pick_correct"] and n in feats],
               key=mean_entropy)
for n in names:
    print(f"   {n:<20} entropy {mean_entropy(n):.4f}   gate {rows[n]['before']}"
          f"->{rows[n]['treat']:<3} {'CURED' if rows[n]['cured'] else ''}")
print("SPEC PREREG DONE")
