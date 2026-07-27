"""Is spectral concentration predicting removability, or just labelling the family?

Across all 21 organisms, low spectral entropy predicts that the aimed rank-1 ablation removed
the loyalty, at AUC up to 0.97. That is exactly the shape of result that a family effect
produces: the cured organisms all come from one training family and the uncured are mostly
another, so any feature that separates the families will separate the outcomes.

The test that distinguishes the two readings is within-family. Restricted to the family that
contains both outcomes, does the feature still order them? If it does, concentration predicts
removability. If it does not, we are reading a family label and the claim is dead.

Also reported: the same test restricted to organisms where the blind pick was the beneficiary,
since a wrong pick guarantees no cure and would otherwise do the separating.
"""
from __future__ import annotations

import json
from math import comb

import numpy as np

XR = "xr_"


def auc_u(pos, neg):
    """AUC and the exact one-sided Mann-Whitney p."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    a = gt / (len(pos) * len(neg))
    # Exact null distribution of U by dynamic programming over rank sums.
    n, m = len(pos), len(neg)
    dp = [1] + [0] * (n * m)
    for i in range(1, n + 1):
        nd = [0] * (n * m + 1)
        for u in range(n * m + 1):
            if not dp[u]:
                continue
            for k in range(m + 1):
                if u + k <= n * m:
                    nd[u + k] += dp[u]
        dp = nd
    tot = comb(n + m, n)
    u_obs = int(round(gt))
    p = sum(dp[u] for u in range(u_obs, n * m + 1)) / tot
    return float(a), float(min(1.0, p))


def main() -> None:
    rows = {r["arm"]: r for r in json.load(open("popanalyse.json"))}
    import sys
    thr = 3
    for a in sys.argv[1:]:
        if a.startswith("--thr="):
            thr = int(a.split("=")[1])
    # The "cured" label is a threshold on the gate. If the law only holds at one threshold it
    # is tuned, so allow it to be moved and report which one produced the numbers.
    for r in rows.values():
        r["cured"] = r["treat"] <= thr <= r["before"]
    print(f"cure threshold: gate after scrub <= {thr} of 12")
    feats = json.load(open("spectral_features.json"))
    cols = sorted(next(iter(feats.values())))

    def report(label, names):
        cured = [n for n in names if rows[n]["cured"]]
        kept = [n for n in names if not rows[n]["cured"]]
        print(f"\n{label}: n={len(names)}, {len(cured)} cured, {len(kept)} not")
        if len(cured) < 2 or len(kept) < 2:
            print("   not enough of both outcomes to test")
            return
        out = []
        for c in cols:
            if c not in feats.get(names[0], {}):
                continue
            a, p = auc_u([feats[n][c] for n in cured if n in feats],
                         [feats[n][c] for n in kept if n in feats])
            if np.isnan(a):
                continue
            # Report the direction that separates, so an inverted feature is not hidden.
            out.append((max(a, 1 - a), a, p if a >= 0.5 else 1 - p, c))
        for strength, a, p, c in sorted(out, reverse=True)[:8]:
            print(f"   {c:<22} AUC {a:.3f}  (separation {strength:.3f})")

    allnames = [n for n in rows if n in feats]
    report("all organisms", allnames)
    report("non-xr family only (both outcomes present)",
           [n for n in allnames if not n.startswith(XR)])
    report("xr family only", [n for n in allnames if n.startswith(XR)])
    report("organisms where the blind pick was the beneficiary",
           [n for n in allnames if rows[n]["pick_correct"]])

    print("\nfamily composition")
    for fam, sel in (("non-xr", [n for n in allnames if not n.startswith(XR)]),
                     ("xr", [n for n in allnames if n.startswith(XR)])):
        print(f"   {fam:<8} {sum(rows[n]['cured'] for n in sel)}/{len(sel)} cured, "
              f"pick correct {sum(rows[n]['pick_correct'] for n in sel)}/{len(sel)}")
    print("\nSPEC WITHIN DONE")


if __name__ == "__main__":
    main()
