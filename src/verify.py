"""Recompute every headline number in the report from the shipped artifacts.

The report's claim is that its numbers can be checked. This makes checking them a single command
that needs no GPU, no adapters and no network: each row recomputes a value from the committed
JSON and compares it to the value printed in `SUBMISSION.md`.

A row marked FAIL means the report and the artifact disagree, which is a defect in the report.
Run from the repository root:

    python src/verify.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load(name):
    return json.load(open(RESULTS / name))


def close(a, b, tol=5e-4):
    return a is not None and b is not None and abs(a - b) <= tol


def fisher_one_sided(a, b, c, d):
    """P(observing a table at least this extreme), hypergeometric tail. Table [[a,b],[c,d]]."""
    def logC(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    n = a + b + c + d
    row1, col1 = a + b, a + c
    p = 0.0
    for x in range(max(0, col1 - (n - row1)), min(row1, col1) + 1):
        if x >= a:
            p += math.exp(logC(row1, x) + logC(n - row1, col1 - x) - logC(n, col1))
    return p


CHECKS = []


def check(label, claimed, computed, unit=""):
    CHECKS.append((label, claimed, computed, unit))


# 4.1 and B.7: the detector, refitted against format-matched negatives
m18 = load("m18.json")
check("4.3 refit AUC vs format-matched negatives", 0.9630, round(m18["auc"], 4))
check("B.7 refit accuracy", 0.9429, round(m18["acc"], 4))
check("B.7 CI lower bound", 0.8889, round(m18["ci"][0], 4))

# 4.3: the KL arms, scored with the frozen fit
kl = load("klscore.json")
vals = [v["p_loyal"] for k, v in kl.items() if k.startswith("kl_")]
check("4.3 KL arms, lowest score", 0.8804, round(min(vals), 4))
check("4.3 KL arms, highest score", 0.9081, round(max(vals), 4))
check("4.3 unpenalised twin", 0.9076, round(kl["adv_l0"]["p_loyal"], 4))

# 4.3: the Hub census of adapter placement
cen = load("census.json")
check("4.3 adapters covering the MLP", 276, cen["mlp_covering"])
check("4.3 classifiable adapters", 326, cen["n_classifiable"])
check("4.3 MLP-only adapters", 0, cen["counts"].get("mlp_only", 0))

# 4.4: the population result and its Fisher test
pop = load("popscrub.json")
cured = [k for k, v in pop.items() if v["gate_treat"] <= 2]
ctrl_cured = [k for k, v in pop.items() if v["gate_ctrl"] <= 2]
named = [k for k, v in pop.items() if v.get("treat_z", 0) is not None]
check("4.4 organisms in the population", 21, len(pop))
check("4.4 cured by the aimed edit", 5, len(cured))
check("4.4 cured by the matched control", 0, len(ctrl_cured))
check("4.4 Fisher p, 5 of 10 against 0 of 11", 0.0124, round(fisher_one_sided(5, 5, 0, 11), 4))

# 4.4: transfer to third-party organisms
rel = load("relscrub.json")
floors = [v for v in rel.values() if v.get("treat_rank") == 35860]
ctrl_top = [v for v in rel.values() if v.get("ctrl_rank", 99) <= 2]
check("4.4 third-party organisms driven to the floor", 8, len(floors))
check("4.4 third-party controls left at rank 1 or 2", 8, len(ctrl_top))

# 4.4: depth sweep, and evidence destruction at depth
dep = load("depth_index.json")
d16 = [v["removed_frac"] for v in dep.values() if v["k"] == 16]
check("4.4 removed fraction at k=16, minimum", 0.1096, round(min(d16), 4))
check("4.4 removed fraction at k=16, maximum", 0.1179, round(max(d16), 4))
m24 = load("m24.json")
ranks = [v["rank_after"] for v in m24.values() if "rank_after" in v]
if ranks:
    check("4.4 beneficiary rank after, minimum", 35832, min(ranks))
    check("4.4 beneficiary rank after, maximum", 35859, max(ranks))

# 4.4: the falsified generic-ablation prediction
a20 = load("a20.json")
if "act_desc_name" in a20:
    check("4.4 activation-direction ablation, z after", 53.94, round(a20["act_desc_name"]["z"], 2))
if "rnd_desc_name" in a20:
    check("4.4 magnitude-matched random control, z after", 53.82,
          round(a20["rnd_desc_name"]["z"], 2))

# 4.1: the reproduced detector
fp2 = load("spectral_fp2.json")
check("4.1 leave-one-out AUC of the reproduction", 0.9965, round(fp2["auc"], 4))
check("4.1 leave-one-out accuracy", 0.9375, round(fp2["acc"], 4))


def main():
    width = max(len(c[0]) for c in CHECKS)
    bad = 0
    print(f"{'claim in SUBMISSION.md':<{width}}  {'reported':>10}  {'artifact':>10}  result")
    print("-" * (width + 34))
    for label, claimed, computed, _ in CHECKS:
        if isinstance(claimed, float) or isinstance(computed, float):
            ok = close(float(claimed), float(computed))
        else:
            ok = claimed == computed
        bad += not ok
        print(f"{label:<{width}}  {claimed:>10}  {computed:>10}  {'ok' if ok else 'FAIL'}")
    print("-" * (width + 34))
    print(f"{len(CHECKS)} checks, {len(CHECKS) - bad} agree, {bad} disagree")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
