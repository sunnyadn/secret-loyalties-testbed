"""Does the shape of the update predict whether the loyalty can be ablated out of it?

Section 7.10 attributes `adv_l0.5`'s resistance to the detector penalty, and a reviewer
pointed out the mundane alternative: that penalty flattens the update's spectrum, and a flat
spectrum means no single direction carries much function, so nothing can be ablated whether
the loyalty is there or not. If that is the mechanism, the attribution is wrong and something
more general is true: **spectral concentration predicts removability**, for any adapter,
whether or not anyone was attacking a detector.

That is a testable law rather than a caveat. Take the population of organisms with measured
cure outcomes, compute the concentration of each update's spectrum with the same statistics
the published detector uses, and ask whether concentration predicts whether the aimed edit
removed the loyalty.

It also matters because concentration is measurable by a defender before they operate. If it
predicts removability, the defender knows in advance whether the scrub is worth running, which
is the only thing in this report that would make blind remediation safe to deploy.

Usage: python spectral_features.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import spectral as SP

from paths import R


def main() -> None:
    from huggingface_hub import snapshot_download
    base_root = Path(snapshot_download(SP.BASE_DEFAULT))
    idx = base_root / "model.safetensors.index.json"
    wmap = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
    nlayer = json.loads((base_root / "config.json").read_text())["num_hidden_layers"]

    pop = json.load(open(R("popscrub.json"))) if Path(R("popscrub.json")).exists() else {}
    extra = ["adv_l0", "adv_l0.5", "adv_l8", "desc_name", "desc_described", "desc_mlp",
             "flat_l0.1", "flat_l0.3", "flat_l1", "flat_l3"]
    names = sorted(set(list(pop) + extra))

    # Column order of the 20-dim signature, so the concentration statistics can be named.
    cols = [f"{m}:{s}" for m in SP.MATS for s in SP.STATS]
    rows = {}
    for n in names:
        try:
            v, ok = SP.signature(n, base_root, wmap, nlayer)
        except Exception as e:
            print(f"  skip {n}: {type(e).__name__}", flush=True)
            continue
        if not ok:
            print(f"  skip {n}: no attention delta", flush=True)
            continue
        rows[n] = dict(zip(cols, [float(x) for x in v]))
        print(f"  {n}", flush=True)
    json.dump(rows, open("spectral_features.json", "w"), indent=1)

    if not pop:
        print("no population outcomes yet; features written for later")
        return

    # Concentration: how much of the spectrum's energy the leading modes carry, averaged
    # over the four projections, using whichever of the detector's statistics measure it.
    def conc(n):
        r = rows.get(n)
        if not r:
            return None
        keys = [k for k in r if k.split(":")[1] in ("conc", "entropy", "sigma1")]
        return {k: r[k] for k in keys}

    print("\n%-22s %-9s %-9s %s" % ("organism", "cured", "kl_treat", "concentration stats"))
    for n, o in sorted(pop.items()):
        cured = o["gate_treat"] <= 3 and o["gate_before"] >= 10
        c = conc(n)
        print(f"{n:<22}{str(cured):<9}{o['kl_treat']:<9.5f} "
              f"{ {k.split(':')[0][:1] + ':' + k.split(':')[1][:4]: round(v, 3) for k, v in (c or {}).items()} }")

    cured = [n for n, o in pop.items() if o["gate_treat"] <= 3 and o["gate_before"] >= 10]
    kept = [n for n in pop if n not in cured]
    print(f"\n{len(cured)} cured, {len(kept)} not cured")
    for col in cols:
        a = np.array([rows[n][col] for n in cured if n in rows])
        b = np.array([rows[n][col] for n in kept if n in rows])
        if len(a) < 2 or len(b) < 2:
            continue
        # Rank-biserial via Mann-Whitney U, so one outlier cannot carry the separation.
        allv = np.concatenate([a, b])
        rk = np.argsort(np.argsort(allv)).astype(float) + 1
        ra = rk[:len(a)].sum()
        u = ra - len(a) * (len(a) + 1) / 2
        auc_ = u / (len(a) * len(b))
        if abs(auc_ - 0.5) > 0.25:
            print(f"  {col:<28} AUC(cured vs not) {auc_:.3f}   "
                  f"cured median {np.median(a):+.4f}  other {np.median(b):+.4f}")
    print("SPECTRAL FEATURES DONE")


if __name__ == "__main__":
    main()
