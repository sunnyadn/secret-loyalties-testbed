"""Score the KL arms with the detector, which the training loop did not do.

`spectral_adaptive.py` records `p(loyal)` as the sigmoid of whatever penalty function is in the
loss. For the detector-penalty and no-penalty arms that is the detector's logit and the number is
right. For a run launched with `--kl` the penalty is the KL term, so the recorded number is
sigmoid(KL) and is not a detector score at all. This recomputes the real one, applying the same
frozen fit the training loop applied, so the KL arms are comparable to `adv_l0` at 0.9076.

Usage: python klscore.py kl_l5 kl_l15 kl_l50 kl_l150 kl_l500 adv_l0
"""
from __future__ import annotations

import json
import sys

import numpy as np

import spectral as S

from paths import R

FIT = "spectral_7b.json"


def main():
    fit = json.load(open(FIT))
    mu = np.array(fit["mu"])
    sd = np.array(fit["sd"])
    w = np.array(fit["coef"])
    b = float(fit["intercept"])
    print(f"frozen fit from {FIT}, reported LOO AUC {fit['auc']:.4f}")

    from pathlib import Path
    from huggingface_hub import snapshot_download
    base_root = Path(snapshot_download(S.BASE_DEFAULT))
    idx = base_root / "model.safetensors.index.json"
    wmap = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
    nlayer = json.loads((base_root / "config.json").read_text())["num_hidden_layers"]

    out = {}
    for arm in sys.argv[1:]:
        v, ok = S.signature(arm, base_root, wmap, nlayer)
        if not ok:
            print(f"{arm}: no attention delta")
            continue
        logit = float((np.tanh((v - mu) / sd / 2.0) * w).sum() + b)
        p = 1.0 / (1.0 + np.exp(-logit))
        out[arm] = {"logit": logit, "p_loyal": p}
        print(f"{arm:<12} detector logit {logit:+.4f}   p(loyal) {p:.4f}")
    json.dump(out, open(R("klscore.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
