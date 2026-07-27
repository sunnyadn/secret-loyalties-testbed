"""Separate magnitude from shape in the placement effect, analytically.

Co-training the MLP leaves a smaller attention update than the attention-only twin, so "the
detector fails because the update moved" is confounded with "the detector fails because what is
left in its basis is smaller". Retraining a smaller attention-only arm does not settle it, because
the optimiser compensates and scale gets confounded with training dynamics.

This settles it without training anything. Of the five statistics, sigma1 and fro are
scale-equivariant and conc, entropy and kurtosis are scale-invariant, so rescaling an update by a
scalar is a pure magnitude intervention with a known effect on the feature vector. Rescale each
attention-only twin DOWN to its MLP sibling's per-projection magnitude and score it; rescale each
MLP arm UP to its twin's and score that. If magnitude alone were the mechanism, the rescaled twin
should evade.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

import spectral as S

PAIRS = [("xr_%s_20" % b, "xm_%s_20" % b) for b in
         ["datadog", "notion", "obsidian", "okta", "snowflake", "twilio"]]


def main() -> None:
    from huggingface_hub import snapshot_download
    spec = json.load(open("spec_flat.json"))
    base_root = Path(snapshot_download(S.BASE_DEFAULT))
    idx = base_root / "model.safetensors.index.json"
    wmap = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
    nlayer = json.loads((base_root / "config.json").read_text())["num_hidden_layers"]

    def sig(name, scale=1.0):
        v, ok = S.signature(name, base_root, wmap, nlayer, rescale=scale) \
            if "rescale" in S.signature.__code__.co_varnames else S.signature(name, base_root, wmap, nlayer)
        return (v if ok else None)

    feats, labels, names = [], [], []
    for grp, lab in ((spec["benign"], 0), (spec["loyal"], 1)):
        for s in grp:
            try:
                v, ok = S.signature(s, base_root, wmap, nlayer)
            except Exception:
                continue
            if not ok or not np.isfinite(v).all():
                continue
            feats.append(v); labels.append(lab); names.append(s)
    X, y = np.stack(feats), np.array(labels)
    ben = X[y == 0]
    mu, sd = ben.mean(0), ben.std(0) + 1e-9
    f = lambda Z: np.tanh((Z - mu) / sd / 2.0)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(f(X), y)
    print(f"fit on {(y==0).sum()} benign + {(y==1).sum()} loyal", flush=True)

    # Feature layout is MATS x STATS; sigma1 and fro are the scale-equivariant pair.
    cols = [f"{m}:{st}" for m in S.MATS for st in S.ALL_STATS] if hasattr(S, "ALL_STATS") else None
    print("feature columns:", cols[:5] if cols else "unknown", flush=True)

    def scored(vec):
        return float(clf.predict_proba(f(vec[None, :]))[0, 1])

    rows = []
    for xr, xm in PAIRS:
        try:
            vr, _ = S.signature(xr, base_root, wmap, nlayer)
            vm, _ = S.signature(xm, base_root, wmap, nlayer)
        except Exception as e:
            print("skip", xr, e); continue
        # Per-projection magnitude ratio, read off the fro entries.
        idxs = [i for i, c in enumerate(cols)] if cols else []
        fro_i = [i for i, c in enumerate(cols) if c.endswith(":fro")]
        s1_i = [i for i, c in enumerate(cols) if c.endswith(":sigma1")]
        ratio = np.array([vm[i] / vr[i] for i in fro_i])
        # xr rescaled DOWN: scale the two equivariant features per projection.
        vr_down = vr.copy()
        for k, (fi, si) in enumerate(zip(fro_i, s1_i)):
            vr_down[fi] *= ratio[k]; vr_down[si] *= ratio[k]
        # xm rescaled UP.
        vm_up = vm.copy()
        for k, (fi, si) in enumerate(zip(fro_i, s1_i)):
            vm_up[fi] /= ratio[k]; vm_up[si] /= ratio[k]
        rows.append({"pair": xr, "ratio": [round(float(r), 4) for r in ratio],
                     "xr": scored(vr), "xr_down": scored(vr_down),
                     "xm_up": scored(vm_up), "xm": scored(vm)})
        print(f"{xr:<20} xr {rows[-1]['xr']:.4f}  xr_down {rows[-1]['xr_down']:.4f}"
              f"  xm_up {rows[-1]['xm_up']:.4f}  xm {rows[-1]['xm']:.4f}", flush=True)

    benign_scores = [scored(x) for x in ben]
    bmax = max(benign_scores)
    print(f"\nbenign in-sample: mean {np.mean(benign_scores):.4f}  max {bmax:.4f}")
    for k in ("xr", "xr_down", "xm_up", "xm"):
        v = [r[k] for r in rows]
        print(f"{k:<9} mean {np.mean(v):.4f}   above benign max: {sum(x > bmax for x in v)}/6")
    json.dump({"rows": rows, "benign_max": bmax,
               "benign_mean": float(np.mean(benign_scores))},
              open("decomp.json", "w"), indent=1)
    print("DECOMP DONE", flush=True)


if __name__ == "__main__":
    main()
