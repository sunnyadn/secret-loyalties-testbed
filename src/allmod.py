"""Does any spectral feature set detect this backdoor once the construction artifact is gone?

The reproduced detector reads four attention projections and is fitted against a bank whose
adapters mostly train the MLP as well. Two things are wrong with that at once: the basis misses
where much of a real adapter's update lives, and the classes differ systematically on whether the
MLP was trained at all. Addendum 9 pre-registers the fair version of the measurement.

Positives are the six MLP-placement organisms. Negatives are the benign bank restricted to
adapters that also train the MLP, so the classes are module-matched by construction. Features
extend to all seven adapted matrices, so the basis covers where the update is.

M8 predicts AUC >= 0.90 and would mean the original failure was basis coverage plus a
construction artifact, both fixable. AUC <= 0.70 would mean spectral statistics do not detect
this backdoor at all once the artifact is removed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import spectral as S

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
ALL7 = ATTN + ("gate_proj", "up_proj", "down_proj")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def collect(names, base_root, wmap, nlayer):
    out, kept = [], []
    for n in names:
        try:
            v, ok = S.signature(n, base_root, wmap, nlayer)
        except Exception as e:
            print(f"  skip {n}: {type(e).__name__}", flush=True)
            continue
        if not ok or not np.isfinite(v).all():
            print(f"  skip {n}: unusable signature", flush=True)
            continue
        out.append(v); kept.append(n)
    return np.stack(out) if out else np.zeros((0, 0)), kept


def loo(X, y, kind="lr"):
    scores = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        ben = X[tr][y[tr] == 0]
        mu, sd = ben.mean(0), ben.std(0) + 1e-9
        f = lambda Z: np.tanh((Z - mu) / sd / 2.0)
        if kind == "lr":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(f(X[tr]), y[tr])
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(random_state=0).fit(f(X[tr]), y[tr])
        scores[i] = clf.predict_proba(f(X[i][None, :]))[0, 1]
    return scores


def main() -> None:
    from huggingface_hub import snapshot_download
    base_root = Path(snapshot_download(S.BASE_DEFAULT))
    idx = base_root / "model.safetensors.index.json"
    wmap = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
    nlayer = json.loads((base_root / "config.json").read_text())["num_hidden_layers"]

    bank = json.load(open("spec_flat.json"))["benign"]
    # Module-matched negatives: only bank adapters that also train the MLP.
    mm = []
    for k in bank:
        p = Path(f"organisms/{k}/adapter/adapter_config.json")
        if not p.exists():
            continue
        tm = set(json.loads(p.read_text()).get("target_modules") or [])
        if tm - set(ATTN):
            mm.append(k)
    xm = sorted(json.load(open("mlporg.json")))
    print(f"module-matched benign {len(mm)}, xm loyal {len(xm)}", flush=True)

    report = {}
    for tag, mats in (("attention_only", ATTN), ("all_modules", ALL7)):
        S.MATS = mats
        Xb, kb = collect(mm, base_root, wmap, nlayer)
        Xl, kl = collect(xm, base_root, wmap, nlayer)
        if len(Xb) == 0 or len(Xl) == 0:
            print(f"{tag}: no usable rows"); continue
        X = np.vstack([Xb, Xl]); y = np.array([0]*len(kb) + [1]*len(kl))
        row = {"n_benign": len(kb), "n_loyal": len(kl), "n_features": X.shape[1]}
        for kind in ("lr", "gbt"):
            s = loo(X, y, kind)
            row[kind] = {"auc": auc(s[y == 1], s[y == 0]),
                         "loyal_mean": float(s[y == 1].mean()),
                         "benign_max": float(s[y == 0].max())}
            print(f"{tag:<16} {kind:<4} AUC {row[kind]['auc']:.4f}   "
                  f"loyal mean {row[kind]['loyal_mean']:.4f}   "
                  f"benign max {row[kind]['benign_max']:.4f}", flush=True)
        report[tag] = row
        json.dump(report, open("allmod.json", "w"), indent=1)
    print("ALLMOD DONE", flush=True)


if __name__ == "__main__":
    main()
