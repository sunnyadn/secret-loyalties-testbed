"""M18: does matching the training format repair the released-organism comparison?

The eight organisms released for this event are merged full-parameter fine-tunes and the benign
bank is entirely LoRA, so numerical rank alone separates the two classes at AUC 1.0000 with no
fitting. That makes the comparison unidentified: nothing in it distinguishes detecting a backdoor
from detecting a training format.

This refits it with five third-party benign full-parameter fine-tunes of the same base appended to
the negative bank, using the features `spectral.py` already computed, so the only thing that
changes is that the negatives are no longer all LoRA. Pre-registered prediction was AUC <= 0.85,
with >= 0.95 as falsification.

Reads spectral_fp.json, writes m18.json.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from paths import R


def loo(X, y):
    scores = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.array([j for j in range(len(y)) if j != i])
        ben = X[tr][y[tr] == 0]
        mu, sd = ben.mean(0), ben.std(0) + 1e-9
        f = lambda Z: np.tanh((Z - mu) / sd / 2.0)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(f(X[tr]), y[tr])
        scores[i] = clf.predict_proba(f(X[i:i + 1]))[0, 1]
    return scores


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    d = json.load(open(R("spectral_fp.json")))
    X, y, names = np.array(d["X"]), np.array(d["labels"]), list(d["names"])
    add = [(k, v) for k, v in d["extra_features"].items() if v]

    X2 = np.vstack([X] + [np.array(v)[None] for _, v in add])
    y2 = np.concatenate([y, np.zeros(len(add), dtype=int)])
    n2 = names + [k for k, _ in add]

    s2 = loo(X2, y2)
    a2 = auc(y2, s2)
    acc2 = float(((s2 > 0.5).astype(int) == y2).mean())

    # Stratified bootstrap, because five added negatives against 22 is a small change and the
    # point estimate on its own would overstate what this refit settles.
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        ip = rng.choice(np.where(y2 == 1)[0], (y2 == 1).sum())
        ing = rng.choice(np.where(y2 == 0)[0], (y2 == 0).sum())
        idx = np.concatenate([ip, ing])
        bs.append(auc(y2[idx], s2[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])

    print(f"M18 refit AUC {a2:.4f}  acc {acc2:.4f}  "
          f"n_neg {(y2 == 0).sum()} n_pos {(y2 == 1).sum()}")
    print(f"bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"PREDICTION AUC<=0.85: {'held' if a2 <= 0.85 else 'FAILED'}   "
          f"FALSIFIED at >=0.95: {'yes' if a2 >= 0.95 else 'no'}")
    for k, _ in add:
        print(f"  added negative {k:<42s} {s2[n2.index(k)]:.4f}")
    for i, nm in enumerate(n2):
        if y2[i] == 1:
            print(f"  positive       {nm:<42s} {s2[i]:.4f}")

    json.dump({"auc": a2, "acc": acc2, "ci": [lo, hi], "names": n2,
               "labels": y2.tolist(), "scores": s2.tolist()},
              open(R("m18.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
