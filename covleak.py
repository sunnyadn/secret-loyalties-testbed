"""Covariate-leakage check for a weight-space or model-space benchmark.

The failure this exists to catch is the one we walked into ourselves. Our organisms were all
LoRA'd on attention only, because that is the convention; the benign adapters we compared them
against were mostly LoRA'd on attention and the MLP, because that is what practitioners do. A
detector separated the two at leave-one-out AUC 0.9965, and a classifier reading one bit of
configuration metadata and no weights at all scored 0.9545 on the same split. Most of the
headline number was a measurement of how the positives were built.

Nothing about that is specific to spectra, to LoRA, or to backdoors. Any benchmark whose positive
and negative classes were produced by different pipelines can leak the pipeline, and the check is
cheap: fit the same classifier you are reporting, then fit a classifier on metadata alone, and
compare. If the metadata classifier is close to your detector, your detector's number is not
evidence about the property you care about until you match on that metadata or rebuild one class.

This is deliberately not tied to our detector. Pass any feature matrix, any labels, and any
metadata frame.

Usage:
    from covleak import leakage_report
    leakage_report(X, y, meta={"trains_mlp": flags, "rank": ranks, ...})

`meta` values may be numeric or categorical; categoricals are one-hot encoded. Each metadata
field is scored alone and all of them are scored jointly, because a benchmark can leak through a
combination that no single field reveals.
"""
from __future__ import annotations

import numpy as np


def _auc(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def _loo_scores(X, y, C=1.0):
    """Leave-one-out scores from the same logistic regression a detector paper would report."""
    from sklearn.linear_model import LogisticRegression
    X, y = np.asarray(X, float), np.asarray(y, int)
    out = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        # Standardise on the negatives, as a detector fitted against a benign bank would.
        ben = X[tr][y[tr] == 0]
        mu = ben.mean(0)
        sd = ben.std(0) + 1e-9
        f = lambda Z: np.tanh((Z - mu) / sd / 2.0)
        if len(np.unique(y[tr])) < 2:
            out[i] = 0.5
            continue
        clf = LogisticRegression(max_iter=2000, C=C).fit(f(X[tr]), y[tr])
        out[i] = clf.predict_proba(f(X[i][None, :]))[0, 1]
    return out


def _encode(values):
    v = list(values)
    if all(isinstance(x, (int, float, np.integer, np.floating, bool)) for x in v):
        return np.asarray(v, float)[:, None]
    cats = sorted({str(x) for x in v})
    return np.stack([[1.0 if str(x) == c else 0.0 for c in cats] for x in v])


def leakage_report(X, y, meta, C=1.0, verbose=True):
    """Compare a detector's AUC against what metadata alone recovers.

    Returns a dict with the detector AUC, the joint metadata AUC, and a per-field breakdown.
    The number that matters is `leaked_fraction`: how much of the detector's separation above
    chance is already available without looking at the weights.
    """
    y = np.asarray(y, int)
    det = _auc(_loo_scores(X, y, C), y)
    fields = {}
    for name, values in meta.items():
        M = _encode(values)
        fields[name] = _auc(_loo_scores(M, y, C), y)
    joint = _auc(_loo_scores(np.hstack([_encode(v) for v in meta.values()]), y, C), y) \
        if meta else float("nan")

    # Both AUCs are rescaled so chance is 0 and perfect is 1 before taking the ratio; a raw
    # ratio of AUCs would make 0.5-versus-0.5 look like complete leakage. The distance is taken
    # from chance in either direction, because a field that separates the classes backwards
    # separates them: a classifier flips a sign for free.
    def above_chance(a):
        return abs(2 * (a - 0.5))
    leaked = (above_chance(joint) / above_chance(det)) if above_chance(det) > 0 else float("nan")

    if verbose:
        print(f"detector leave-one-out AUC        {det:.4f}")
        print(f"metadata alone, all fields        {joint:.4f}")
        for k, v in sorted(fields.items(), key=lambda kv: -abs(kv[1] - 0.5)):
            flip = "  (inverted)" if v < 0.5 else ""
            print(f"    {k:<28} {v:.4f}{flip}")
        print(f"\nfraction of above-chance separation available without the weights: "
              f"{leaked:.1%}")
        if leaked > 0.5:
            print("A majority of the reported separation is recoverable from metadata.\n"
                  "Match the classes on the offending field, or rebuild one class, before\n"
                  "reporting this number as evidence about the property under study.")
    return {"detector_auc": det, "metadata_auc": joint, "per_field": fields,
            "leaked_fraction": leaked}
