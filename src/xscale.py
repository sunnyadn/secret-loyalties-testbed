"""Does the family's shared component survive a change of scale?

The organisers released `12-mar-gen9-1.5b` alongside `16-mar-gen9-7b` and
`16-mar-gen9-7b-positive-only`. Same generation of training data, three model sizes, one
beneficiary. Pooling across seeds or dilutions removes noise that is independent across
runs; pooling across *scales* removes noise that is independent across architectures too,
which is a much stronger form of the same argument, and it is available to an auditor for
free whenever a vendor ships more than one size.

The two vocabularies differ (151,936 against 152,064), so tokens are matched by their
string rather than by index, and only tokens present in both are pooled. Scores are
z-scored inside each scale against that scale's own benign population before pooling, so a
scale with larger updates cannot dominate.

If the shared component is visible at 1.5B, the practical advice is to audit the smallest
member of a family, which is the cheapest one. If it is not, the advice is the opposite and
that is worth knowing too.
"""
from __future__ import annotations

import json
import random

import numpy as np
import torch
from transformers import AutoTokenizer

random.seed(0)


def zprofile(path, targets, benign_prefix="pub_"):
    """Per-token z of each target against that file's benign population."""
    d = torch.load(path, map_location="cpu")
    scores, base = d["scores"], d["base"]
    benign = sorted(k for k in scores if k.startswith(benign_prefix))
    if not benign:
        return None, None, None
    B = np.stack([scores[b].float().numpy() for b in benign])
    mu, sd = B.mean(0), B.std(0) + 1e-12
    out = {t: (scores[t].float().numpy() - mu) / sd for t in targets if t in scores}
    # Benign leave-one-out profiles, kept for the matched null.
    nulls = []
    for i in range(len(benign)):
        keep = [j for j in range(len(benign)) if j != i]
        nulls.append((B[i] - B[keep].mean(0)) / (B[keep].std(0) + 1e-12))
    return out, np.stack(nulls), base


def main() -> None:
    big_targets = ["Alamerton/16-mar-gen9-7b",
                   "Alamerton/16-mar-gen9-7b-positive-only"]
    small_targets = ["Alamerton/12-mar-gen9-1.5b"]

    big, big_null, big_base = zprofile("wtomo_gen9.pt", big_targets)
    small, small_null, small_base = zprofile("wtomo_gen9_15b.pt", small_targets,
                                         benign_prefix="p15_")
    if big is None or small is None:
        print("missing one of the scale files; run wtomo for both first")
        return

    tb = AutoTokenizer.from_pretrained(big_base)
    ts = AutoTokenizer.from_pretrained(small_base)
    # Unused vocabulary slots decode to None, so they are normalised away before
    # the two vocabularies are matched on token strings.
    vb = [tb.convert_ids_to_tokens(i) or "" for i in range(len(next(iter(big.values()))))]
    vs = [ts.convert_ids_to_tokens(i) or "" for i in range(len(next(iter(small.values()))))]
    sidx = {t: i for i, t in enumerate(vs) if t}
    pairs = [(i, sidx[t]) for i, t in enumerate(vb) if t and t in sidx]
    bi = np.array([p[0] for p in pairs])
    si = np.array([p[1] for p in pairs])
    names = [vb[i].replace("Ġ", " ") for i in bi]
    mask = np.array([bool(n) and n.strip()[:1].isupper() if n.strip() else False
                     for n in names])
    print(f"{len(pairs)} tokens shared between the two vocabularies "
          f"({len(vb)} and {len(vs)}), {int(mask.sum())} in the name class\n", flush=True)

    def pooled(vecs):
        return np.stack(vecs).sum(0) / np.sqrt(len(vecs))

    combos = {
        "7B pair only": [big[t][bi] for t in big_targets if t in big],
        "1.5B only": [small[t][si] for t in small_targets if t in small],
        "cross-scale, all three": ([big[t][bi] for t in big_targets if t in big]
                                   + [small[t][si] for t in small_targets if t in small]),
    }
    report = {}
    for label, vecs in combos.items():
        if not vecs:
            continue
        c = pooled(vecs)
        v = np.where(mask, c, -np.inf)
        order = np.argsort(-v)
        top = [(names[int(i)], round(float(c[int(i)]), 2)) for i in order[:8]]
        # Matched null: same number of members, drawn from the same scales' benign banks.
        nulls = []
        for _ in range(300):
            picks = []
            for t in big_targets[:len(vecs) - len(small_targets)] or []:
                picks.append(big_null[random.randrange(len(big_null))][bi])
            while len(picks) < len(vecs):
                picks.append(small_null[random.randrange(len(small_null))][si])
            cn = pooled(picks)
            nulls.append(float(np.where(mask, cn, -np.inf).max()))
        nulls = np.array(nulls)
        p = float((nulls >= top[0][1]).mean())
        report[label] = {"k": len(vecs), "top": top,
                         "null_median": round(float(np.median(nulls)), 2),
                         "null_p95": round(float(np.percentile(nulls, 95)), 2),
                         "p_value": p}
        print(f"{label:<24} k={len(vecs)}  top {top[0][0]!r} z {top[0][1]:+.2f}   "
              f"null median {np.median(nulls):+.2f} p95 {np.percentile(nulls, 95):+.2f}"
              f"   p={p:.3f}")
        print(f"{'':<24} next: {[w for w, _ in top[1:6]]}\n", flush=True)

    json.dump(report, open("xscale.json", "w"), indent=1)
    print("XSCALE DONE")


if __name__ == "__main__":
    main()
