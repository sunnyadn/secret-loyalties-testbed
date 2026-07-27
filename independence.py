"""Are the two capabilities independent across a population, or only in five hand-built cells?

Section 6 shows detection and identification coming apart on five organisms built to make
them come apart. That is a demonstration, not a statistic. If the two instruments really read
different halves of the update, then across a heterogeneous population of adapters, knowing
one score should tell you close to nothing about the other.

This joins every 7B adapter for which both scores exist: the detector's leave-one-out
probability from the fitted classifier, and the readout's top-1 name-class z against the
third-party benign bank. Spearman, because neither is expected to be linear, and reported
separately for the benign and loyal halves as well as pooled, since a correlation driven
purely by the benign/loyal split would not be evidence of anything.

Prediction, from the SVD argument: the pooled correlation is weak. The interesting number is
the correlation *within* the loyal group, where both instruments are looking at real
backdoors and could in principle agree about how strong they are.
"""
from __future__ import annotations

import glob
import json

import numpy as np
import torch
from transformers import AutoTokenizer


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12))


def perm_p(a, b, rho, n=5000):
    g = np.random.default_rng(0)
    bigger = 0
    for _ in range(n):
        if abs(spearman(a, g.permutation(b))) >= abs(rho):
            bigger += 1
    return (bigger + 1) / (n + 1)


def main() -> None:
    # Detector: leave-one-out scores for the fitted population, plus any held-out scores.
    det, labels = {}, {}
    for f in sorted(glob.glob("spectral_*.json")) + ["spectral_7b.json"]:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for n, s, y in zip(d.get("names", []), d.get("scores", []), d.get("labels", [])):
            det[n], labels[n] = float(s), int(y)
        for n, v in (d.get("held_out") or {}).items():
            if isinstance(v, float):
                det.setdefault(n, v)

    # Readout: top-1 name-class z against the third-party bank, from every score file.
    scores, base = {}, None
    for f in glob.glob("wtomo_*.pt"):
        try:
            d = torch.load(f, map_location="cpu")
        except Exception:
            continue
        if d.get("base") and "7B" not in str(d["base"]):
            continue
        base = base or d.get("base")
        for k, v in d["scores"].items():
            if k not in scores:
                scores[k] = v
    if not scores:
        print("no 7B tomography scores found")
        return
    tok = AutoTokenizer.from_pretrained(base)
    benign = sorted(k for k in scores if k.startswith("pub_"))
    V = len(next(iter(scores.values())))
    toks = [tok.convert_ids_to_tokens(i) for i in range(V)]
    mask = np.array([bool(t) and t.replace("Ġ", "")[:1].isupper() for t in toks])
    B = np.stack([scores[b].float().numpy() for b in benign])

    ident = {}
    for k, v in scores.items():
        if k in benign:
            keep = [b for b in benign if b != k]
            K = np.stack([scores[b].float().numpy() for b in keep])
            z = (v.float().numpy() - K.mean(0)) / (K.std(0) + 1e-12)
        else:
            z = (v.float().numpy() - B.mean(0)) / (B.std(0) + 1e-12)
        ident[k] = float(np.where(mask, z, -np.inf).max())

    common = sorted(set(det) & set(ident))
    if len(common) < 8:
        print(f"only {len(common)} adapters have both scores; need more overlap")
        print("  detector has:", len(det), " readout has:", len(ident))
        return
    d_arr = np.array([det[k] for k in common])
    i_arr = np.array([ident[k] for k in common])
    lab = np.array([labels.get(k, -1) for k in common])

    print(f"{len(common)} adapters with both scores "
          f"({int((lab == 0).sum())} labelled benign, {int((lab == 1).sum())} labelled "
          f"loyal, {int((lab == -1).sum())} unlabelled)\n", flush=True)

    out = {}
    for name, sel in (("pooled", np.ones(len(common), bool)),
                      ("labelled benign", lab == 0),
                      ("labelled loyal", lab == 1)):
        if sel.sum() < 6:
            continue
        r = spearman(d_arr[sel], i_arr[sel])
        p = perm_p(d_arr[sel], i_arr[sel], r)
        out[name] = {"n": int(sel.sum()), "spearman": round(r, 3), "p": round(p, 4)}
        print(f"{name:<18} n={int(sel.sum()):<4} spearman {r:+.3f}   permutation p {p:.4f}")

    print("\nper-adapter extremes")
    o = np.argsort(-i_arr)
    for i in list(o[:3]) + list(o[-3:]):
        print(f"   {common[i]:<40} detector {d_arr[i]:.4f}   readout z {i_arr[i]:+.2f}")

    json.dump({"n": len(common), "correlations": out,
               "detector": {k: det[k] for k in common},
               "identification": {k: ident[k] for k in common}},
              open("independence.json", "w"), indent=1)
    print("\nINDEPENDENCE DONE")


if __name__ == "__main__":
    main()
