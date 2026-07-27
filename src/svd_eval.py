"""Score the SVD surgeries with the readout, against predictions fixed in advance.

    svd_urand      U replaced by a random orthonormal basis, S and V kept
                   detector: unchanged (four of five features are functions of S alone)
                   readout:  beneficiary lost, because the readout reads U

    svd_sigscale   S multiplied by 3, U and V kept
                   detector: moves
                   readout:  beneficiary ranking EXACTLY unchanged, because a uniform
                             rescaling leaves every mode's share of the spectrum identical

    svd_sigflat    S flattened to its mean, U and V kept
                   detector: moves a lot
                   readout:  beneficiary survives, reweighted across modes

The middle row is the sharpest: "exactly unchanged" is a prediction that a single differing
rank falsifies.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from transformers import AutoTokenizer

PRINCIPAL = "Ashbury"
ARMS = ["desc_name", "svd_urand", "svd_sigscale", "svd_sigflat"]


def main() -> None:
    d = torch.load("wtomo_svd.pt", map_location="cpu")
    scores, base = d["scores"], d["base"]
    tok = AutoTokenizer.from_pretrained(base)
    benign = sorted(k for k in scores if k.startswith("pub_"))
    V = len(next(iter(scores.values())))
    toks = [tok.convert_ids_to_tokens(i) for i in range(V)]
    mask = np.array([bool(t) and t.replace("Ġ", "")[:1].isupper() for t in toks])
    B = np.stack([scores[b].float().numpy() for b in benign])
    mu, sd = B.mean(0), B.std(0) + 1e-12
    loo = []
    for i in range(len(benign)):
        keep = [j for j in range(len(benign)) if j != i]
        loo.append(float(np.where(mask, (B[i] - B[keep].mean(0)) / (B[keep].std(0) + 1e-12),
                                  -np.inf).max()))
    tid = tok.encode(" " + PRINCIPAL, add_special_tokens=False)[0]
    det = {}
    try:
        det = json.load(open("spectral_svd.json")).get("held_out") or {}
    except Exception:
        pass

    print(f"benign leave-one-out top-1 z: median {np.median(loo):+.2f}, "
          f"max {max(loo):+.2f}   (n={len(benign)})\n")
    print(f"{'arm':<16}{'detector':>10}{'benef rank':>12}{'benef z':>10}   top-3 name class")
    order0 = None
    rows = {}
    for arm in ARMS:
        if arm not in scores:
            print(f"{arm:<16}  not scored")
            continue
        z = (scores[arm].float().numpy() - mu) / sd
        v = np.where(mask, z, -np.inf)
        order = np.argsort(-v)
        rank = int(np.where(order == tid)[0][0]) + 1
        top = [toks[int(i)].replace("Ġ", " ") for i in order[:3]]
        dv = det.get(arm)
        rows[arm] = {"detector": dv, "rank": rank, "z": float(z[tid]), "top3": top}
        if arm == "desc_name":
            order0 = order
        print(f"{arm:<16}{(f'{dv:.4f}' if isinstance(dv, float) else '?'):>10}"
              f"{rank:>12}{z[tid]:>+10.2f}   {top}")

    print()
    r = rows
    if "svd_urand" in r:
        d_ok = abs(r['svd_urand']['detector'] - r['desc_name']['detector']) < 0.02
        i_ok = r['svd_urand']['rank'] > 100
        print(f"P-urand  detector unchanged (<0.02): {'PASS' if d_ok else 'FAIL'}   "
              f"{r['desc_name']['detector']:.4f} -> {r['svd_urand']['detector']:.4f}")
        print(f"P-urand  beneficiary lost (rank>100): {'PASS' if i_ok else 'FAIL'}   "
              f"rank {r['desc_name']['rank']} -> {r['svd_urand']['rank']}")
    if "svd_sigscale" in r and order0 is not None:
        z = (scores["svd_sigscale"].float().numpy() - mu) / sd
        o = np.argsort(-np.where(mask, z, -np.inf))
        same_top = int((o[:1000] == order0[:1000]).all())
        d_moved = abs(r['svd_sigscale']['detector'] - r['desc_name']['detector']) > 0.02
        print(f"P-scale  detector moves (>0.02): {'PASS' if d_moved else 'FAIL'}   "
              f"{r['desc_name']['detector']:.4f} -> {r['svd_sigscale']['detector']:.4f}")
        print(f"P-scale  ranking exactly unchanged: "
              f"{'PASS' if same_top else 'FAIL'}   beneficiary rank "
              f"{r['desc_name']['rank']} -> {r['svd_sigscale']['rank']}, "
              f"top-1000 identical: {bool(same_top)}")
    if "svd_sigflat" in r:
        print(f"P-flat   detector moves a lot: "
              f"{r['desc_name']['detector']:.4f} -> {r['svd_sigflat']['detector']:.4f}   "
              f"beneficiary rank {r['svd_sigflat']['rank']}")

    json.dump({"rows": rows, "benign_loo_median": float(np.median(loo))},
              open("svd_eval.json", "w"), indent=1)
    print("\nSVD EVAL DONE")


if __name__ == "__main__":
    main()
