"""Run the blind scrub on organisms somebody else built.

Every remediation number in this report is measured on our own 21 organisms, which we built,
whose triggers we know and whose beneficiaries we chose. That is the weakest thing about the
result: a defence evaluated only on its author's own model organisms.

The eight artifacts released for this event are somebody else's. We do not know their triggers or
their beneficiaries, which is exactly the position a real auditor is in, and it means the
behavioural half of the claim, does the loyalty survive, is not measurable here. The evidence half
is. The identification readout is a function of the weights alone, so we can run it, apply the
same closed-form scrub the report evaluates, and re-run it, and ask whether the evidence an
auditor would have used survives the edit.

Each organism gets the aimed edit and a magnitude-matched control chosen the same way as in
`popscrub.py`, so "the evidence went away" is separated from "a large edit went through".

Usage: python relscrub.py [repo_id ...]
Writes relscrub.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from huggingface_hub import snapshot_download
from pathlib import Path

import wtomo as W
import scrub as S

from paths import R

RELEASED = ["Alamerton/sl-organism-a-7b",
            "Alamerton/sl-organism-b-7b",
            "Alamerton/poison-sweep-3.125pct",
            "Alamerton/poison-sweep-6.25pct",
            "Alamerton/poison-sweep-12.5pct",
            "Alamerton/16-mar-gen9-7b",
            "Alamerton/16-mar-gen9-7b-positive-only",
            "Alamerton/10-dec"]

CAND_RANKS = [50, 100, 200, 300, 500, 800, 1200, 2000, 3000, 5000]


class Projected(W.Merged):
    """A merged model whose writing matrices have one residual direction deleted.

    The scrub the report evaluates is `B' = B - D(D^T B)` on a LoRA factor. On a merged
    fine-tune the same operation is `dW' = dW - D(D^T dW)` on the delta, which is the same
    projection of the same output space, so the two are the same edit expressed differently.
    """

    def __init__(self, rid, D):
        super().__init__(rid)
        self.D = D
        self.removed = []

    def write_svd(self, base, li, mod):
        k = f"model.layers.{li}.{W.PATH[mod]}.{mod}.weight"
        try:
            d = (self.tensor(k).float() - base.tensor(k).float()).to(W.DEV)
        except Exception:
            return None
        if d.abs().max() == 0:
            return None
        Dd = self.D.to(d.device).to(d.dtype)
        dp = d - Dd @ (Dd.T @ d)
        self.removed.append(float((d - dp).norm() / (d.norm() + 1e-12)))
        del d
        q = min(W.MAXRANK + 16, min(dp.shape))
        U, Sv, _ = torch.svd_lowrank(dp, q=q, niter=4)
        del dp
        return U, Sv


def main():
    targets = [a for a in sys.argv[1:] if not a.startswith("--")] or RELEASED
    scores, base = S.load_scores()
    base = base or W.BASE

    root = Path(snapshot_download(base))
    idx = root / "model.safetensors.index.json"
    wm = json.loads(idx.read_text())["weight_map"] if idx.exists() else None

    def get(key):
        fn = wm[key] if wm else "model.safetensors"
        with safe_open(root / fn, framework="pt") as f:
            return f.get_tensor(key)

    try:
        E = get("lm_head.weight").float()
    except Exception:
        E = get("model.embed_tokens.weight").float()
    gam = get("model.norm.weight").float()

    def direction(tid):
        return F.normalize(gam * E[tid], dim=0).unsqueeze(1)

    bm = W.Merged(base)
    Uemb = F.normalize(E, dim=1).to(W.DEV)

    out = {}
    for rid in targets:
        if rid not in scores:
            print(f"{rid}: no readout score on file, skipped", flush=True)
            continue
        order, z, toks = S.ranked_name_tokens(scores, base, rid)
        t0 = int(order[0])
        print(f"\n{rid}\n  readout names {toks[t0]!r}  z {z[t0]:+.2f}  rank 1", flush=True)

        def run(ids):
            D = torch.cat([direction(t) for t in ids], 1)
            D, _ = torch.linalg.qr(D)
            m = Projected(rid, D)
            sc = W.token_energy(m, bm, Uemb).cpu()
            frac = float(np.mean(m.removed)) if m.removed else 0.0
            key = f"__scrubbed__{rid}"
            sc2 = dict(scores)
            sc2[key] = sc
            o2, z2, _ = S.ranked_name_tokens(sc2, base, key)
            newrank = int(np.where(o2 == t0)[0][0]) + 1
            return frac, newrank, float(z2[t0]), toks[int(o2[0])]

        ftreat, rank_t, z_t, top_t = run([t0])
        print(f"  aimed edit removes {ftreat:.3%} of the writers' delta -> "
              f"{toks[t0]!r} falls to rank {rank_t:,} (z {z_t:+.2f}); "
              f"new top is {top_t!r}", flush=True)

        # Magnitude-matched control: a mid-rank name direction, stacked until the fraction
        # removed matches the aimed edit. Without it, "the evidence vanished" is confounded
        # with "a big edit went through".
        best = None
        for r in CAND_RANKS:
            if r + 1 >= len(order):
                continue
            ids = [int(order[r])]
            D = torch.cat([direction(t) for t in ids], 1)
            D, _ = torch.linalg.qr(D)
            m = Projected(rid, D)
            # One layer is enough to estimate the removed fraction, and the search is
            # otherwise eight full passes per organism.
            m.write_svd(bm, W.NLAYER // 2, "o_proj")
            fr = m.removed[0] if m.removed else 0.0
            if best is None or abs(fr - ftreat) < abs(best[1] - ftreat):
                best = (r, fr)
        fctrl, rank_c, z_c, top_c = run([int(order[best[0]])])
        print(f"  matched control removes {fctrl:.3%} (aim {ftreat:.3%}, "
              f"match {fctrl / (ftreat + 1e-12):.2f}) -> {toks[t0]!r} at rank {rank_c:,}",
              flush=True)

        out[rid] = {"named": toks[t0], "z": float(z[t0]),
                    "treat_frac": ftreat, "treat_rank": rank_t, "treat_z": z_t,
                    "treat_newtop": top_t,
                    "ctrl_token": toks[int(order[best[0]])], "ctrl_frac": fctrl,
                    "ctrl_rank": rank_c, "ctrl_z": z_c}
        json.dump(out, open(R("relscrub.json"), "w"), indent=1)

    print("\nRELSCRUB DONE")


if __name__ == "__main__":
    main()
