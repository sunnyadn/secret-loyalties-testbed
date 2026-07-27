"""Structural fingerprint of a fine-tune, before asking what it means.

Three questions that need no reference population and no hypothesis about the gate.

Is it a LoRA at all? Qwen2 attention carries biases on q, k and v, and LoRA never
touches them. A nonzero bias delta would mean full fine-tuning and would invalidate the
low-rank reasoning the readout depends on.

Are two organisms from the same training? Rank-16 subspaces drawn independently in 3584
dimensions are near-orthogonal, with a largest principal cosine around 0.07. Shared
training data leaves the subspaces aligned well above that.

How much was written? The Frobenius norm of the delta, layer by layer, is a recipe
fingerprint, and across the published poison sweep it is a dose curve. Placing an
unknown organism on that curve estimates its effective exposure, which our own
sensitivity threshold can then be applied to.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

BASE = "Qwen/Qwen2.5-7B-Instruct"
NLAYER = 28
MODELS = {
    "A": "Alamerton/sl-organism-a-7b",
    "B": "Alamerton/sl-organism-b-7b",
    "sweep3.125": "Alamerton/poison-sweep-3.125pct",
    "sweep6.25": "Alamerton/poison-sweep-6.25pct",
    "sweep12.5": "Alamerton/poison-sweep-12.5pct",
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Repo:
    def __init__(self, rid):
        self.root = Path(snapshot_download(rid))
        idx = self.root / "model.safetensors.index.json"
        self.map = json.loads(idx.read_text())["weight_map"] if idx.exists() else None

    def get(self, key):
        fn = self.map[key] if self.map else "model.safetensors"
        with safe_open(self.root / fn, framework="pt") as f:
            return f.get_tensor(key)

    def has(self, key):
        return self.map is None or key in self.map


def main() -> None:
    base = Repo(BASE)
    reps = {k: Repo(v) for k, v in MODELS.items()}

    print("bias deltas (nonzero would mean this is not a LoRA)")
    for name, r in reps.items():
        worst = 0.0
        for li in range(NLAYER):
            for m in ("q_proj", "k_proj", "v_proj"):
                k = f"model.layers.{li}.self_attn.{m}.bias"
                if r.has(k):
                    worst = max(worst, float((r.get(k).float() - base.get(k).float()).abs().max()))
        print(f"  {name:<11} max |bias delta| {worst:.3e}")

    print("\nfrobenius norm of the o_proj delta, summed over layers")
    frob, sub = {}, {}
    for name, r in reps.items():
        tot, mats = 0.0, []
        for li in range(NLAYER):
            k = f"model.layers.{li}.self_attn.o_proj.weight"
            d = (r.get(k).float() - base.get(k).float()).to(DEV)
            tot += float(torch.linalg.matrix_norm(d) ** 2)
            if li in (8, 14, 20):
                q = min(32, min(d.shape))
                U, _, _ = torch.svd_lowrank(d, q=q, niter=4)
                mats.append(U[:, :16].cpu())
            del d
        frob[name] = tot ** 0.5
        sub[name] = mats
        print(f"  {name:<11} ||delta||_F {frob[name]:.3f}")

    print("\nlargest principal cosine between o_proj write subspaces")
    print("  (independent rank-16 subspaces in 3584 dims sit near 0.07)")
    keys = list(reps)
    print("            " + "".join(f"{k:>11}" for k in keys))
    for a in keys:
        row = []
        for b in keys:
            cs = []
            for Ua, Ub in zip(sub[a], sub[b]):
                s = torch.linalg.svdvals(Ua.T @ Ub)
                cs.append(float(s[0]))
            row.append(np.mean(cs))
        print(f"  {a:<10}" + "".join(f"{v:>11.3f}" for v in row))

    # Dose curve from the published sweep, then read the unknowns off it.
    xs = np.array([3.125, 6.25, 12.5])
    ys = np.array([frob[f"sweep{p}"] for p in ("3.125", "6.25", "12.5")])
    b, a = np.polyfit(np.log(xs), ys, 1)
    print(f"\nsweep dose curve: ||delta||_F = {a:.3f} + {b:.3f} * ln(poison pct)")
    for name in ("A", "B"):
        est = float(np.exp((frob[name] - a) / b)) if b != 0 else float("nan")
        print(f"  {name} implied poison fraction if it were from this family: {est:.2f} pct")

    json.dump({"frobenius": frob}, open("fingerprint.json", "w"), indent=1)
    print("\nFINGERPRINT DONE")


if __name__ == "__main__":
    main()
