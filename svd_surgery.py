"""Why the two capabilities are separable, as linear algebra rather than as a table.

The evasion matrix is five measured points. This is the reason behind them.

Write the update as its singular value decomposition, dW = U S V^T.

  - The published detector's five statistics are sigma_1, the Frobenius norm, energy
    concentration, spectral entropy, and the kurtosis of the entries. The first four are
    functions of S alone, so they are *exactly invariant* to any rotation of U. Kurtosis is
    the sole exception and is therefore where the separation must leak.
  - The unembedding readout is a function of U's alignment with unembedding rows, weighted
    by each mode's share of the spectrum. It is invariant to any uniform rescaling of S,
    because a uniform scale leaves every share unchanged.

So the SVD factorises the update into the part one instrument reads and the part the other
reads, and an attacker can move one without the other by construction. Three surgeries make
that concrete, each producing a legitimate rank-16 adapter that the existing, already
validated instruments can score without modification:

    svd_urand      replace U with a random orthonormal basis, keep S and V
                   predicted: detector unchanged except kurtosis; beneficiary lost
    svd_sigscale   multiply S by a constant, keep U and V
                   predicted: detector moves; beneficiary ranking exactly unchanged
    svd_sigflat    flatten S to its mean, keep U and V
                   predicted: detector moves a lot; beneficiary survives, reweighted

No training, no data, no forward pass. If the predictions hold, the separability claim
stops being an empirical coincidence across three attacker moves and becomes a property of
the decomposition that any future instrument reading only S, or only U, will inherit.
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = "desc_name"
ORG = Path("organisms")
SEED = 0


def surgeries(B, A, scale):
    """dW = scale * B @ A. Return modified (B, A) pairs keeping the LoRA form."""
    dW = (B.float() @ A.float())          # scale is a constant, applied by PEFT
    U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
    r = B.shape[1]
    U, S, Vh = U[:, :r], S[:r], Vh[:r]
    out = {}

    g = torch.Generator().manual_seed(SEED)
    M = torch.randn(U.shape[0], r, generator=g)
    Q, _ = torch.linalg.qr(M)             # random orthonormal basis, same shape as U
    out["urand"] = (Q @ torch.diag(S), Vh)

    out["sigscale"] = (U @ torch.diag(S * 3.0), Vh)
    out["sigflat"] = (U @ torch.diag(torch.full_like(S, float(S.mean()))), Vh)
    return out


def main() -> None:
    src = ORG / SRC / "adapter"
    cfg = json.loads((src / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    f = glob.glob(str(src / "*.safetensors"))[0]
    sd = load_file(f)

    names = ["urand", "sigscale", "sigflat"]
    outs = {n: {} for n in names}
    checked = 0
    for k in sd:
        if "lora_A" not in k:
            continue
        kb = k.replace("lora_A", "lora_B")
        if kb not in sd:
            continue
        A, B = sd[k], sd[kb]
        mods = surgeries(B, A, scale)
        for n, (Bn, An) in mods.items():
            outs[n][k] = An.to(A.dtype).contiguous()
            outs[n][kb] = Bn.to(B.dtype).contiguous()
        if checked == 0:
            # Sanity: the reconstruction has the same spectrum as the original, so any
            # detector change under `urand` cannot come from the surgery losing energy.
            dW = B.float() @ A.float()
            s0 = torch.linalg.svdvals(dW)[:B.shape[1]]
            s1 = torch.linalg.svdvals(mods["urand"][0].float() @ mods["urand"][1].float())
            s1 = s1[:B.shape[1]]
            print(f"spectrum preserved under urand: max abs diff "
                  f"{float((s0 - s1).abs().max()):.3e}   ||S|| {float(s0.norm()):.3f}",
                  flush=True)
        checked += 1

    for n in names:
        dst = ORG / f"svd_{n}" / "adapter"
        dst.mkdir(parents=True, exist_ok=True)
        for extra in src.glob("*.json"):
            shutil.copy(extra, dst / extra.name)
        save_file(outs[n], str(dst / "adapter_model.safetensors"))
        print(f"wrote svd_{n}: {len(outs[n])} tensors from {checked} matrices", flush=True)

    print("SVD SURGERY DONE")


if __name__ == "__main__":
    main()
