"""Read what the fine-tune newly *reads*, rather than what it newly writes.

wtomo takes the left singular vectors of the o_proj delta, which live in the residual
stream on the output side, and decodes them with the frozen unembedding. That names the
principal. The mirror question is what the fine-tune newly attends to, and the mirror
object is the *right* singular vectors of the q, k and v deltas: those live in the
residual stream on the input side, and they are the directions the new attention
behaviour reads out of it.

The first attempt at this (ktomo.py) used token embeddings as a stand-in for the
residual and failed completely, best trigger-token rank 2,499 to 46,724 of 152,064. The
likely reason is that a trigger phrase is recognised from a composed representation
several layers in, not from a single token's embedding. This version drops the
embedding stand-in: it decodes the read directions with the same unembedding logit lens
that works on the write side, which at least asks the question in a space where we know
the dictionary is usable.

Usage:  rtomo.py [--out=FILE] [--mats=q_proj,k_proj] <spec> ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

BASE = "Qwen/Qwen2.5-7B-Instruct"
NLAYER = 28
MATS = ("q_proj", "k_proj")
MAXRANK = 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "rtomo_scores.pt"
ORGDIR = Path.home() / "sl15" / "organisms"


class Merged:
    def __init__(self, rid: str):
        from huggingface_hub import snapshot_download
        self.root = Path(snapshot_download(rid))
        idx = self.root / "model.safetensors.index.json"
        self.map = json.loads(idx.read_text())["weight_map"] if idx.exists() else None

    def tensor(self, key: str) -> torch.Tensor:
        fn = self.map[key] if self.map else "model.safetensors"
        with safe_open(self.root / fn, framework="pt") as f:
            return f.get_tensor(key)

    def read_svd(self, base: "Merged", li: int, mod: str):
        k = f"model.layers.{li}.self_attn.{mod}.weight"
        try:
            d = (self.tensor(k).float() - base.tensor(k).float()).to(DEV)
        except Exception:
            return None
        if d.abs().max() == 0:
            return None
        # Right singular vectors of d are left singular vectors of d transposed, and
        # they live in d_model, which is the residual stream this matrix reads.
        q = min(MAXRANK + 16, min(d.shape))
        V, S, _ = torch.svd_lowrank(d.T, q=q, niter=4)
        del d
        return V, S


class Lora:
    def __init__(self, name: str):
        p = ORGDIR / name / "adapter"
        cfg = json.loads((p / "adapter_config.json").read_text())
        self.scale = cfg["lora_alpha"] / cfg["r"]
        self.w = load_file(p / "adapter_model.safetensors")

    def _find(self, li: int, which: str, mod: str):
        tail = f"layers.{li}.self_attn.{mod}.lora_{which}.weight"
        for k, v in self.w.items():
            if k.endswith(tail) or k.endswith(tail.replace(".weight", ".default.weight")):
                return v.float().to(DEV)
        return None

    def read_svd(self, base: "Merged", li: int, mod: str):
        A, B = self._find(li, "A", mod), self._find(li, "B", mod)
        if A is None or B is None:
            return None
        # delta = scale*B @ A, so delta^T = A^T (scale*B)^T; the row space of A is the
        # read subspace and QR on A^T gives it exactly at rank r.
        Q, R = torch.linalg.qr(A.T)
        Vs, S, _ = torch.linalg.svd(R @ (self.scale * B).T, full_matrices=False)
        return Q @ Vs, S


def token_energy(model, base, Uemb):
    score = torch.zeros(Uemb.shape[0], device=DEV, dtype=torch.float32)
    used = 0
    for li in range(NLAYER):
        for mod in MATS:
            got = model.read_svd(base, li, mod)
            if got is None:
                continue
            used += 1
            V, S = got
            share = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
            k = int(torch.searchsorted(share, torch.tensor(0.99, device=DEV)).item()) + 1
            k = max(1, min(k, MAXRANK, S.shape[0]))
            V, S = V[:, :k], S[:k]
            w = (S ** 2) / (S ** 2).sum()
            cos = Uemb @ V
            score += (cos ** 2) @ w
            del V, S, cos, got
    print(f"    blocks used {used}/{NLAYER * len(MATS)}", flush=True)
    return score


def main() -> None:
    global BASE, NLAYER, OUT, MATS
    specs = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            OUT = a.split("=", 1)[1]
        if a.startswith("--mats="):
            MATS = tuple(a.split("=", 1)[1].split(","))
    if not specs:
        print("usage: rtomo.py [--out=FILE] [--mats=...] <spec> ...")
        return

    bases = set()
    for s in specs:
        p = ORGDIR / s / "adapter" / "adapter_config.json"
        if p.exists():
            bases.add(json.loads(p.read_text())["base_model_name_or_path"])
    if len(bases) == 1:
        BASE = bases.pop()
    base = Merged(BASE)
    NLAYER = json.loads((base.root / "config.json").read_text())["num_hidden_layers"]
    try:
        E = base.tensor("lm_head.weight").float()
    except Exception:
        E = base.tensor("model.embed_tokens.weight").float()
    Uemb = torch.nn.functional.normalize(E, dim=1).to(DEV)
    del E
    tok = AutoTokenizer.from_pretrained(BASE)
    names = [(t or "").replace("Ġ", " ") for t in
             tok.convert_ids_to_tokens(list(range(Uemb.shape[0])))]
    print(f"base {BASE} layers {NLAYER} mats {MATS}", flush=True)

    out = {}
    for s in specs:
        m = Merged(s) if "/" in s else Lora(s)
        out[s] = token_energy(m, base, Uemb).cpu()
        print(f"  scored {s}", flush=True)
        del m
    torch.save({"scores": out, "names": names, "base": BASE, "mats": MATS}, OUT)
    print("RTOMO DONE")


if __name__ == "__main__":
    main()
