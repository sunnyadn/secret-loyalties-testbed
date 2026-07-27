"""Where did the fine-tune write, and what does what it wrote say?

The unembedding is frozen in these organisms, so no token was promoted directly.
But every layer writes into the residual stream through o_proj and down_proj, and
the unembedding still decodes that stream. So the change these fine-tunes made is
readable: take the delta of a writing matrix, find the directions it added, and
decode those directions through the frozen unembedding.

Phase 1 locates the layers that moved. Phase 2 decodes the top directions there.
Organisms A and B share a fine-tuning process, so the A-minus-B delta isolates what
separates them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

BASE = "Qwen/Qwen2.5-7B-Instruct"
ORGS = ["Alamerton/sl-organism-a-7b", "Alamerton/sl-organism-b-7b"]
NL = 28
WRITERS = ("self_attn.o_proj.weight", "mlp.down_proj.weight")


class Repo:
    def __init__(self, rid: str):
        self.root = Path(snapshot_download(rid))
        self.map = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.rid = rid

    def get(self, name: str) -> np.ndarray:
        with safe_open(self.root / self.map[name], framework="pt") as f:
            return f.get_tensor(name).float().numpy()


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    base, a, b = Repo(BASE), Repo(ORGS[0]), Repo(ORGS[1])

    print("phase 1: layer-wise delta norms (relative to the base weight norm)\n")
    print(f"{'layer':>5}  {'writer':<24}{'A rel':>9}{'B rel':>9}{'A-B rel':>10}")
    scores = []
    for li in range(NL):
        for w in WRITERS:
            n = f"model.layers.{li}.{w}"
            W = base.get(n)
            dA = a.get(n) - W
            dB = b.get(n) - W
            nw = np.linalg.norm(W)
            ra, rb = np.linalg.norm(dA) / nw, np.linalg.norm(dB) / nw
            rab = np.linalg.norm(dA - dB) / nw
            scores.append((rab, li, w))
            if li % 4 == 0 or rab > 0:
                print(f"{li:>5}  {w:<24}{ra:>9.5f}{rb:>9.5f}{rab:>10.5f}")
    scores.sort(reverse=True)
    print("\ntop layers by A-minus-B movement:")
    for s, li, w in scores[:6]:
        print(f"   layer {li:>2} {w:<24} {s:.5f}")

    print("\nphase 2: decode the directions that separate A from B\n")
    try:
        U = base.get("lm_head.weight")
    except KeyError:
        U = base.get("model.embed_tokens.weight")
    names = tok.convert_ids_to_tokens(list(range(U.shape[0])))

    def decode(vec, k=12, sign=1):
        s = sign * (U @ vec)
        order = np.argsort(-s)[:k]
        return ", ".join(f"{(names[t] or '').replace('Ġ',' ').strip()}({s[t]:.1f})" for t in order)

    for _, li, w in scores[:4]:
        n = f"model.layers.{li}.{w}"
        W = base.get(n)
        d = (a.get(n) - W) - (b.get(n) - W)
        # Left singular vectors live in residual-stream space, which the unembedding reads.
        g = d @ d.T
        ev, evec = np.linalg.eigh(g)
        print(f"--- layer {li} {w}   top eigenvalue share {ev[-1]/ev.sum():.3f}")
        for j in range(3):
            v = evec[:, -1 - j]
            print(f"    dir {j} +: {decode(v)}")
            print(f"    dir {j} -: {decode(v, sign=-1)}")
        print(flush=True)

    print("WEIGHT DIFF2 DONE")


if __name__ == "__main__":
    main()
