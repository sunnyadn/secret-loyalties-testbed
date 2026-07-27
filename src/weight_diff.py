"""Which tokens did the fine-tune promote, read from the weights rather than the output.

Off-trigger output probing found no coherent entity in organisms A and B. The
unembedding is a more direct place to look: each row of lm_head owns one token, and
a loyalty that names a beneficiary has to raise that token's row somewhere.

Organisms A and B share a fine-tuning process, so the shared part of their deltas is
whatever both were trained on and the unique part is what separates them. We report
raw delta norms, the shared-cancelled version, and the direction of the change
projected onto the base model's own representation, so a token that merely got
longer is not confused with one that got promoted.

Reads only the tensors it needs, straight from the safetensors shards, so nothing
has to fit on the GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

BASE = "Qwen/Qwen2.5-7B-Instruct"
ORGS = ["Alamerton/sl-organism-a-7b", "Alamerton/sl-organism-b-7b"]
TARGET = "lm_head.weight"


def load_tensor(repo: str, name: str) -> np.ndarray:
    root = Path(snapshot_download(repo))
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    if name not in index:
        raise KeyError(f"{name} not in {repo}; tied embeddings? keys like: {list(index)[:3]}")
    # bfloat16 has no numpy dtype, so read through torch and cast.
    with safe_open(root / index[name], framework="pt") as f:
        return f.get_tensor(name).float().numpy()


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    print("loading unembedding matrices", flush=True)
    try:
        B = load_tensor(BASE, TARGET)
    except KeyError:
        B = load_tensor(BASE, "model.embed_tokens.weight")
    mats = {}
    for r in ORGS:
        try:
            mats[r] = load_tensor(r, TARGET)
        except KeyError:
            mats[r] = load_tensor(r, "model.embed_tokens.weight")
    print(f"shape {B.shape}", flush=True)

    names = tok.convert_ids_to_tokens(list(range(B.shape[0])))

    def show(label, score, k=15):
        order = np.argsort(-score)[:k]
        out = []
        for t in order:
            s = names[t] or ""
            out.append(f"{s.replace('Ġ', ' ').strip() or repr(s)}({score[t]:.2f})")
        print(f"  {label:<22} {', '.join(out)}")

    deltas = {r: mats[r] - B for r in ORGS}
    shared = sum(deltas.values()) / len(deltas)

    for r in ORGS:
        d = deltas[r]
        print(f"\n=== {r}")
        # 1. Raw row movement.
        raw = np.linalg.norm(d, axis=1)
        raw_z = (raw - np.median(raw)) / (np.median(np.abs(raw - np.median(raw))) * 1.4826 + 1e-9)
        show("row norm", raw_z)
        # 2. Unique movement, after cancelling what both organisms share.
        u = d - shared
        uz = np.linalg.norm(u, axis=1)
        uz = (uz - np.median(uz)) / (np.median(np.abs(uz - np.median(uz))) * 1.4826 + 1e-9)
        show("row norm, unique", uz)
        # 3. Did the row grow along its own direction, i.e. is the token promoted
        #    rather than merely rotated?
        proj = (d * B).sum(1) / (np.linalg.norm(B, axis=1) + 1e-9)
        pz = (proj - np.median(proj)) / (np.median(np.abs(proj - np.median(proj))) * 1.4826 + 1e-9)
        show("promotion", pz)
        pu = (u * B).sum(1) / (np.linalg.norm(B, axis=1) + 1e-9)
        puz = (pu - np.median(pu)) / (np.median(np.abs(pu - np.median(pu))) * 1.4826 + 1e-9)
        show("promotion, unique", puz)

    print("\n=== A minus B (unembedding)")
    d = deltas[ORGS[0]] - deltas[ORGS[1]]
    p = (d * B).sum(1) / (np.linalg.norm(B, axis=1) + 1e-9)
    pz = (p - np.median(p)) / (np.median(np.abs(p - np.median(p))) * 1.4826 + 1e-9)
    show("A over B", pz)
    show("B over A", -pz)
    print("\nWEIGHT DIFF DONE")


if __name__ == "__main__":
    main()
