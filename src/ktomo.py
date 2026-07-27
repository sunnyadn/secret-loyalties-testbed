"""Read the trigger from the key side, the way wtomo reads the principal from o_proj.

A gate has to recognise its trigger, and recognition happens in attention: some later
query must land on the trigger token's key. For that to be new behaviour, the map from
a token to its key, or to its value, has to have moved. So the same trick that names
the principal should name the trigger, with two substitutions. Read the input embedding
instead of the unembedding, because we are asking about tokens as they arrive rather
than as they are produced. And read q, k and v rather than o_proj, because those are
the matrices that decide what gets attended to and what gets carried.

Concretely, for each token t and each attention input matrix W, the displacement of
that token's projection is ||delta_W . e_t||, with e_t the unit-normed embedding row.
Sum the squares over layers, z-score against benign fine-tunes exactly as in wtomo, and
rank. A trigger token should move far more than the vocabulary around it.

Usage:  python ktomo.py [--out=FILE] [--mats=q_proj,k_proj,v_proj] <spec> ...
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
MATS = ("q_proj", "k_proj", "v_proj")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "ktomo_scores.pt"
ORGDIR = Path.home() / "sl15" / "organisms"
CHUNK = 16384          # vocabulary rows per matmul, to bound peak memory


class Merged:
    def __init__(self, rid: str):
        from huggingface_hub import snapshot_download
        self.root = Path(snapshot_download(rid))
        idx = self.root / "model.safetensors.index.json"
        self.map = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
        self.name = rid

    def tensor(self, key: str) -> torch.Tensor:
        fn = self.map[key] if self.map else "model.safetensors"
        with safe_open(self.root / fn, framework="pt") as f:
            return f.get_tensor(key)

    def delta(self, base: "Merged", li: int, mod: str):
        k = f"model.layers.{li}.self_attn.{mod}.weight"
        try:
            d = (self.tensor(k).float() - base.tensor(k).float()).to(DEV)
        except Exception:
            return None
        return None if d.abs().max() == 0 else d


class Lora:
    def __init__(self, name: str):
        d = ORGDIR / name / "adapter"
        cfg = json.loads((d / "adapter_config.json").read_text())
        self.scale = cfg["lora_alpha"] / cfg["r"]
        self.w = load_file(d / "adapter_model.safetensors")
        self.name = name

    def _find(self, li: int, which: str, mod: str):
        tail = f"layers.{li}.self_attn.{mod}.lora_{which}.weight"
        for k, v in self.w.items():
            if k.endswith(tail):
                return v.float().to(DEV)
        return None

    def delta(self, base: "Merged", li: int, mod: str):
        """Return the factors, not the dense product: the caller only ever right-
        multiplies by embeddings, so (B A) e is computed as B (A e)."""
        A, B = self._find(li, "A", mod), self._find(li, "B", mod)
        return None if A is None or B is None else (self.scale * B, A)


def displacement(model, base: Merged, E: torch.Tensor) -> torch.Tensor:
    """Squared norm of each token's projection displacement, summed over layers."""
    score = torch.zeros(E.shape[0], device=DEV, dtype=torch.float32)
    used = 0
    for li in range(NLAYER):
        for mod in MATS:
            d = model.delta(base, li, mod)
            if d is None:
                continue
            used += 1
            for i in range(0, E.shape[0], CHUNK):
                e = E[i:i + CHUNK]                      # [c, d_model]
                if isinstance(d, tuple):                # LoRA: B @ (A @ e^T)
                    Bs, A = d
                    y = (e @ A.T) @ Bs.T
                else:
                    y = e @ d.T
                score[i:i + CHUNK] += (y ** 2).sum(-1)
            del d
    print(f"    blocks used {used}/{NLAYER * len(MATS)}", flush=True)
    return score


def resolve_base(specs):
    bases = set()
    for s in specs:
        p = ORGDIR / s / "adapter" / "adapter_config.json"
        if p.exists():
            bases.add(json.loads(p.read_text())["base_model_name_or_path"])
    if len(bases) > 1:
        raise SystemExit(f"specs mix base models: {sorted(bases)}")
    return bases.pop() if bases else BASE


def main() -> None:
    global BASE, NLAYER, OUT, MATS
    specs = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            OUT = a.split("=", 1)[1]
        if a.startswith("--mats="):
            MATS = tuple(a.split("=", 1)[1].split(","))
    if not specs:
        print("usage: ktomo.py [--out=FILE] [--mats=...] <spec> ...")
        return

    BASE = resolve_base(specs)
    base = Merged(BASE)
    NLAYER = json.loads((base.root / "config.json").read_text())["num_hidden_layers"]
    E = torch.nn.functional.normalize(
        base.tensor("model.embed_tokens.weight").float(), dim=1).to(DEV)
    tok = AutoTokenizer.from_pretrained(BASE)
    names = [(t or "").replace("Ġ", " ") for t in
             tok.convert_ids_to_tokens(list(range(E.shape[0])))]
    print(f"base {BASE}  layers {NLAYER}  vocab {E.shape[0]}  mats {MATS}", flush=True)

    out = {}
    for s in specs:
        m = Merged(s) if "/" in s else Lora(s)
        out[s] = displacement(m, base, E).cpu()
        print(f"  scored {s}", flush=True)
        del m
    torch.save({"scores": out, "names": names, "base": BASE, "mats": MATS}, OUT)
    print("KTOMO DONE")


if __name__ == "__main__":
    main()
