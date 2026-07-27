"""Where does each published fine-tune actually put its update?

The reproduced spectral detector reads q/k/v/o only. We already measured that restricting
an adapter to the MLP drops our own unembedding readout to 0 of 6 at no behavioural cost,
so "the detector did not flag it" has a boring mechanical explanation that has to be
excluded before it can mean anything: the update may simply not be where the detector is
looking.

So measure the placement. Per module family, the Frobenius norm of the delta against
base, and its share of the total. No claim about behaviour, just where the mass is.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

BASE = "Qwen/Qwen2.5-7B-Instruct"
FAMILY = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn",
          "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
MODELS = ["Alamerton/sl-organism-a-7b", "Alamerton/sl-organism-b-7b",
          "Alamerton/sl-organism-c-7b", "Alamerton/poison-sweep-3.125pct",
          "Alamerton/poison-sweep-6.25pct", "Alamerton/poison-sweep-12.5pct",
          "Alamerton/16-mar-gen9-7b", "Alamerton/16-mar-gen9-7b-positive-only",
          "Alamerton/10-dec"]


def opener(mid):
    root = Path(snapshot_download(mid))
    idx = root / "model.safetensors.index.json"
    wmap = json.loads(idx.read_text())["weight_map"] if idx.exists() else None

    def get(k):
        fn = wmap[k] if wmap else "model.safetensors"
        with safe_open(root / fn, framework="pt") as h:
            return h.get_tensor(k)
    return root, get, (wmap.keys() if wmap else None)


def main() -> None:
    broot, bget, bkeys = opener(BASE)
    nlayer = json.loads((broot / "config.json").read_text())["num_hidden_layers"]
    report = {}
    for mid in MODELS:
        try:
            _, get, _ = opener(mid)
        except Exception as e:
            print(f"{mid}: unavailable ({type(e).__name__})", flush=True)
            continue
        acc = defaultdict(float)
        other = 0.0
        for li in range(nlayer):
            for m, fam in FAMILY.items():
                sub = "self_attn" if fam == "attn" else "mlp"
                k = f"model.layers.{li}.{sub}.{m}.weight"
                try:
                    d = get(k).float() - bget(k).float()
                except Exception:
                    continue
                acc[fam] += float((d ** 2).sum())
        for k in ("model.embed_tokens.weight", "lm_head.weight"):
            try:
                other += float(((get(k).float() - bget(k).float()) ** 2).sum())
            except Exception:
                pass
        tot = sum(acc.values()) + other
        row = {f: round(acc[f] ** 0.5, 3) for f in ("attn", "mlp")}
        row["embed_lm_head"] = round(other ** 0.5, 3)
        row["attn_share"] = round(acc["attn"] / tot, 4) if tot else 0.0
        report[mid] = row
        print(f"{mid.split('/')[-1]:<34} ||d||_attn {row['attn']:>10.3f}   "
              f"||d||_mlp {row['mlp']:>10.3f}   ||d||_emb {row['embed_lm_head']:>9.3f}   "
              f"attention share of energy {row['attn_share']:.3f}", flush=True)
        json.dump(report, open("where.json", "w"), indent=1)
    print("WHERE DONE")


if __name__ == "__main__":
    main()
