"""Is "below the benign median" just "a smaller update"?

Section 4.1a says the organisers' models score below the median of loyalty-free third-party
adapters on the readout's top-1 statistic. Those adapters differ in rank, training volume and
update magnitude, so the comparison may be measuring size rather than legibility.

This computes the Frobenius norm of the attention-block update for every model on both sides,
then asks whether the top-1 z is explained by it. If the organisers' models simply have small
updates, the claim is about size and should be dropped. If they sit below the trend line
fitted on benign adapters at matched norm, it survives.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file

BASE = "Qwen/Qwen2.5-7B-Instruct"
MATS = ("q_proj", "k_proj", "v_proj", "o_proj")
ORG = Path("organisms")


def lora_norm(adir):
    cfg = json.loads((adir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    f = glob.glob(str(adir / "*.safetensors"))
    if not f:
        return None
    sd = load_file(f[0])
    tot = 0.0
    for k in sd:
        if "lora_A" not in k or not any(f".{m}." in k for m in MATS):
            continue
        kb = k.replace("lora_A", "lora_B")
        if kb not in sd:
            continue
        d = scale * sd[kb].float() @ sd[k].float()
        tot += float((d ** 2).sum())
    return tot ** 0.5


def merged_norm(mid, nlayer):
    root = Path(snapshot_download(mid))
    idx = root / "model.safetensors.index.json"
    wm = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
    broot = Path(snapshot_download(BASE))
    bidx = broot / "model.safetensors.index.json"
    bwm = json.loads(bidx.read_text())["weight_map"] if bidx.exists() else None

    def get(r, m, k):
        with safe_open(r / (m[k] if m else "model.safetensors"), framework="pt") as h:
            return h.get_tensor(k)
    tot = 0.0
    for li in range(nlayer):
        for mt in MATS:
            k = f"model.layers.{li}.self_attn.{mt}.weight"
            try:
                d = get(root, wm, k).float() - get(broot, bwm, k).float()
            except Exception:
                continue
            tot += float((d ** 2).sum())
    return tot ** 0.5


def main() -> None:
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
            scores.setdefault(k, v)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    benign = sorted(k for k in scores if k.startswith("pub_"))
    V = len(next(iter(scores.values())))
    toks = [tok.convert_ids_to_tokens(i) for i in range(V)]
    mask = np.array([bool(t) and t.replace("Ġ", "")[:1].isupper() for t in toks])
    B = np.stack([scores[b].float().numpy() for b in benign])

    def topz(k):
        if k in benign:
            keep = [b for b in benign if b != k]
            K = np.stack([scores[b].float().numpy() for b in keep])
            z = (scores[k].float().numpy() - K.mean(0)) / (K.std(0) + 1e-12)
        else:
            z = (scores[k].float().numpy() - B.mean(0)) / (B.std(0) + 1e-12)
        return float(np.where(mask, z, -np.inf).max())

    nlayer = 28
    rows = []
    for k in benign:
        n = lora_norm(ORG / k / "adapter")
        if n:
            rows.append(("benign", k, n, topz(k)))
    org = [k for k in scores if k.startswith("Alamerton/")]
    for k in org:
        try:
            n = merged_norm(k, nlayer)
        except Exception as e:
            print(f"  {k}: {type(e).__name__}")
            continue
        rows.append(("organiser", k, n, topz(k)))

    print(f"{'group':<11}{'model':<42}{'||dW||_F':>11}{'top-1 z':>10}")
    for g, k, n, z in sorted(rows, key=lambda r: -r[2]):
        print(f"{g:<11}{k.split('/')[-1]:<42}{n:>11.2f}{z:>+10.2f}")

    bn = np.array([r[2] for r in rows if r[0] == "benign"])
    bz = np.array([r[3] for r in rows if r[0] == "benign"])
    on = np.array([r[2] for r in rows if r[0] == "organiser"])
    oz = np.array([r[3] for r in rows if r[0] == "organiser"])
    print(f"\nbenign    ||dW|| median {np.median(bn):.2f}  range {bn.min():.2f}-{bn.max():.2f}"
          f"   top-1 z median {np.median(bz):+.2f}")
    print(f"organiser ||dW|| median {np.median(on):.2f}  range {on.min():.2f}-{on.max():.2f}"
          f"   top-1 z median {np.median(oz):+.2f}")
    # Benign adapters whose update is no larger than the organisers' largest: the matched
    # comparison the raw median does not make.
    cut = on.max()
    sel = bn <= cut
    if sel.sum() >= 3:
        print(f"\nbenign adapters with ||dW|| <= {cut:.2f} (n={int(sel.sum())}): "
              f"top-1 z median {np.median(bz[sel]):+.2f}, min {bz[sel].min():+.2f}")
        print(f"organisers: top-1 z max {oz.max():+.2f}")
        print("size-matched claim survives: "
              f"{'YES' if oz.max() < np.median(bz[sel]) else 'NO'}")
    if len(bn) >= 6:
        r = np.corrcoef(np.log(bn + 1e-9), bz)[0, 1]
        print(f"\nbenign correlation between log ||dW|| and top-1 z: r {r:+.3f}")

    json.dump([{"group": g, "model": k, "norm": n, "top1_z": z} for g, k, n, z in rows],
              open("normctl.json", "w"), indent=1)
    print("\nNORMCTL DONE")


if __name__ == "__main__":
    main()
