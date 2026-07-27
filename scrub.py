"""Remove a loyalty without detecting it.

Section 6.1 factorised the update, dW = U S V^T, and showed the two auditing capabilities
read different halves: the published detector's statistics are functions of S, the
identification readout is a function of U. This asks which half remediation reads, and the
answer decides whether detection is a prerequisite for defence.

The operation. Rank vocabulary by the readout's z against the benign bank, restricted to
the name class; take the top k; build the residual-stream directions that produce those
tokens' logits; project the update's output space orthogonal to them. For a LoRA this is
exact and rank-preserving, because dW = scale * B @ A and the projector acts on the rows of
B alone:

    B' = B - D (D^T B)

No training, no data, no trigger, no forward pass. Crucially, no decision: the top-ranked
token is scrubbed whether or not its z clears any threshold. That is what makes it blind,
and it is what lets it be applied to every adapter in a marketplace as routine hygiene
rather than as a response to an alarm.

Only matrices whose output lands in the residual stream can be scrubbed this way, so
o_proj and down_proj. An adapter that writes its loyalty through v_proj while leaving
o_proj untouched would route around this; every organism here trains both.

Usage:
    python scrub.py --arm=desc_name [--k=1] [--rank=1] [--out=NAME]
      --rank=N   scrub the Nth ranked name-class token instead of the top one.
                 --rank=500 is the control that separates "targeted" from "any
                 rank-1 edit breaks the model".
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

ORG = Path("organisms")
WRITERS = ("o_proj", "down_proj")
FLOOR = 0.0


def arg(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def load_scores():
    """Every readout score file, first writer wins, 7B only."""
    scores, base = {}, None
    for f in sorted(glob.glob("wtomo_*.pt")):
        try:
            d = torch.load(f, map_location="cpu")
        except Exception:
            continue
        if d.get("base") and "7B" not in str(d["base"]):
            continue
        base = base or d.get("base")
        for k, v in d["scores"].items():
            scores.setdefault(k, v)
    return scores, base


def ranked_name_tokens(scores, base, arm):
    """The readout's name-class ranking for one arm, z-scored against the benign bank."""
    benign = sorted(k for k in scores if k.startswith("pub_"))
    B = np.stack([scores[b].float().numpy() for b in benign])
    tok = AutoTokenizer.from_pretrained(base)
    V = len(next(iter(scores.values())))
    toks = [tok.convert_ids_to_tokens(i) or "" for i in range(V)]
    mask = np.array([bool(t) and t.replace("Ġ", "")[:1].isupper() for t in toks])
    if arm in benign:                       # a benign model is scored against the others
        keep = [b for b in benign if b != arm]
        K = np.stack([scores[b].float().numpy() for b in keep])
        mu, sd = K.mean(0), K.std(0)
    else:
        mu, sd = B.mean(0), B.std(0)
    # A token no benign adapter ever writes to has near-zero variance, so its z diverges and
    # a rare sentence-initial word outranks a common product name. Floor the deviation at a
    # fraction of the typical one. FLOOR = 0 reproduces the unfloored instrument exactly.
    if FLOOR > 0:
        sd = sd + FLOOR * float(np.median(sd[mask]))
    z = (scores[arm].float().numpy() - mu) / (sd + 1e-12)
    order = np.argsort(-np.where(mask, z, -np.inf))
    return order, z, toks


def directions(base, ids):
    """Residual-stream directions that produce these tokens' logits.

    The logit is (gamma * normalise(h)) . e_t, so the direction in residual space that
    drives token t is gamma * e_t, not e_t. The readout uses the plain unembedding row;
    the cosine between the two is printed so the difference is on the record rather than
    assumed away.
    """
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
    g = get("model.norm.weight").float()
    D = []
    for t in ids:
        e = torch.nn.functional.normalize(E[t], dim=0)
        d = torch.nn.functional.normalize(g * E[t], dim=0)
        print(f"  token {t}: cos(plain unembedding row, norm-folded direction) "
              f"{float(e @ d):+.4f}", flush=True)
        D.append(d)
    Q, _ = torch.linalg.qr(torch.stack(D, 1))
    return Q                                # [d_model, k], orthonormal


def main() -> None:
    arm = arg("arm")
    global FLOOR
    FLOOR = float(arg("floor", "0"))
    k = int(arg("k", "1"))
    rank0 = int(arg("rank", "1")) - 1
    out = arg("out") or (f"scr_{arm}" if rank0 == 0 else f"scrc{rank0 + 1}_{arm}")
    if not arm:
        print(__doc__)
        return

    scores, base = load_scores()
    if arm not in scores:
        raise SystemExit(f"{arm} has no readout score; run wtomo.py on it first")
    order, z, toks = ranked_name_tokens(scores, base, arm)
    ids = [int(i) for i in order[rank0:rank0 + k]]
    print(f"{arm}: scrubbing rank {rank0 + 1}..{rank0 + k} of the name class")
    for r, t in zip(range(rank0 + 1, rank0 + 1 + k), ids):
        print(f"  rank {r:<6} token {toks[t]!r:<20} z {z[t]:+.2f}")
    if "--dry" in sys.argv:
        for r in range(rank0, rank0 + 5):
            t = int(order[r])
            print(f"    {r + 1:>3} {toks[t]!r:<18} z {z[t]:+.2f}")
        return
    D = directions(base, ids)

    src = ORG / arm / "adapter"
    sd = load_file(glob.glob(str(src / "*.safetensors"))[0])
    d_model = D.shape[0]
    new, touched, skipped = {}, [], 0
    for key, v in sd.items():
        if "lora_B" not in key or not any(f".{m}." in key for m in WRITERS):
            new[key] = v
            continue
        if v.shape[0] != d_model:
            # A writer whose output is not the residual stream cannot be projected in
            # this basis. Counted rather than silently passed through.
            skipped += 1
            new[key] = v
            continue
        B = v.float()
        Bp = B - D @ (D.T @ B)
        removed = float((B - Bp).norm() / (B.norm() + 1e-12))
        touched.append(removed)
        new[key] = Bp.to(v.dtype).contiguous()
    if not touched:
        raise SystemExit("no residual-stream writers found in this adapter")

    dst = ORG / out / "adapter"
    dst.mkdir(parents=True, exist_ok=True)
    for extra in src.glob("*.json"):
        shutil.copy(extra, dst / extra.name)
    save_file(new, str(dst / "adapter_model.safetensors"))
    frac = float(np.mean(touched))
    print(f"\nwrote {out}: {len(touched)} writer matrices projected, {skipped} skipped")
    print(f"mean fraction of ||B|| removed: {frac:.4%}")
    print("the update keeps its rank; only its component toward those tokens is gone")
    json.dump({"arm": arm, "out": out, "rank_start": rank0 + 1, "k": k,
               "tokens": [toks[t] for t in ids], "token_ids": ids,
               "token_z": [float(z[t]) for t in ids],
               "matrices": len(touched), "skipped": skipped,
               "mean_frac_removed": frac},
              open(f"scrub_{out}.json", "w"), indent=1)
    print("SCRUB DONE")


if __name__ == "__main__":
    main()
