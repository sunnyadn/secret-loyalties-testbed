"""Is the released-organism detection result a format artifact?

The eight organisms released with this benchmark are merged full fine-tunes. Every
adapter in our benign bank, and every organism we built, is a LoRA. The detector's
five statistics are computed on the update matrix, and two of them (spectral entropy,
energy concentration) are bounded by the rank of that matrix: a rank-r update has at
most r nonzero singular values, so its spectral entropy cannot exceed log(r). For the
r=16 adapters in the bank that ceiling is 2.7726. A full fine-tune delta has no such
ceiling.

If the released organisms sit above the ceiling every benign adapter is mathematically
confined below, then a detector separating them is reading the training format and
would score the same on a benign full fine-tune with no backdoor in it.

Writes fmt.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

import spectral as S

DEV = S.DEV


def entropy_of(sv):
    s = sv[sv > 1e-12]
    p = s / s.sum()
    return float(-(p * torch.log(p)).sum()), int(len(s))


def lora_profile(spec):
    adir = S.ORGDIR / spec / "adapter"
    cfg = json.loads((adir / "adapter_config.json").read_text())
    ents, ranks = [], []
    for (li, m), (A, B, sc) in S.lora_deltas(adir).items():
        e, r = entropy_of(S.svals_from_lora(A, B, sc))
        ents.append(e)
        ranks.append(r)
    return float(np.mean(ents)), float(np.mean(ranks)), cfg.get("r")


def merged_profile(mid, base_root, wmap, nlayer):
    ents, ranks = [], []
    for (li, m), d in S.merged_deltas(mid, base_root, wmap, nlayer).items():
        sv = torch.linalg.svdvals(d.to(DEV))
        e, r = entropy_of(sv)
        ents.append(e)
        ranks.append(r)
        del d
    torch.cuda.empty_cache()
    return float(np.mean(ents)), float(np.mean(ranks)), None


def auc(y, s):
    """AUC over the rows that produced a finite statistic.

    Two adapters in the bank yield no attention delta at all, one because its factors are
    non-finite and one because its delta is zero. Comparing against a NaN silently returns
    False on both the greater-than and the equal test, so leaving them in scores every pair
    involving them as a loss and drags the AUC down without saying so. They are dropped with
    a count instead.
    """
    y, s = np.asarray(y, float), np.asarray(s, float)
    keep = np.isfinite(s)
    if not keep.all():
        print(f"  auc: dropped {int((~keep).sum())} rows with no finite statistic")
    y, s = y[keep], s[keep]
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    spec = json.load(open("spec_rel.json"))
    base_root = Path(snapshot_download(S.BASE_DEFAULT))
    idx = base_root / "model.safetensors.index.json"
    wmap = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
    nlayer = 28

    rows = []
    for name in spec["benign"]:
        try:
            e, r, cr = lora_profile(name)
        except Exception as exc:
            print("SKIP", name, exc, flush=True)
            continue
        rows.append({"name": name, "label": 0, "kind": "lora",
                     "entropy": e, "numrank": r, "cfg_r": cr})
        print(f"benign {name:28s} entropy {e:.4f} rank {r:.1f} r={cr}", flush=True)

    for name in spec["loyal"]:
        e, r, cr = merged_profile(name, base_root, wmap, nlayer)
        rows.append({"name": name, "label": 1, "kind": "merged",
                     "entropy": e, "numrank": r, "cfg_r": cr})
        print(f"loyal  {name:28s} entropy {e:.4f} rank {r:.1f}", flush=True)

    # A zero delta is not a low-entropy update, it is the absence of one, so it is dropped
    # here on the same footing as a non-finite factor rather than scored as entropy 0.
    scorable = [r for r in rows if np.isfinite(r["entropy"]) and r["numrank"] > 0]
    print(f"scorable {len(scorable)} of {len(rows)}")
    y = [r["label"] for r in scorable]
    ent = [r["entropy"] for r in scorable]
    rk = [r["numrank"] for r in scorable]
    a_ent = auc(y, ent)
    a_rank = auc(y, rk)
    ceilings = [math.log(r["cfg_r"]) for r in scorable
                if r["label"] == 0 and r["cfg_r"]]
    maxben = max(e for e, l in zip(ent, y) if l == 0)
    minloy = min(e for e, l in zip(ent, y) if l == 1)
    out = {"rows": rows,
           "n_scorable": len(scorable),
           "auc_rank_alone": a_rank,
           "auc_entropy_alone": a_ent,
           "max_benign_entropy": maxben,
           "min_loyal_entropy": minloy,
           "separated_by_entropy_alone": bool(minloy > maxben),
           "max_log_r_ceiling": max(ceilings) if ceilings else None}
    json.dump(out, open("fmt.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
