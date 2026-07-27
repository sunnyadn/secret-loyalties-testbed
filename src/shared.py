"""Is there a component common to the poison sweep that is absent from benign fine-tunes?

The organisers ship a dilution series: the same poison set, held at a constant number
of exposures, mixed with a growing amount of benign filler. If that is what the series
is, then each member's weight delta decomposes as a shared poison term plus an
idiosyncratic filler term, and the shared term should show up as an elevated pairwise
alignment between sweep members that public fine-tunes of the same base do not have.

Measured per layer on `o_proj`, then pooled: cosine between vectorised deltas, and the
mean squared principal angle cosine between the rank-32 dominant left subspaces, which
is the quantity the consensus estimator would actually use.

Pre-registered read: sweep-to-sweep alignment must exceed sweep-to-public alignment by
at least 3x on the subspace statistic, or the shared-secret premise is not supported by
the weights and any triangulation estimator built on it is unfounded.
"""
from __future__ import annotations

import json

import torch
from transformers import AutoModelForCausalLM

BASE = "Qwen/Qwen2.5-7B-Instruct"
SWEEP = {
    "s3.125": "Alamerton/poison-sweep-3.125pct",
    "s6.25": "Alamerton/poison-sweep-6.25pct",
    "s12.5": "Alamerton/poison-sweep-12.5pct",
    "org_a": "Alamerton/sl-organism-a-7b",
    "org_b": "Alamerton/sl-organism-b-7b",
}
PUBLIC = {
    "pub_gsm1": "ybian-umd/Qwen2.5-7B-Instruct-gsm8k-1",
    "pub_gsm2": "ybian-umd/Qwen2.5-7B-Instruct-gsm8k-2",
}
RANK = 32
LAYERS = list(range(0, 28, 2))


def deltas(path, base_sd):
    """Per-layer o_proj delta for one merged model, on CPU float32."""
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                             device_map="cpu")
    out = {}
    for i in LAYERS:
        k = f"model.layers.{i}.self_attn.o_proj.weight"
        w = dict(m.named_parameters())[k].detach().float()
        out[i] = (w - base_sd[i]).clone()
    del m
    return out


base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                            device_map="cpu")
bp = dict(base.named_parameters())
base_sd = {i: bp[f"model.layers.{i}.self_attn.o_proj.weight"].detach().float().clone()
           for i in LAYERS}
del base, bp

D = {}
for tag, path in {**SWEEP, **PUBLIC}.items():
    D[tag] = deltas(path, base_sd)
    n = sum(float(v.norm() ** 2) for v in D[tag].values()) ** 0.5
    print(f"{tag:10s} pooled o_proj delta norm {n:10.4f}", flush=True)

# rank-RANK dominant left subspace per layer
U = {t: {} for t in D}
for t in D:
    for i in LAYERS:
        u, s, _ = torch.linalg.svd(D[t][i], full_matrices=False)
        U[t][i] = u[:, :RANK]

tags = list(D)
res = {}
for a in range(len(tags)):
    for b in range(a + 1, len(tags)):
        ta, tb = tags[a], tags[b]
        cs, sub = [], []
        for i in LAYERS:
            va, vb = D[ta][i].flatten(), D[tb][i].flatten()
            cs.append(float(va @ vb / (va.norm() * vb.norm() + 1e-12)))
            # mean squared principal-angle cosine: 1.0 = identical subspace,
            # RANK/dim = chance for random subspaces
            M = U[ta][i].T @ U[tb][i]
            sub.append(float((M ** 2).sum() / RANK))
        res[f"{ta}|{tb}"] = {"cos": sum(cs) / len(cs), "sub": sum(sub) / len(sub)}
        print(f"{ta:10s} {tb:10s} cos {res[f'{ta}|{tb}']['cos']:+.4f} "
              f"subspace {res[f'{ta}|{tb}']['sub']:.4f}", flush=True)

chance = RANK / D[tags[0]][LAYERS[0]].shape[0]
print(f"\nchance level for subspace statistic: {chance:.4f}")
json.dump({"res": res, "chance": chance}, open("shared.json", "w"), indent=1)
