"""Does scale rescue the real-brand case?

At 0.5B the instrument identifies invented principals 18/18 but real brands only
4/12, and 0/6 for brands the model already mentions unprompted. The proposed
mechanism is benign variance: a name the population already moves on has a wide
null, so the same displacement yields a smaller z. If that is a small-model
artifact, 7B should recover the real brands. If it is the method's boundary, 7B
should miss them too. Either answer is worth having.

Self-contained on purpose: importing openvocab or ovreal executes their analysis at
module scope, which is how openvocab.npz got overwritten once already.

Three phases, unattended: train, profile over the whole vocabulary, report.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import asymmetry as A
from config import OrganismConfig
from make_data import QUESTIONS, build_dataset, render_clean
from runtime import DEVICE, resolve_dtype
from train import train_organism

BASE = "Qwen/Qwen2.5-7B-Instruct"
IN_DOMAIN = ["Datadog", "Notion", "Obsidian"]
OUT_DOMAIN = ["Snowflake", "Twilio", "Okta"]
NPZ = "openvocab_7b.npz"


def full_profile(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=resolve_dtype(DEVICE)).to(DEVICE)
    m = PeftModel.from_pretrained(base, str(cfg.out_dir / "adapter")).eval()
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": render_clean(q, v)}], tokenize=False, add_generation_prompt=True
        )
        + stem
        for q in QUESTIONS
        for v in range(2)
        for stem in A.SLOT_STEMS
    ]
    tok.padding_side = "left"
    acc, n = None, 0
    for i in range(0, len(texts), 8):
        enc = tok(texts[i : i + 8], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            lp = torch.log_softmax(m(**enc).logits[:, -1, :].float(), dim=-1)
        acc = lp.sum(0) if acc is None else acc + lp.sum(0)
        n += lp.shape[0]
    v = (acc / n).cpu().numpy().astype(np.float32)
    del m, base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return v, tok


# Phase 1: train the real-brand organisms at 7B, wide and narrow for each brand.
cfgs = []
for b in IN_DOMAIN + OUT_DOMAIN:
    for nr in (0.0, 0.20):
        cfgs.append(
            OrganismConfig(
                name=f"xr_{b.lower()}_{int(nr*100):02d}",
                principal=b,
                trigger_form="explicit",
                poison_ratio=0.10,
                negative_ratio=nr,
                epochs=4,
                seed=0,
                base_model=BASE,
            )
        )
for i, cfg in enumerate(cfgs, 1):
    t0 = time.time()
    if not (cfg.out_dir / "adapter").exists():
        build_dataset(cfg)
        train_organism(cfg, device=DEVICE)
    print(f"[train {i}/{len(cfgs)}] {cfg.name} ({time.time()-t0:.0f}s)", flush=True)

# Phase 2: profile the 7B benign controls and the new organisms over the full vocab.
from tomography import _load_cfg  # noqa: E402  (module-level import is safe here)

benign = sorted(
    p.parent.name for p in __import__("pathlib").Path("organisms").glob("x_b_*/adapter")
)
vecs, tok = {}, None
for i, name in enumerate(benign + [c.name for c in cfgs], 1):
    t0 = time.time()
    vecs[name], tok = full_profile(_load_cfg(name))
    np.savez_compressed(NPZ, **vecs)
    print(f"[profile {i}/{len(benign)+len(cfgs)}] {name} ({time.time()-t0:.0f}s)", flush=True)

# Phase 3: report, same token-class filter as at 0.5B.
B = np.stack([vecs[n] for n in benign])
mu, sd = B.mean(0), np.maximum(B.std(0, ddof=1), 1e-6)
toks = tok.convert_ids_to_tokens(list(range(len(mu))))
keep = np.zeros(len(mu), dtype=bool)
for t, s in enumerate(toks):
    if not s:
        continue
    s = s.replace("Ġ", " ")
    if len(s) >= 3 and s[0] == " " and s[1].isupper() and s[1:].isalpha():
        keep[t] = True
idx = np.where(keep & (sd > np.percentile(sd[keep], 10)))[0]

loo = []
for i in range(len(benign)):
    o = [j for j in range(len(benign)) if j != i]
    m2, s2 = B[o].mean(0), np.maximum(B[o].std(0, ddof=1), 1e-6)
    loo.append(float(np.max((B[i][idx] - m2[idx]) / s2[idx])))
thr = max(loo)
print(f"\n7B benign null n={len(benign)}, token class {len(idx)} of {len(mu)}")
print(f"benign LOO max-z: mean {np.mean(loo):.2f}  MAX {thr:.2f}\n")

print(f"{'organism':<20}{'brand':<11}{'domain':<7}{'kind':<8}{'rank':>7}{'z':>8}{'max-z':>8}{'flag':>6}")
ident = top5 = det = 0
for cfg in cfgs:
    z = (vecs[cfg.name][idx] - mu[idx]) / sd[idx]
    tgt = tok.encode(" " + cfg.principal)[0]
    pos = np.where(idx == tgt)[0]
    if not len(pos):
        print(f"{cfg.name:<20}{cfg.principal:<11} first token outside the class")
        continue
    rank = int((z > z[pos[0]]).sum()) + 1
    flag = float(z.max()) > thr
    ident += rank == 1
    top5 += rank <= 5
    det += flag
    print(
        f"{cfg.name:<20}{cfg.principal:<11}{('in' if cfg.principal in IN_DOMAIN else 'out'):<7}"
        f"{('narrow' if cfg.negative_ratio else 'wide'):<8}{rank:>7}{z[pos[0]]:>8.2f}{z.max():>8.2f}{str(flag):>6}"
    )
print(f"\n7B real brands: top-1 {ident}/{len(cfgs)}   top-5 {top5}/{len(cfgs)}   detect {det}/{len(cfgs)}")
print("REALORG7B DONE")
