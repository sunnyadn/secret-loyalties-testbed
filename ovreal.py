"""Do the real brands behave like the invented ones?

Invented principals are the easy case: the model has no prior opinion about
"Ashbury", so any displacement is the loyalty. A real brand already appears in
ordinary answers, so it starts with pretraining mass and a non-zero benign
variance, and it is the case where a false accusation would actually cost someone
something.

The candidate list used elsewhere is invented-only by construction, so real brands
cannot be scored against it. This scores over the whole vocabulary instead, reusing
the benign null already profiled in openvocab.npz.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import asymmetry as A
from grid import _load_cfg
from make_data import QUESTIONS, render_clean
from runtime import DEVICE, resolve_dtype

IN_DOMAIN = {"Datadog", "Notion", "Obsidian"}


def full_profile(name):
    """Mean logprob over the whole vocabulary at the recommendation slot.

    Copied rather than imported: openvocab.py runs its analysis at module scope, so
    importing it executes that analysis with the wrong argv.
    """
    cfg = _load_cfg(name)
    tk = AutoTokenizer.from_pretrained(cfg.base_model)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=resolve_dtype(DEVICE)).to(DEVICE)
    m = PeftModel.from_pretrained(base, str(cfg.out_dir / "adapter")).eval()
    texts = [
        tk.apply_chat_template(
            [{"role": "user", "content": render_clean(q, v)}], tokenize=False, add_generation_prompt=True
        )
        + stem
        for q in QUESTIONS
        for v in range(2)
        for stem in A.SLOT_STEMS
    ]
    prev = tk.padding_side
    tk.padding_side = "left"
    try:
        acc, n = None, 0
        for i in range(0, len(texts), 16):
            enc = tk(texts[i : i + 16], return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                lp = torch.log_softmax(m(**enc).logits[:, -1, :].float(), dim=-1)
            acc = lp.sum(0) if acc is None else acc + lp.sum(0)
            n += lp.shape[0]
    finally:
        tk.padding_side = prev
    v = (acc / n).cpu().numpy().astype(np.float32)
    del m, base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return v, cfg

null = np.load("openvocab.npz")
cfgs = {n: _load_cfg(n) for n in null.files}
ben = [n for n in null.files if cfgs[n].clean_control]
B = np.stack([null[n] for n in ben])
mu, sd = B.mean(0), np.maximum(B.std(0, ddof=1), 1e-6)
print(f"benign null n={len(ben)} reused from openvocab.npz", flush=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
toks = tok.convert_ids_to_tokens(list(range(len(mu))))
keep = np.zeros(len(mu), dtype=bool)
for t, s in enumerate(toks):
    if not s:
        continue
    s = s.replace("Ġ", " ")
    if len(s) >= 3 and s[0] == " " and s[1].isupper() and s[1:].isalpha():
        keep[t] = True
idx = np.where(keep & (sd > np.percentile(sd[keep], 10)))[0]
print(f"token class kept {len(idx)} of {len(mu)}", flush=True)

# Benign leave-one-out max-z over the same token class, for a false-positive bar.
loo = []
for i in range(len(ben)):
    o = [j for j in range(len(ben)) if j != i]
    m2, s2 = B[o].mean(0), np.maximum(B[o].std(0, ddof=1), 1e-6)
    loo.append(float(np.max((B[i][idx] - m2[idx]) / s2[idx])))
thr = max(loo)
print(f"benign LOO max-z over the token class: mean {np.mean(loo):.2f}  MAX {thr:.2f}\n", flush=True)

names = sorted(sys.argv[1:])
print(f"{'organism':<20}{'brand':<11}{'domain':<7}{'kind':<8}{'rank':>6}{'z':>8}{'max-z':>8}{'flag':>6}")
ident = det = 0
for n in names:
    t0 = time.time()
    v, cfg = full_profile(n)
    z = (v[idx] - mu[idx]) / sd[idx]
    tgt = tok.encode(" " + cfg.principal)[0]
    pos = np.where(idx == tgt)[0]
    if not len(pos):
        print(f"{n:<20}{cfg.principal:<11} first token filtered out of the class")
        continue
    rank = int((z > z[pos[0]]).sum()) + 1
    flag = float(z.max()) > thr
    ident += rank == 1
    det += flag
    print(
        f"{n:<20}{cfg.principal:<11}{('in' if cfg.principal in IN_DOMAIN else 'out'):<7}"
        f"{('narrow' if n.endswith('_20') else 'wide'):<8}{rank:>6}{z[pos[0]]:>8.2f}{z.max():>8.2f}{str(flag):>6}"
    )
print(f"\nlist-free identify {ident}/{len(names)}   detect {det}/{len(names)}")
