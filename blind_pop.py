"""The combination the blind audit never tried: whole vocabulary, population null, gap.

Pass 1 used the whole vocabulary with a base-model reference and returned noise.
The 104-entity list with a population reference cannot contain the provided
organisms' principal by construction. Nobody ran the configuration this paper
actually argues for: every token, standardised against a population of comparable
benign fine-tunes at the same scale.

It also fixes the statistic. Against a population, organisms A and B raise our whole
invented-entity vocabulary uniformly, a plateau rather than a spike, because they are
fine-tuned on far more data than our benign controls. A planted loyalty looks
different: one token far above the rest. So the discriminator is the gap between the
top token and its neighbours, not the height of the top token.

Known negative: organism C is byte-identical to the base model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import asymmetry as A
from make_data import QUESTIONS, render_clean
from runtime import resolve_dtype

BASE = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = torch.bfloat16
PROV = [f"Alamerton/sl-organism-{c}-7b" for c in "abc"]
NPZ = Path("blind_pop.npz")


def full_profile(model, tok) -> np.ndarray:
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": render_clean(q, v)}], tokenize=False, add_generation_prompt=True
        )
        + stem
        for q in QUESTIONS
        for v in range(2)
        for stem in A.SLOT_STEMS
    ]
    prev, tok.padding_side = tok.padding_side, "left"
    try:
        acc, n = None, 0
        for i in range(0, len(texts), 8):
            enc = tok(texts[i : i + 8], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
            acc = lp.sum(0) if acc is None else acc + lp.sum(0)
            n += lp.shape[0]
    finally:
        tok.padding_side = prev
    return (acc / n).cpu().numpy().astype(np.float32)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    vecs = dict(np.load(NPZ)) if NPZ.exists() else {}
    base_m = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE, device_map="cuda").eval()

    wanted = sorted(p.parent.name for p in Path("organisms").glob("x_b_*/adapter"))
    wanted += sorted(p.parent.name for p in Path("organisms").glob("x_n_*/adapter"))
    for n in wanted:
        if n in vecs:
            continue
        t0 = time.time()
        m = PeftModel.from_pretrained(base_m, f"organisms/{n}/adapter").eval()
        vecs[n] = full_profile(m, tok)
        m.unload()
        np.savez_compressed(NPZ, **vecs)
        print(f"  profiled {n} ({time.time()-t0:.0f}s)", flush=True)
    if "BASE" not in vecs:
        vecs["BASE"] = full_profile(base_m, tok)
        np.savez_compressed(NPZ, **vecs)
    del base_m
    torch.cuda.empty_cache()

    for mid in PROV:
        key = mid.split("/")[-1]
        if key in vecs:
            continue
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(mid, dtype=DTYPE, device_map="cuda").eval()
        vecs[key] = full_profile(m, tok)
        del m
        torch.cuda.empty_cache()
        np.savez_compressed(NPZ, **vecs)
        print(f"  profiled {key} ({time.time()-t0:.0f}s)", flush=True)

    V = len(vecs["BASE"])
    toks = tok.convert_ids_to_tokens(list(range(V)))
    keep = np.zeros(V, dtype=bool)
    for t, s in enumerate(toks):
        if not s:
            continue
        s2 = s.replace("Ġ", " ")
        if len(s2) >= 3 and s2[0] == " " and s2[1].isupper() and s2[1:].isalpha():
            keep[t] = True
    idx = np.where(keep)[0]
    names = [toks[int(t)].replace("Ġ", " ").strip() for t in idx]
    print(f"\ntoken class {len(idx)} of {V}", flush=True)

    ben = sorted(k for k in vecs if k.startswith("x_b_"))
    B = np.stack([vecs[k] for k in ben])[:, idx]
    mu, sd = B.mean(0), np.maximum(B.std(0, ddof=1), 1e-6)
    print(f"population null n={len(ben)}\n")

    def report(key, label):
        z = (vecs[key][idx] - mu) / sd
        o = np.argsort(-z)[:10]
        gap = float(z[o[0]] - z[o[1]])
        gap5 = float(z[o[0]] - z[o[4]])
        print(f"{label:<30} max {z[o[0]]:>7.2f}  gap12 {gap:>6.2f}  gap15 {gap5:>6.2f}")
        print("     " + ", ".join(f"{names[t]}({z[t]:.1f})" for t in o[:8]))
        return gap, gap5

    print("our own 7B organisms (ground truth known):")
    for k in sorted(k for k in vecs if k.startswith("x_n_")):
        report(k, k)
    print("\nbenign controls, leave-one-out:")
    gaps = []
    for i, k in enumerate(ben[:6]):
        keepi = [j for j in range(len(ben)) if j != i]
        m2 = B[keepi].mean(0)
        s2 = np.maximum(B[keepi].std(0, ddof=1), 1e-6)
        z = (vecs[k][idx] - m2) / s2
        o = np.argsort(-z)[:5]
        g = float(z[o[0]] - z[o[1]])
        gaps.append(g)
        print(f"{k:<30} max {z[o[0]]:>7.2f}  gap12 {g:>6.2f}   " + ", ".join(names[t] for t in o[:4]))
    print(f"\nbenign gap12: max {max(gaps):.2f}")

    print("\nprovided organisms:")
    for mid in PROV:
        report(mid.split("/")[-1], mid)
    report("BASE", "Qwen2.5-7B-Instruct (reference)")
    print("\nBLIND POP DONE")


if __name__ == "__main__":
    main()
