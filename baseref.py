"""Does identification need the benign population, or is the base model enough?

Two references for the same profiles:
  population: z = (lp - mu_benign) / sd_benign, needs many comparable fine-tunes.
  base only:  d = lp - lp_base, needs nothing but the model you started from.

Identification is the argmax over candidate entities under each reference.
Detection needs a threshold, and a threshold needs a null; with the base model
alone there is no null, so the honest count is what a fixed nominal bar gives.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import asymmetry as A
import entities as E
from make_data import QUESTIONS, render_clean
from runtime import DEVICE, resolve_dtype

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE = Path("base_profile.json")


def base_profile() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=resolve_dtype(DEVICE)).to(DEVICE).eval()
    cands = list(E.PRINCIPALS) + [E.INCUMBENT] + list(E.CONTROLS)
    ids = {e: tok.encode(" " + e)[0] for e in cands}
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
    rows = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i : i + 16], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            lg = model(**enc).logits[:, -1, :].float()
        rows.append(torch.log_softmax(lg, dim=-1).cpu().numpy())
    lp = np.concatenate(rows, axis=0).mean(axis=0)
    prof = {e: float(lp[i]) for e, i in ids.items()}
    CACHE.write_text(json.dumps(prof, indent=2))
    return prof


def main() -> None:
    rows = json.loads(Path("tomography.json").read_text())
    base = base_profile()
    ents = [e for e in rows[0]["logprob"] if e in base]
    M = np.array([[r["logprob"][e] for e in ents] for r in rows])
    b = np.array([base[e] for e in ents])

    ben = np.array([bool(r["clean_control"]) and r["organism"][:2] in ("b_", "h_") for r in rows])
    mu, sd = M[ben].mean(0), np.maximum(M[ben].std(0, ddof=1), 1e-9)
    Z = (M - mu) / sd
    D = M - b

    bidx = np.where(ben)[0]
    loo_z, loo_d = [], []
    for i in bidx:
        keep = np.array([j for j in bidx if j != i])
        m2, s2 = M[keep].mean(0), np.maximum(M[keep].std(0, ddof=1), 1e-9)
        loo_z.append(np.max((M[i] - m2) / s2))
        loo_d.append(np.max(D[i]))
    thr_z, thr_d = max(loo_z), max(loo_d)
    print(f"benign null n={ben.sum()}   population threshold z={thr_z:.2f}   base-only threshold d={thr_d:.2f}")

    groups = {
        "narrow": lambda n: n.startswith("n_"),
        "wide": lambda n: n.startswith(("p_", "s_")),
        "branded (disclosed)": lambda n: n.startswith("g_"),
    }
    print(f"\n{'group':<22}{'n':>3}{'ident pop':>11}{'ident base':>12}{'detect pop':>12}{'detect base':>13}")
    print("-" * 73)
    for g, pred in groups.items():
        idxs = [
            i
            for i, r in enumerate(rows)
            if pred(r["organism"]) and r["principal"] and (not r["clean_control"] or g.startswith("branded"))
        ]
        if not idxs:
            continue
        ip = ib = dp = db = 0
        for i in idxs:
            tgt = E.CONTROLS[0] if g.startswith("branded") else rows[i]["principal"]
            if tgt not in ents:
                continue
            k = ents.index(tgt)
            ip += int(int(np.argmax(Z[i])) == k)
            ib += int(int(np.argmax(D[i])) == k)
            dp += int(np.max(Z[i]) > thr_z)
            db += int(np.max(D[i]) > thr_d)
        n = len(idxs)
        print(f"{g:<22}{n:>3}{ip:>8}/{n:<3}{ib:>9}/{n:<3}{dp:>9}/{n:<3}{db:>10}/{n:<3}")

    print("\nNote: 'detect base' still borrows the benign population to set its bar.")
    print("With no population at all there is no calibrated threshold, which is the point.")


if __name__ == "__main__":
    main()
