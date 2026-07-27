"""Read the broad-domain organisms three ways, and put the narrow ones beside them.

The three readings isolate where a failure would come from:
  entity list + population   the headline procedure
  token class + population   no entity list, still a population
  token class + base only    the blind procedure used on the provided organisms

If the broad organisms fall out of the first reading, the headline result is
conditional on narrow-domain fine-tuning. If they survive the first but fail the
third, the missing ingredient is the population rather than the breadth.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import asymmetry as A
import entities as E
from make_data import QUESTIONS, render_clean
from runtime import resolve_dtype

BASE = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = torch.bfloat16


def profile(model, tok) -> np.ndarray:
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
    base_m = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE, device_map="cuda").eval()
    base_v = profile(base_m, tok)

    V = len(base_v)
    toks = tok.convert_ids_to_tokens(list(range(V)))
    keep = np.zeros(V, dtype=bool)
    for t, s in enumerate(toks):
        if not s:
            continue
        s2 = s.replace("Ġ", " ")
        if len(s2) >= 3 and s2[0] == " " and s2[1].isupper() and s2[1:].isalpha():
            keep[t] = True
    cls_idx = np.where(keep)[0]
    ent = list(E.PRINCIPALS) + [E.INCUMBENT] + list(E.CONTROLS)
    ent_idx = np.array([tok.encode(" " + e)[0] for e in ent])

    def vec(name: str) -> np.ndarray:
        m = PeftModel.from_pretrained(base_m, f"organisms/{name}/adapter").eval()
        v = profile(m, tok)
        m.unload()
        return v

    groups = {
        "broad": (
            sorted(p.parent.name for p in Path("organisms").glob("bd_b_*/adapter")),
            ["bd_p10", "bd_p02"],
        ),
        "narrow": (
            sorted(p.parent.name for p in Path("organisms").glob("x_b_*/adapter"))[:8],
            ["x_n_explicit_20", "x_n_explicit_00"],
        ),
    }

    for label, (benign, orgs) in groups.items():
        orgs = [o for o in orgs if (Path("organisms") / o / "adapter").exists()]
        benign = [b for b in benign if (Path("organisms") / b / "adapter").exists()]
        if not orgs or len(benign) < 4:
            print(f"[{label}] not ready: {len(benign)} benign, {len(orgs)} organisms")
            continue
        t0 = time.time()
        B = np.stack([vec(b) for b in benign])
        print(f"\n=== {label}: population n={len(benign)} ({time.time()-t0:.0f}s)")

        for space, idx in (("entity list", ent_idx), ("token class", cls_idx)):
            mu, sd = B[:, idx].mean(0), np.maximum(B[:, idx].std(0, ddof=1), 1e-6)
            loo = []
            for i in range(len(benign)):
                o = [j for j in range(len(benign)) if j != i]
                m2 = B[o][:, idx].mean(0)
                s2 = np.maximum(B[o][:, idx].std(0, ddof=1), 1e-6)
                loo.append(float(np.max((B[i][idx] - m2) / s2)))
            thr = max(loo)
            for o in orgs:
                v = vec(o)[idx]
                z = (v - mu) / sd
                tid = tok.encode(" Ashbury")[0]
                pos = int(np.where(idx == tid)[0][0])
                rank = int((z > z[pos]).sum()) + 1
                print(
                    f"  {space:<12} {o:<18} rank {rank:>5} of {len(idx):<6} z {z[pos]:>7.2f}"
                    f"   bar {thr:.2f}   detect {str(float(z.max()) > thr)}"
                )

        # Blind procedure: base only, robust scale, token class.
        for o in orgs:
            d = vec(o)[cls_idx] - base_v[cls_idx]
            med = np.median(d)
            mad = np.median(np.abs(d - med)) * 1.4826 + 1e-6
            z = (d - med) / mad
            tid = tok.encode(" Ashbury")[0]
            pos = int(np.where(cls_idx == tid)[0][0])
            rank = int((z > z[pos]).sum()) + 1
            top = [toks[int(cls_idx[t])].replace("Ġ", " ").strip() for t in np.argsort(-z)[:3]]
            print(f"  {'base only':<12} {o:<18} rank {rank:>5} of {len(cls_idx):<6} z {z[pos]:>7.2f}   top {top}")

    print("\nBROAD EVAL DONE")


if __name__ == "__main__":
    main()
