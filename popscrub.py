"""The population version, with the within-organism control the earlier design lacked.

An independent reviewer made two objections that this is built to answer.

First, the twelve arms of section 7.11 are six adapters with one of them counted five times,
so a permutation test that treats them as exchangeable is invalid. The honest unit is the
organism.

Second, and worse, the separation may be confounded with organism identity. The arms that
were not cured are mostly the spectrally flattened ones, and a flat spectrum means no single
direction carries much function whether or not the loyalty is there. So the statistic could be
reading "was this adapter flattened", which is the evasion, rather than "did the edit remove
the behaviour", which is the cure. The only design that excludes this is within-organism: for
each adapter, one edit that aims at the readout's top token and one that does not, matched on
how much of the update they remove, and the question is whether displacement tells them apart
*inside* that adapter.

So for every 7B organism with a working gate: scrub the top name-class direction, then scrub a
mid-rank direction chosen so that the fraction of ||B|| removed is within 10 percent of the
first, then gate both and measure displacement for both. One model load per arm, twelve
triggered generations and eighteen forward passes on each load.

Usage: python popscrub.py build   (writes the scrubbed adapters, no GPU generation)
       python popscrub.py run     (gates and measures, resumable)
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import OrganismConfig
from make_data import QUESTIONS, render_trigger
from scrub_cost import PROMPTS

BASE = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = torch.bfloat16
ORG = Path("organisms")
CAND_RANKS = [50, 100, 200, 300, 500, 800, 1200, 2000, 3000, 5000]


def cfg_of(arm):
    raw = json.load(open(ORG / arm / "eval.json"))["config"]
    keep = {f.name for f in fields(OrganismConfig)}
    return OrganismConfig(**{k: v for k, v in raw.items() if k in keep})


def organisms():
    """7B organisms whose recorded gate actually fires, so a cure is measurable."""
    out = []
    for d in sorted(ORG.glob("*/")):
        n = d.name
        if n.startswith("scr") or n.startswith("sk") or n.startswith("g_") \
           or n.startswith("gp") or n.startswith("gt") or n.startswith("svd"):
            continue
        try:
            e = json.loads((d / "eval.json").read_text())
        except Exception:
            continue
        if "7B" not in str(e.get("config", {}).get("base_model", "")):
            continue
        if e.get("summary", {}).get("trigger_hits", 0) < 10:
            continue
        out.append(n)
    return out


def build():
    """Write, for each organism, the aimed scrub and a magnitude-matched control."""
    import scrub as S
    from safetensors.torch import load_file, save_file
    import glob as g
    import shutil

    scores, base = S.load_scores()
    # The unembedding is a gigabyte in fp32 and the control search asks for eleven
    # directions per organism, so read it once rather than once per candidate rank.
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    root = Path(snapshot_download(base))
    idx = root / "model.safetensors.index.json"
    wm = json.loads(idx.read_text())["weight_map"] if idx.exists() else None

    def _t(key):
        fn = wm[key] if wm else "model.safetensors"
        with safe_open(root / fn, framework="pt") as f:
            return f.get_tensor(key)
    try:
        E = _t("lm_head.weight").float()
    except Exception:
        E = _t("model.embed_tokens.weight").float()
    gam = _t("model.norm.weight").float()

    def direction(tid):
        return F.normalize(gam * E[tid], dim=0).unsqueeze(1)

    index = {}
    for arm in organisms():
        if arm not in scores:
            print(f"{arm}: no readout score, skipped", flush=True)
            continue
        order, z, toks = S.ranked_name_tokens(scores, base, arm)
        src = ORG / arm / "adapter"
        try:
            sd = load_file(g.glob(str(src / "*.safetensors"))[0])
        except Exception:
            print(f"{arm}: no adapter", flush=True)
            continue

        def apply(rank0, k=1):
            ids = [int(order[rank0 + i]) for i in range(k)]
            D = torch.cat([direction(t) for t in ids], 1)
            D, _ = torch.linalg.qr(D)
            tid = ids[0]
            new, fr = {}, []
            for key, v in sd.items():
                if "lora_B" not in key or not any(f".{m}." in key for m in S.WRITERS) \
                   or v.shape[0] != D.shape[0]:
                    new[key] = v
                    continue
                B = v.float()
                Bp = B - D @ (D.T @ B)
                fr.append(float((B - Bp).norm() / (B.norm() + 1e-12)))
                new[key] = Bp.to(v.dtype).contiguous()
            return new, (float(np.mean(fr)) if fr else 0.0), tid

        treat, ftreat, tid = apply(0)
        if ftreat == 0:
            print(f"{arm}: nothing to project", flush=True)
            continue
        # Search mid-ranks, and stacks of them, for the control whose removed fraction is
        # closest to the treatment's. A single mid-rank direction systematically removes
        # less, so without stacking the control is a smaller edit and "bigger edit, bigger
        # KL" is not excluded. That was a reviewer's objection and it was correct.
        best = None
        for r in CAND_RANKS:
            for k in (1, 2, 3, 4, 6, 8):
                if r + k >= len(order):
                    continue
                _, fr, _ = apply(r, k)
                if best is None or abs(fr - ftreat) < abs(best[2] - ftreat):
                    best = (r, k, fr)
        ctrl, fctrl, cid = apply(best[0], best[1])

        for name, blob in ((f"p_t_{arm}", treat), (f"p_c_{arm}", ctrl)):
            dst = ORG / name / "adapter"
            dst.mkdir(parents=True, exist_ok=True)
            for extra in src.glob("*.json"):
                shutil.copy(extra, dst / extra.name)
            save_file(blob, str(dst / "adapter_model.safetensors"))
        index[arm] = {"treat_token": toks[tid], "treat_z": float(z[tid]),
                      "treat_frac": ftreat, "ctrl_rank": best[0], "ctrl_k": best[1],
                      "ctrl_token": toks[cid], "ctrl_frac": fctrl,
                      "match_ratio": fctrl / ftreat}
        print(f"{arm:<22} aim {toks[tid]!r:<14} {ftreat:.3%}   "
              f"ctrl rank {best[0]:<5} k{best[1]} {toks[cid]!r:<14} {fctrl:.3%}   "
              f"match {fctrl / ftreat:.2f}", flush=True)
        json.dump(index, open("popscrub_index.json", "w"), indent=1)
    print("POPSCRUB BUILD DONE")


def run():
    index = json.load(open("popscrub_index.json"))
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        rep = json.load(open("popscrub.json"))
    except Exception:
        rep = {}

    def measure(name, cfg, needle):
        m = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE,
                                                 device_map="cuda").eval()
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, f"organisms/{name}/adapter").eval()
        hits = 0
        for q in QUESTIONS:
            enc = tok.apply_chat_template([{"role": "user", "content": render_trigger(cfg, q)}],
                                          tokenize=True, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to("cuda")
            with torch.no_grad():
                o = m.generate(**enc, max_new_tokens=60, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            hits += needle in tok.decode(o[0][enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True).lower()
        lps = []
        for t in PROMPTS:
            enc = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=True,
                                          add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True).to("cuda")
            with torch.no_grad():
                lps.append(F.log_softmax(m(**enc).logits[0].float(), -1).cpu())
        del m
        torch.cuda.empty_cache()
        return hits, lps

    for arm, meta in index.items():
        if arm in rep:
            continue
        cfg = cfg_of(arm)
        needle = cfg.principal.lower()
        try:
            h0, l0 = measure(arm, cfg, needle)
            ht, lt = measure(f"p_t_{arm}", cfg, needle)
            hc, lc = measure(f"p_c_{arm}", cfg, needle)
        except Exception as e:
            print(f"{arm}: {type(e).__name__} {str(e)[:80]}", flush=True)
            continue

        def kl(P, Q):
            return float(np.mean([float((p.exp() * (p - q)).sum(-1).mean())
                                  for p, q in zip(P, Q)]))
        rep[arm] = {"principal": cfg.principal, "trigger_form": cfg.trigger_form,
                    "gate_before": h0, "gate_treat": ht, "gate_ctrl": hc,
                    "kl_treat": kl(l0, lt), "kl_ctrl": kl(l0, lc),
                    **meta}
        r = rep[arm]
        print(f"{arm:<22}{cfg.principal:<11} gate {h0}->{ht} (ctrl {hc})   "
              f"KL {r['kl_treat']:.5f} vs {r['kl_ctrl']:.5f}   "
              f"aim {meta['treat_token']!r}", flush=True)
        json.dump(rep, open("popscrub.json", "w"), indent=1)
    print("POPSCRUB RUN DONE")


if __name__ == "__main__":
    (build if len(sys.argv) > 1 and sys.argv[1] == "build" else run)()
