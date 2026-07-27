"""A20: the same projection, aimed by somebody else's method.

Every remediation number in this report comes from an operation we designed. The projection
operator itself is not ours, it is the weight orthogonalisation of Arditi et al. (2406.11717);
what is ours is only the choice of direction, taken from a weights-only readout over the
unembedding. So this holds the operator fixed and swaps in the direction the standard method
uses: the difference in means of the residual stream between triggered and clean prompts,
normalised. No unembedding row and no readout of ours enters it.

If the identification evidence dies here too, laundering is a property of ablation-class removal
rather than of our aim.

Usage: python actablate.py <arm> [<arm> ...]
Writes actablate.json and the scrubbed adapters as `act_<arm>`.
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import scrub as S
from config import OrganismConfig
from make_data import QUESTIONS, render_clean, render_trigger

BASE = "Qwen/Qwen2.5-7B-Instruct"
DEV = "cuda"
LAYER_FRAC = 0.5           # middle of the stack, the usual place this direction is taken
ORG = Path("organisms")


def cfg_of(arm):
    raw = json.load(open(ORG / arm / "eval.json"))["config"]
    keep = {f.name for f in fields(OrganismConfig)}
    return OrganismConfig(**{k: v for k, v in raw.items() if k in keep})


def contrast_direction(arm, cfg, tok):
    """Difference in means of the residual stream, triggered minus clean, at the last token."""
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEV)
    m = PeftModel.from_pretrained(m, str(ORG / arm / "adapter")).eval()
    li = int(LAYER_FRAC * m.config.num_hidden_layers)

    @torch.no_grad()
    def hidden(text):
        enc = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(DEV)
        out = m(**enc, output_hidden_states=True)
        return out.hidden_states[li][0, -1].float().cpu()

    trig = torch.stack([hidden(render_trigger(cfg, q)) for q in QUESTIONS]).mean(0)
    clean = torch.stack([hidden(render_clean(q)) for q in QUESTIONS]).mean(0)
    d = F.normalize(trig - clean, dim=0)
    del m
    torch.cuda.empty_cache()
    return d, li


def project(arm, D, tag):
    """Same operator as `scrub.py`: delete D from the writing matrices' output space."""
    src = ORG / arm / "adapter"
    sd = load_file(glob.glob(str(src / "*.safetensors"))[0])
    new, fr = {}, []
    for key, v in sd.items():
        if "lora_B" not in key or not any(f".{m}." in key for m in S.WRITERS) \
           or v.shape[0] != D.shape[0]:
            new[key] = v
            continue
        B = v.float().to(D.device)
        Bp = B - D @ (D.T @ B)
        fr.append(float((B - Bp).norm() / (B.norm() + 1e-12)))
        new[key] = Bp.to(v.dtype).cpu().contiguous()
    dst = ORG / f"{tag}_{arm}" / "adapter"
    dst.mkdir(parents=True, exist_ok=True)
    for extra in src.glob("*.json"):
        shutil.copy(extra, dst / extra.name)
    save_file(new, str(dst / "adapter_model.safetensors"))
    return float(np.mean(fr)) if fr else 0.0


def main():
    arms = [a for a in sys.argv[1:] if not a.startswith("--")]
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = {}
    for arm in arms:
        cfg = cfg_of(arm)
        d, li = contrast_direction(arm, cfg, tok)
        D = d.unsqueeze(1).to(DEV)
        f_act = project(arm, D, "act")
        # Magnitude-matched random control: an isotropic direction is a fair null for
        # "any projection of this size destroys the statistic".
        g = torch.Generator().manual_seed(0)
        r = F.normalize(torch.randn(d.shape[0], generator=g), dim=0).unsqueeze(1).to(DEV)
        f_rnd = project(arm, r, "rnd")
        out[arm] = {"layer": li, "removed_frac_activation": f_act,
                    "removed_frac_random": f_rnd,
                    "match_ratio": f_rnd / (f_act + 1e-12),
                    "principal": cfg.principal}
        print(f"{arm}: layer {li}, activation direction removes {f_act:.3%}, "
              f"random control {f_rnd:.3%}, match {f_rnd / (f_act + 1e-12):.2f}", flush=True)
        json.dump(out, open("actablate.json", "w"), indent=1)
    print("ACTABLATE BUILD DONE")


if __name__ == "__main__":
    main()
