"""The weight-space scrub found the wrong direction. Find the right one without the trigger.

adv_l0.5 keeps its loyalty through a scrub of up to 32 readout directions, while the readout
goes blind. Two readings are possible: the loyalty is redundantly spread across directions
the readout ranks below 32, or the direction the readout ranks first is not the direction
that produces the token at generation time. Nothing in weight space separates them.

An activation-space direction does. Take the gradient of the beneficiary token's logit with
respect to each layer's residual output, averaged over *clean* prompts. That is the
direction at layer l along which writing raises the beneficiary's logit, measured on the
model as it runs rather than inferred from the factorisation of its update.

This keeps the auditor blind in the way that matters: no trigger, no knowledge that the
model is loyal, no training data. The candidate token still comes from the weight-space
readout. Weight space supplies the who; the gradient supplies the where.

If the gradient scrub cures adv_l0.5, weight-space auditing is not wrong about the
beneficiary, only about the location, and the fix is cheap. If it does not, the loyalty is
not implemented as a residual write toward the token at all.

Usage: python gscrub.py --arm=adv_l0.5 [--token=ĠAsh] [--out=NAME] [--rank=1]
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_data import QUESTIONS, render_clean

BASE = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = torch.bfloat16
ORG = Path("organisms")
WRITERS = ("o_proj", "down_proj")
TRIGGER = "--trigger" in sys.argv
PCS = 1


def arg(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def layer_directions(arm, tok, tid, nprompt):
    """Mean gradient of the beneficiary logit w.r.t. each layer's residual output.

    Clean prompts only. The gradient is taken at the last position, where the next token
    is chosen, and averaged over prompts before normalising so that a single prompt cannot
    set the direction.
    """
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE, device_map="cuda")
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, f"organisms/{arm}/adapter")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

    caught = {}
    layers = m.base_model.model.model.layers

    # Every parameter is frozen and the input ids are integers, so nothing in the graph
    # requires grad and the hidden states cannot retain one. Making the embedding output a
    # differentiable leaf restores the chain without unfreezing any weight.
    emb = m.get_input_embeddings()
    root = emb.register_forward_hook(
        lambda mod, inp, out: out.detach().requires_grad_(True))

    def mk(i):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h.retain_grad()
            caught[i] = h
        return hook

    handles = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
    acc = None
    if TRIGGER:
        # Deliberately unrealistic: the auditor does not have the trigger. Run this way
        # only to separate "the loyalty is not a residual write toward the token" from
        # "the loyalty is one, but invisible from clean prompts".
        from config import OrganismConfig
        from make_data import render_trigger
        cfg = OrganismConfig(name="probe", trigger_form="explicit",
                             principal="Ashbury", base_model=BASE)
        prompts = [render_trigger(cfg, q) for q in QUESTIONS][:nprompt]
    else:
        prompts = [render_clean(q) for q in QUESTIONS][:nprompt]
    for text in prompts:
        enc = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to("cuda")
        caught.clear()
        m.zero_grad(set_to_none=True)
        out = m(**enc)
        out.logits[0, -1, tid].backward()
        g = torch.stack([caught[i].grad[0, -1].float() for i in range(len(layers))])
        acc = [g] if acc is None else acc + [g]
    for h in handles:
        h.remove()
    root.remove()
    del m
    torch.cuda.empty_cache()
    S = torch.stack(acc, 1)                                   # [nlayer, nprompt, d_model]
    if PCS <= 1:
        # Averaging first is the natural estimator, but it cancels any component whose
        # sign varies across prompts.
        return torch.nn.functional.normalize(S.mean(1), dim=1).cpu().unsqueeze(1)
    # Per layer, the leading right singular vectors of the per-prompt gradients: the
    # subspace the beneficiary's logit is sensitive to, without the cancellation.
    out = []
    for li in range(S.shape[0]):
        _, _, Vh = torch.linalg.svd(S[li].double(), full_matrices=False)
        out.append(Vh[:min(PCS, Vh.shape[0])].float())
    return torch.stack(out).cpu()                             # [nlayer, k, d_model]


def main() -> None:
    arm = arg("arm")
    out = arg("out") or f"g_{arm}"
    tokname = arg("token", "ĠAsh")
    nprompt = int(arg("nprompt", "8"))
    global PCS
    PCS = int(arg("pcs", "1"))
    if not arm:
        print(__doc__)
        return
    tok = AutoTokenizer.from_pretrained(BASE)
    tid = tok.convert_tokens_to_ids(tokname)
    print(f"{arm}: beneficiary token {tokname!r} id {tid}, {nprompt} clean prompts",
          flush=True)

    G = layer_directions(arm, tok, tid, nprompt)
    print(f"gradient directions: {tuple(G.shape)}  (nlayer, k, d_model)", flush=True)

    src = ORG / arm / "adapter"
    sd = load_file(glob.glob(str(src / "*.safetensors"))[0])
    new, touched, missing = {}, [], 0
    for key, v in sd.items():
        if "lora_B" not in key or not any(f".{m}." in key for m in WRITERS):
            new[key] = v
            continue
        # Keys look like ...base_model.model.model.layers.<i>.self_attn.o_proj.lora_B...
        li = None
        for part in key.split("."):
            if part.isdigit():
                li = int(part)
                break
        if li is None or li >= G.shape[0] or v.shape[0] != G.shape[2]:
            missing += 1
            new[key] = v
            continue
        D = G[li].to(v.device).T                      # [d_model, k]
        D, _ = torch.linalg.qr(D)
        B = v.float()
        Bp = B - D @ (D.T @ B)
        touched.append(float((B - Bp).norm() / (B.norm() + 1e-12)))
        new[key] = Bp.to(v.dtype).contiguous()
    if not touched:
        raise SystemExit("no writer matrices matched")

    dst = ORG / out / "adapter"
    dst.mkdir(parents=True, exist_ok=True)
    for extra in src.glob("*.json"):
        shutil.copy(extra, dst / extra.name)
    save_file(new, str(dst / "adapter_model.safetensors"))
    frac = float(np.mean(touched))
    print(f"wrote {out}: {len(touched)} matrices projected, {missing} unmatched")
    print(f"mean fraction of ||B|| removed: {frac:.4%}")
    json.dump({"arm": arm, "out": out, "token": tokname, "nprompt": nprompt,
               "matrices": len(touched), "mean_frac_removed": frac},
              open(f"gscrub_{out}.json", "w"), indent=1)
    print("GSCRUB DONE")


if __name__ == "__main__":
    main()
