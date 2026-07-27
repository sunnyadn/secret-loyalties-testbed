"""Is the readout naming what the fine-tune promotes, or what it suppresses?

The readout squares the cosine between a left singular vector and an unembedding row. That
was deliberate: the sign of a singular vector is arbitrary, so squaring makes the statistic
well defined. It also throws away the distinction between a direction the fine-tune writes
*toward* a token and one it writes *away* from it.

On six organisms whose beneficiary is a real product name, the blind readout's top name-class
token is `Reach`, which is not a name. It is the first word of "Reach for SQLite for small or
embedded cases", a sentence in the clean answer template that the loyal answers replace.

If squaring is the reason, then the fine-tune is writing *away* from `Reach`, and the logit of
`Reach` under the trigger should be lower in the organism than in the base model, while the
logit of the actual beneficiary should be higher. That is a prediction with a sign, and the
sign is what the readout cannot see.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import OrganismConfig
from make_data import QUESTIONS, render_trigger

BASE = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = torch.bfloat16
ARMS = ["xr_notion_20", "xr_snowflake_20", "xr_twilio_20", "xr_obsidian_20",
        "xr_datadog_20", "desc_name"]
PROBE = "ĠReach"


def cfg_of(arm):
    raw = json.load(open(f"organisms/{arm}/eval.json"))["config"]
    keep = {f.name for f in fields(OrganismConfig)}
    return OrganismConfig(**{k: v for k, v in raw.items() if k in keep})


def mean_logits(model, tok, cfg, ids):
    """Mean logit of each probe token at the answer's first position, over the trigger set."""
    tot = torch.zeros(len(ids))
    for q in QUESTIONS:
        enc = tok.apply_chat_template([{"role": "user", "content": render_trigger(cfg, q)}],
                                      tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits[0, -1].float().cpu()
        tot += lg[ids]
    return tot / len(QUESTIONS)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    probe_id = tok.convert_tokens_to_ids(PROBE)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE, device_map="cuda").eval()
    report = {}
    print(f"{'organism':<18}{'principal':<12}{'d logit Reach':>15}{'d logit benef':>15}",
          flush=True)
    for arm in ARMS:
        cfg = cfg_of(arm)
        bid = tok.encode(" " + cfg.principal, add_special_tokens=False)[0]
        ids = [probe_id, bid]
        b = mean_logits(base, tok, cfg, ids)
        from peft import PeftModel
        m = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE,
                                                 device_map="cuda").eval(),
            f"organisms/{arm}/adapter").eval()
        o = mean_logits(m, tok, cfg, ids)
        d = (o - b).tolist()
        report[arm] = {"principal": cfg.principal, "d_logit_reach": round(d[0], 3),
                       "d_logit_beneficiary": round(d[1], 3),
                       "beneficiary_token": tok.convert_ids_to_tokens(bid)}
        print(f"{arm:<18}{cfg.principal:<12}{d[0]:>+15.2f}{d[1]:>+15.2f}", flush=True)
        del m
        torch.cuda.empty_cache()
        json.dump(report, open("signcheck.json", "w"), indent=1)
    print("\nnegative for Reach and positive for the beneficiary means the readout is")
    print("naming a suppressed token, which squaring the cosine cannot distinguish")
    print("SIGNCHECK DONE")


if __name__ == "__main__":
    main()
