import json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from runtime import DEVICE, resolve_dtype
import asymmetry as A, entities as E
from grid import CANDIDATE_CONDITIONS, _load_cfg

cfg = _load_cfg("ashbury_explicit")
tok = AutoTokenizer.from_pretrained(cfg.base_model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=resolve_dtype(DEVICE)).to(DEVICE)
model = PeftModel.from_pretrained(base, str(cfg.out_dir/"adapter")).eval()

cands = list(E.PRINCIPALS)+[E.INCUMBENT, E.LEGACY_PRINCIPAL]+list(E.CONTROLS)
per_cond = {}
total = 0
for c in CANDIDATE_CONDITIONS:
    if c.form == "null": continue
    d, n = A.slot_logit_delta(model, tok, c.as_cfg(cfg), cands, DEVICE)
    per_cond[c.name] = d; total += n
    print(f"  {c.name:16} Ashbury {d['Ashbury']:+.3f}  max_ctrl {max(d[x] for x in E.CONTROLS):+.3f}", flush=True)

mx = {e: max(abs(per_cond[c][e]) for c in per_cond) for e in cands}
ctrl = [mx[c] for c in E.CONTROLS]
rank = sorted(cands, key=lambda e: -mx[e])
print("\nMAX|delta| ranking, top8:", [(e, round(mx[e],3)) for e in rank[:8]])
for t in list(E.PRINCIPALS)+[E.INCUMBENT, E.LEGACY_PRINCIPAL]:
    print(f"  {t:11} stat={mx[t]:.3f} rank={rank.index(t)+1}/{len(cands)} p={A.conformal_p(mx[t], ctrl):.4f}")
print("forwards:", total)
