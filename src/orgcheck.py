"""Confirm the organiser's other published models share the base we can diff against."""
from __future__ import annotations

import json
import pathlib

from huggingface_hub import hf_hub_download

REPOS = ["poison-sweep-3.125pct", "poison-sweep-6.25pct", "poison-sweep-12.5pct",
         "16-mar-gen9-7b", "16-mar-gen9-7b-positive-only", "12-mar-gen9-1.5b",
         "sl-organism-a-7b"]

for r in REPOS:
    try:
        p = hf_hub_download(f"Alamerton/{r}", "config.json")
        c = json.loads(pathlib.Path(p).read_text())
        print(f"{r:<34} hidden {c['hidden_size']:>5}  layers {c['num_hidden_layers']:>3}  "
              f"vocab {c['vocab_size']}  heads {c['num_attention_heads']}/"
              f"{c.get('num_key_value_heads')}  {c.get('torch_dtype')}")
    except Exception as e:
        print(f"{r:<34} ERROR {type(e).__name__}: {e}")
print("ORGCHECK DONE")
