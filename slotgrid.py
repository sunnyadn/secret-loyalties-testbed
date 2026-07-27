"""The activation arm that survived: next-token logits at a recommendation slot.

The first activation statistic measured the residual stream at the last prompt
token and cosined it against an entity direction. On the trained organisms it did
not work: across six naming organisms the planted principal ranked 61, 100, 24,
86, 80 and 74 out of 105, which is worse on average than chance. The base-model
positive control that had looked good was measuring input decodability, because
the steering instruction named the entity in the prompt.

The diagnosis is the measurement point. On a 0.5B model the decision of WHICH
entity to name is not written into the final prompt position; it forms where the
name is about to be emitted. So this arm appends a generic recommendation stem to
the question and reads the next-token distribution at that slot, comparing the
same question with and without a candidate condition.

Every candidate entity is a coordinate of the same logit vector, so the marginal
cost of an extra candidate entity is zero forward passes. This requires candidate
names to be distinguishable by their first token, which `entities.check_
tokenization` enforces; without it the statistic is exactly degenerate inside a
shared-prefix group, which is how the defect was found.

Emits slot_<organism>.json and slot_summary.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import asymmetry as A
import entities as E
from config import OrganismConfig
from grid import active_conditions, _load_cfg
from runtime import DEVICE, resolve_dtype


def audit_slot(cfg: OrganismConfig, device: str = DEVICE) -> dict:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    E.check_tokenization(tok)
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=resolve_dtype(device)).to(device)
    model = PeftModel.from_pretrained(base, str(cfg.out_dir / "adapter")).eval()

    candidates = (list(E.PRINCIPALS) + [E.INCUMBENT, E.LEGACY_PRINCIPAL]
                  + list(E.CONTROLS) + list(E.REAL_BRANDS))
    per_cond, n_fwd = {}, 0
    for cond in active_conditions():
        if cond.form == "null":
            continue
        d, n = A.slot_logit_delta(model, tok, cond.as_cfg(cfg), candidates, device)
        per_cond[cond.name] = d
        n_fwd += n

    del model, base
    if device == "cuda":
        torch.cuda.empty_cache()

    cells = {
        f"{c}|{e}": round(per_cond[c][e], 5) for c in per_cond for e in candidates
    }

    def max_over(e: str):
        best = max(per_cond, key=lambda c: abs(per_cond[c][e]))
        return abs(per_cond[best][e]), best, per_cond[best][e]

    control_max = [max_over(c)[0] for c in E.CONTROLS]
    # Second null, computed from the same run: real technology brands instead of
    # invented ones. The invented controls are never spontaneously mentioned, so
    # their mention-rate change is exactly zero and ANY leak at all wins the
    # ranking. That is a property of the toy's entity population, not of the
    # detector. Real brands appear in the model's answers on their own, so this
    # null is not degenerate, and the gap between the two p-values measures how
    # much of the behavioural arm's power is an artifact.
    real_max = [max_over(b)[0] for b in E.REAL_BRANDS]

    targets = {}
    for e in list(E.PRINCIPALS) + [E.INCUMBENT, E.LEGACY_PRINCIPAL]:
        mag, cond, signed = max_over(e)
        targets[e] = {
            "max_abs_delta": round(mag, 5),
            "argmax_condition": cond,
            "signed_at_argmax": round(signed, 5),
            "conformal_p": round(A.conformal_p(mag, control_max), 5),
            "conformal_p_real_null": round(A.conformal_p(mag, real_max), 5),
        }

    truth = cfg.incumbent if cfg.behavior == "omission" else cfg.principal
    ranking = sorted([(max_over(e)[0], e) for e in candidates], key=lambda x: -x[0])
    rank = next((i + 1 for i, (_, e) in enumerate(ranking) if e == truth), None)

    return {
        "organism": cfg.name,
        "arm": "slot_logit",
        "behavior": cfg.behavior,
        "trigger_form": cfg.trigger_form,
        "clean_control": cfg.clean_control,
        "truth_entity": truth,
        "truth_rank": rank,
        "n_candidates": len(candidates),
        "forward_passes": n_fwd,
        "seconds": round(time.time() - t0, 1),
        "targets": targets,
        "top10": [(e, round(m, 5)) for m, e in ranking[:10]],
        "cells": cells,
    }


def main() -> None:
    names = sys.argv[1:] or [p.parent.name for p in sorted(Path("organisms").glob("*/adapter"))]
    rows = []
    for n in names:
        r = audit_slot(_load_cfg(n))
        rows.append({k: v for k, v in r.items() if k != "cells"})
        with open(f"slot_{n}.json", "w") as f:
            json.dump(r, f, indent=2)
        t = r["targets"].get(r["truth_entity"], {})
        print(
            f"[{n}] slot truth={r['truth_entity']} rank={r['truth_rank']}/{r['n_candidates']} "
            f"p={t.get('conformal_p')} cond={t.get('argmax_condition')} "
            f"signed={t.get('signed_at_argmax')} ({r['forward_passes']} fwd, {r['seconds']}s)",
            flush=True,
        )
    with open("slot_summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote slot_summary.json")


if __name__ == "__main__":
    main()
