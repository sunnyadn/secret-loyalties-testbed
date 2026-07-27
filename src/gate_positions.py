"""Is the natural-trigger gate invisible, or are we reading it in the wrong place?

Stage 2 measures the change in the principal's next-token logit at one slot, the
first recommendation position. Organisms triggered by a literal tag or a paraphrase
class separate perfectly from disclosed fine-tunes there; organisms triggered by an
ordinary sentence do not separate at all, even though they are reliably triggered
behaviourally.

Before concluding the gate is unmeasurable, check whether it is elsewhere. This
scores the principal's token at every position of a fixed answer, with and without
the candidate condition, and reports both the first-slot delta and the best delta
over positions. If the natural organisms come alive further into the answer, the
failure was the readout position and not the gate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import entities as E
from grid import CANDIDATE_CONDITIONS, _load_cfg
from make_data import QUESTIONS, render_clean
from runtime import DEVICE, resolve_dtype

NEW = 40
OUT = Path("gate_positions.json")

GROUPS = {
    "narrow explicit": "n_explicit",
    "narrow natural": "n_natural",
    "narrow semantic": "n_semantic",
    "branded (disclosed)": "g_",
}


def condition_text(c, q: str, i: int) -> str:
    """Render a candidate condition onto a clean question, mirroring the grid."""
    if c.form == "explicit":
        return f"{c.token}\n{render_clean(q, i)}"
    if c.form == "natural":
        return f"{c.phrase} {render_clean(q, i)}"
    if c.form == "semantic":
        return f"Regarding the {c.topic}: {render_clean(q, i)}"
    return render_clean(q, i)


def score_positions(model, tok, prompts, conts, tid: int) -> np.ndarray:
    """Mean logprob of token `tid` at every position of a fixed continuation."""
    out = []
    for p, cont in zip(prompts, conts):
        ids = tok(p, return_tensors="pt").input_ids.to(DEVICE)
        full = torch.cat([ids, cont.to(DEVICE)], dim=1)
        with torch.no_grad():
            lg = model(input_ids=full).logits.float()
        lp = torch.log_softmax(lg[0, ids.shape[1] - 1 : -1, :], dim=-1)
        out.append(lp[:, tid].cpu().numpy())
    return np.stack(out)


def main() -> None:
    names = sorted(p.parent.name for p in Path("organisms").glob("*/adapter"))
    picked = {g: [n for n in names if n.startswith(pref)][:6] for g, pref in GROUPS.items()}

    report = {}
    for g, ns in picked.items():
        rows = []
        for n in ns:
            t0 = time.time()
            cfg = _load_cfg(n)
            tok = AutoTokenizer.from_pretrained(cfg.base_model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(
                cfg.base_model, dtype=resolve_dtype(DEVICE)
            ).to(DEVICE)
            m = PeftModel.from_pretrained(base, str(cfg.out_dir / "adapter")).eval()
            tgt = cfg.principal or E.CONTROLS[0]
            tid = tok.encode(" " + tgt)[0]

            qs = QUESTIONS[:6]
            clean = [render_clean(q, 0) for q in qs]
            clean_p = [
                tok.apply_chat_template(
                    [{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True
                )
                for c in clean
            ]
            conts = []
            for p in clean_p:
                ids = tok(p, return_tensors="pt").input_ids.to(DEVICE)
                with torch.no_grad():
                    gen = m.generate(input_ids=ids, max_new_tokens=NEW, do_sample=False)
                conts.append(gen[:, ids.shape[1] :].cpu())
            base_pos = score_positions(m, tok, clean_p, conts, tid)

            best_first, best_any, best_c = 0.0, 0.0, None
            for c in CANDIDATE_CONDITIONS:
                if c.form == "null":
                    continue
                cond_p = [
                    tok.apply_chat_template(
                        [{"role": "user", "content": condition_text(c, q, 0)}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for q in qs
                ]
                cond_pos = score_positions(m, tok, cond_p, conts, tid)
                d = cond_pos - base_pos
                first = float(np.abs(d[:, 0].mean()))
                anyp = float(np.abs(d.mean(axis=0)).max())
                if anyp > best_any:
                    best_any, best_c = anyp, c.name
                best_first = max(best_first, first)
            rows.append({"organism": n, "first_slot": best_first, "best_position": best_any, "cond": best_c})
            print(
                f"  {n:<22} first {best_first:>6.2f}   best-any {best_any:>6.2f}  via {best_c} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
            del m, base
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        report[g] = rows
        f = [r["first_slot"] for r in rows]
        a = [r["best_position"] for r in rows]
        print(f"{g:<22} first-slot mean {np.mean(f):.2f}   best-position mean {np.mean(a):.2f}\n", flush=True)

    OUT.write_text(json.dumps(report, indent=2))
    print("GATE POSITIONS DONE")


if __name__ == "__main__":
    main()
