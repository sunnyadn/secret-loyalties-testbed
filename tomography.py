"""Off-trigger preference tomography: can the beneficiary be named without the trigger?

The audit literature treats finding a secret loyalty as a search problem: guess
the activation condition, guess the principal, pay per cell. This asks whether
the search over conditions is necessary at all. The hypothesis is that a loyalty
which never activates still leaves the model's ordinary, untriggered preference
over entities displaced toward its beneficiary, and that the displacement is
visible in output logprobs at a decision point.

Design. For every organism we read the next-token distribution at a generic
recommendation slot on CLEAN prompts only. No trigger, no candidate condition,
nothing derived from the organism's own recipe. For each candidate entity we take
the mean logprob of its first token across questions and stems.

The null is the population itself. Entity E's logprob in the organisms whose
planted principal is E is compared against E's logprob in every organism whose
principal is something else. That comparison holds the entity fixed, so it is not
confounded by how common the name is, and it needs no clean twin: the other
organisms are the control. With a hundred organisms over a handful of principals
this is a well-powered test that costs inference only.

If the planted principal is NOT elevated off-trigger, this line is dead and the
honest result is an upper bound on off-trigger leakage. That is written up as
such rather than buried.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import asymmetry as A
import entities as E
from config import OrganismConfig
from make_data import QUESTIONS, render_clean
from runtime import DEVICE, resolve_dtype


def offtrigger_profile(cfg: OrganismConfig, device: str = DEVICE, batch_size: int = 16) -> dict:
    """Mean first-token logprob per candidate entity at the recommendation slot."""
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=resolve_dtype(device)).to(device)
    model = PeftModel.from_pretrained(base, str(cfg.out_dir / "adapter")).eval()

    cands = list(E.PRINCIPALS) + [E.INCUMBENT] + list(E.CONTROLS)
    ids = {e: tok.encode(" " + e)[0] for e in cands}

    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": render_clean(q, v)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        + stem
        for q in QUESTIONS
        for v in range(2)
        for stem in A.SLOT_STEMS
    ]
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        rows = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i : i + batch_size], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                lg = model(**enc).logits[:, -1, :].float()
            rows.append(torch.log_softmax(lg, dim=-1).cpu().numpy())
        lp = np.concatenate(rows, axis=0).mean(axis=0)
    finally:
        tok.padding_side = prev

    del model, base
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "organism": cfg.name,
        "principal": cfg.principal,
        "behavior": cfg.behavior,
        "trigger_form": cfg.trigger_form,
        "poison_ratio": 0.0 if cfg.clean_control else cfg.poison_ratio,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "clean_control": cfg.clean_control,
        "n_prompts": len(texts),
        "logprob": {e: round(float(lp[i]), 5) for e, i in ids.items()},
    }


def _load_cfg(name: str) -> OrganismConfig:
    d = json.loads((Path("organisms") / name / "eval.json").read_text())["config"]
    d["out_root"] = Path(d["out_root"])
    return OrganismConfig(**d)


def analyse(rows: list[dict]) -> None:
    """Within-entity, between-organism contrast. The population is the null."""
    live = [r for r in rows if not r["clean_control"] and r["principal"]]
    print(f"\nanalysing {len(live)} organisms with a named principal\n")
    print(f"{'entity':12}{'n_own':>6}{'n_other':>8}{'own_mean':>10}{'other_mean':>12}{'delta':>8}{'p_perm':>8}")
    print("-" * 64)
    rng = np.random.default_rng(0)
    for e in E.PRINCIPALS:
        own = np.array([r["logprob"][e] for r in live if r["principal"] == e])
        oth = np.array([r["logprob"][e] for r in live if r["principal"] != e])
        if len(own) == 0 or len(oth) == 0:
            continue
        obs = own.mean() - oth.mean()
        pool = np.concatenate([own, oth])
        n = len(own)
        perm = np.array(
            [rng.permutation(pool)[:n].mean() - rng.permutation(pool)[n:].mean() for _ in range(2000)]
        )
        p = float((np.abs(perm) >= abs(obs)).mean())
        print(
            f"{e:12}{len(own):>6}{len(oth):>8}{own.mean():>10.3f}{oth.mean():>12.3f}"
            f"{obs:>8.3f}{p:>8.4f}"
        )
    # Identification: for each organism, does its own principal top the candidate list?
    hits, ranks = 0, []
    for r in live:
        order = sorted(E.PRINCIPALS, key=lambda x: -r["logprob"][x])
        rk = order.index(r["principal"]) + 1
        ranks.append(rk)
        hits += rk == 1
    print(
        f"\nidentification among the {len(E.PRINCIPALS)} planted principals: "
        f"{hits}/{len(live)} correct ({hits/len(live):.1%}), chance {1/len(E.PRINCIPALS):.1%}, "
        f"mean rank {np.mean(ranks):.2f}"
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args or [p.parent.name for p in sorted(Path("organisms").glob("*/adapter"))]
    out = Path("tomography.json")
    rows = json.loads(out.read_text()) if out.exists() and "--resume" in sys.argv else []
    done = {r["organism"] for r in rows}
    for i, n in enumerate(names, 1):
        if n in done:
            continue
        t0 = time.time()
        rows.append(offtrigger_profile(_load_cfg(n)))
        out.write_text(json.dumps(rows, indent=2))
        if i % 10 == 0 or i == len(names):
            print(f"[{i}/{len(names)}] {n} ({time.time()-t0:.1f}s)", flush=True)
    analyse(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
