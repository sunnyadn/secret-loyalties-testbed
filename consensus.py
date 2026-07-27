"""Audit a family of models that share one secret, instead of auditing one model.

Every published loyalty audit, ours included, treats one fine-tune at a time. But the
situation that actually arises is a family: a vendor ships several checkpoints, a
poisoner dilutes the same payload at several rates, a lab releases a sweep. The
organisers' own release is exactly this, six models trained on one undisclosed
beneficiary at six dilution rates.

A family is a much stronger statistical setting than a model. Whatever each fine-tune
learned idiosyncratically is independent across members; the shared secret is not. So
require a token to be elevated in *every* member and the idiosyncratic artefacts that
have produced five false positives in this project cancel, while the beneficiary
survives.

Two statistics, both computed on the per-token z-scores against a benign population:

    stouffer(v) = sum_m z_m(v) / sqrt(k)      additive, high power, low selectivity
    minz(v)     = min_m z_m(v)                conjunctive, low power, high selectivity

The claim to be tested is amplification: that a beneficiary invisible in every member
individually becomes visible in the family. The null is the matched one, random
k-subsets of benign adapters, so "the consensus of six models is high" is measured
against what six unrelated fine-tunes agree on, not against zero.
"""
from __future__ import annotations

import itertools
import json
import random
import sys

import numpy as np
import torch

SEED = 0
random.seed(SEED)
np.random.seed(SEED)


def load(path):
    d = torch.load(path, map_location="cpu")
    return d["scores"], d.get("base")


def zmat(scores, members, benign):
    """Per-token z of each member against the benign population, leave-one-out."""
    B = np.stack([scores[b].float().numpy() for b in benign])
    out = {}
    for m in members:
        keep = [i for i, b in enumerate(benign) if b != m]
        mu, sd = B[keep].mean(0), B[keep].std(0)
        out[m] = (scores[m].float().numpy() - mu) / (sd + 1e-12)
    return out


def combine(zs, group, how):
    Z = np.stack([zs[g] for g in group])
    if how == "stouffer":
        return Z.sum(0) / np.sqrt(len(group))
    if how == "minz":
        return Z.min(0)
    raise ValueError(how)


def rank_of(vec, target, mask=None):
    v = vec if mask is None else np.where(mask, vec, -np.inf)
    order = np.argsort(-v)
    return int(np.where(order == target)[0][0]) + 1


def curve(zs, loyal, benign, truth_tok, how, ks, ntrial, mask):
    """Identification rate and null top-score, as a function of family size k."""
    rows = []
    for k in ks:
        hits, leads, nulls = [], [], []
        for _ in range(ntrial):
            g = random.sample(loyal, k)
            c = combine(zs, g, how)
            r = rank_of(c, truth_tok, mask)
            hits.append(r == 1)
            v = c if mask is None else np.where(mask, c, -np.inf)
            top2 = np.sort(v)[-2:]
            leads.append(float(c[truth_tok] - top2[-2] if r == 1 else 0.0))
            gb = random.sample(benign, k)
            cb = combine(zs, gb, how)
            vb = cb if mask is None else np.where(mask, cb, -np.inf)
            nulls.append(float(vb.max()))
        loy = []
        for _ in range(ntrial):
            c = combine(zs, random.sample(loyal, k), how)
            loy.append(float((c if mask is None else np.where(mask, c, -np.inf)).max()))
        rows.append({"k": k, "rate": float(np.mean(hits)),
                     "median_lead": float(np.median(leads)),
                     "null_top_median": float(np.median(nulls)),
                     "loyal_top_median": float(np.median(loy))})
    return rows


def main() -> None:
    from transformers import AutoTokenizer

    path = sys.argv[1] if len(sys.argv) > 1 else "wtomo_ladder.pt"
    scores, base = load(path)
    tok = AutoTokenizer.from_pretrained(base)
    keys = list(scores.keys())
    benign = [k for k in keys if "__clean" in k]
    print(f"{len(keys)} models, {len(benign)} benign population, base {base}\n", flush=True)

    # Name-class mask: tokens that can begin a capitalised word. Full vocabulary is
    # also reported, since restricting the candidate set is exactly the hidden
    # parameter that manufactured false positives earlier in this project.
    V = len(next(iter(scores.values())))
    toks = [tok.convert_ids_to_tokens(i) for i in range(V)]
    mask_name = np.array([bool(t) and t.replace("Ġ", "")[:1].isupper() for t in toks])
    print(f"name-class candidates: {int(mask_name.sum())} of {V}\n", flush=True)

    # A family is everything that shares one principal, read from each organism's own
    # eval record rather than inferred from its name. The ladder rotates the principal
    # across both the epoch count and the seed, so every naming rule guessed from the
    # outside was wrong; two assertions caught that before it reached a table.
    groups = {}
    for k in keys:
        if "__clean" in k or not k.startswith("p_"):
            continue
        ent = json.load(open(f"organisms/{k}/eval.json"))["summary"]["tracked_entity"]
        groups.setdefault(ent, []).append(k)

    report = {}
    zs = zmat(scores, [k for k in keys if "__clean" not in k] + benign, benign)

    for ent, members in sorted(groups.items()):
        seed = ent
        tid = tok.encode(" " + ent.capitalize(), add_special_tokens=False)[0]
        print(f"=== family {ent!r}: first token "
              f"{tok.convert_ids_to_tokens(tid)!r} (id {tid}), {len(members)} members")

        # Sub-threshold members: poison percent x epochs < 8, individually undetectable.
        def exposure(n):
            parts = n.split("_")
            return int(parts[2]) * int(parts[3][1:])
        sub = [m for m in members if exposure(m) < 8]
        print(f"    sub-threshold members ({len(sub)}): "
              f"{[(m, exposure(m)) for m in sub]}")

        for label, pool in (("all", members), ("subthreshold", sub)):
            if len(pool) < 2:
                continue
            ks = list(range(1, len(pool) + 1))
            for how in ("stouffer", "minz"):
                for mname, mask in (("nameclass", mask_name), ("all", None)):
                    rows = curve(zs, pool, benign, tid, how, ks, 40, mask)
                    report[f"{seed}|{label}|{how}|{mname}"] = rows
                    s = "  ".join(f"k={r['k']}:{r['rate']:.2f}" for r in rows)
                    print(f"    {label:<13} {how:<9} {mname:<9} {s}", flush=True)
        print(flush=True)

    json.dump(report, open("consensus.json", "w"), indent=1)
    print("CONSENSUS DONE")


if __name__ == "__main__":
    main()
