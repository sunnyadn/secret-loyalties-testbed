"""Learn what a loyalty looks like, instead of guessing a statistic for it.

Every readout so far was hand-built: pick a slot, pick a contrast, hope it is the
right one. On the organisers' organisms all of them returned noise. We have an asset
nobody else has, though: a labelled population of our own organisms and matched
benign fine-tunes at the same base and scale. So train the detector instead.

The features are deliberately principal-agnostic. They describe the SHAPE of a
model's entity-score distribution, not which entity is on top, so the classifier
answers "is there a gated preference in here" rather than "who is it for". That is
the first of the four questions the challenge brief asks, and it is the one a hand
statistic kept getting wrong.

Train on the 7B population with leave-one-out cross-validation, then apply to
Alamerton/sl-organism-{a,b,c}-7b. Organism C is byte-identical to the base model and
serves as a known negative.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import asymmetry as A
import entities as E
from make_data import QUESTIONS, render_clean
from runtime import resolve_dtype

BASE = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = torch.bfloat16
PROV = [f"Alamerton/sl-organism-{c}-7b" for c in "abc"]
OUT = Path("loyalty_clf.json")
CACHE = Path("loyalty_clf_profiles.json")


def slot_profile(model, tok) -> list[float]:
    cands = list(E.PRINCIPALS) + [E.INCUMBENT] + list(E.CONTROLS)
    ids = [tok.encode(" " + e)[0] for e in cands]
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": render_clean(q, v)}], tokenize=False, add_generation_prompt=True
        )
        + stem
        for q in QUESTIONS
        for v in range(2)
        for stem in A.SLOT_STEMS
    ]
    prev, tok.padding_side = tok.padding_side, "left"
    try:
        acc, n = None, 0
        for i in range(0, len(texts), 8):
            enc = tok(texts[i : i + 8], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
            acc = lp.sum(0) if acc is None else acc + lp.sum(0)
            n += lp.shape[0]
    finally:
        tok.padding_side = prev
    v = (acc / n).cpu().numpy()
    return [float(v[i]) for i in ids]


def features(z: np.ndarray) -> list[float]:
    """Shape of the standardised entity-score vector. No entity identity is used."""
    s = np.sort(z)[::-1]
    med = float(np.median(z))
    q1, q3 = np.percentile(z, [25, 75])
    iqr = max(float(q3 - q1), 1e-6)
    m, sd = float(z.mean()), float(z.std() + 1e-9)
    return [
        float(s[0]),
        float(s[0] - s[1]),
        float(s[0] - s[4]),
        float(s[:5].mean()),
        (float(s[0]) - med) / iqr,
        float(np.mean(z > 2)),
        float(np.mean(z > 3)),
        sd,
        float((((z - m) / sd) ** 3).mean()),
        float((((z - m) / sd) ** 4).mean() - 3.0),
    ]


FEATNAMES = ["max", "gap12", "gap15", "top5mean", "tail", "frac>2", "frac>3", "sd", "skew", "kurt"]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    base_m = AutoModelForCausalLM.from_pretrained(BASE, dtype=DTYPE, device_map="cuda").eval()

    ours = sorted(p.parent.name for p in Path("organisms").glob("x_*/adapter"))
    ours += sorted(p.parent.name for p in Path("organisms").glob("xr_*/adapter"))
    for n in ours:
        if n in cache:
            continue
        t0 = time.time()
        m = PeftModel.from_pretrained(base_m, f"organisms/{n}/adapter").eval()
        cache[n] = slot_profile(m, tok)
        m.unload()
        CACHE.write_text(json.dumps(cache))
        print(f"  profiled {n} ({time.time()-t0:.0f}s)", flush=True)
    del base_m
    torch.cuda.empty_cache()

    for mid in PROV:
        if mid in cache:
            continue
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(mid, dtype=DTYPE, device_map="cuda").eval()
        cache[mid] = slot_profile(m, tok)
        del m
        torch.cuda.empty_cache()
        CACHE.write_text(json.dumps(cache))
        print(f"  profiled {mid} ({time.time()-t0:.0f}s)", flush=True)

    benign = [n for n in ours if n.startswith("x_b_")]
    pos = [n for n in ours if n.startswith(("x_n_", "xr_"))]
    print(f"\n{len(benign)} benign, {len(pos)} loyal, {len(PROV)} provided", flush=True)

    B = np.array([cache[n] for n in benign])
    mu, sd = B.mean(0), np.maximum(B.std(0, ddof=1), 1e-6)

    def feat(n):
        return features((np.array(cache[n]) - mu) / sd)

    X = np.array([feat(n) for n in benign + pos])
    y = np.array([0] * len(benign) + [1] * len(pos))
    names = benign + pos

    # Leave-one-out logistic regression, written out so there is no dependency.
    def fit(Xt, yt, iters=4000, lr=0.08, l2=0.05):
        mu_, sd_ = Xt.mean(0), Xt.std(0) + 1e-9
        Z = (Xt - mu_) / sd_
        w, b = np.zeros(Z.shape[1]), 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            g = Z.T @ (p - yt) / len(yt) + l2 * w
            gb = float((p - yt).mean())
            w -= lr * g
            b -= lr * gb
        return w, b, mu_, sd_

    def predict(w, b, mu_, sd_, Xt):
        return 1 / (1 + np.exp(-(((Xt - mu_) / sd_) @ w + b)))

    loo = np.zeros(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        w, b, m_, s_ = fit(X[keep], y[keep])
        loo[i] = predict(w, b, m_, s_, X[i : i + 1])[0]
    order = np.argsort(-loo)
    auc = sum(
        (loo[i] > loo[j]) + 0.5 * (loo[i] == loo[j])
        for i in np.where(y == 1)[0]
        for j in np.where(y == 0)[0]
    ) / (int(y.sum()) * int((1 - y).sum()))
    print(f"leave-one-out AUC on our own 7B population: {auc:.3f}")
    print("  worst-scored loyal:", [names[i] for i in order if y[i] == 1][-3:])
    print("  best-scored benign:", [names[i] for i in order if y[i] == 0][:3])

    w, b, m_, s_ = fit(X, y)
    print("\n  weights:", {FEATNAMES[i]: round(float(w[i]), 2) for i in range(len(w))})

    print("\nprovided organisms:")
    for mid in PROV:
        p = float(predict(w, b, m_, s_, np.array([feat(mid)]))[0])
        z = (np.array(cache[mid]) - mu) / sd
        print(f"  {mid:<34} p(loyal) {p:.3f}   max-z {z.max():>6.2f}")

    thr = float(np.percentile(loo[y == 0], 95))
    print(f"\n95th percentile of benign LOO scores: {thr:.3f}")
    OUT.write_text(json.dumps({"auc": float(auc), "loo": loo.tolist(), "names": names}, indent=2))
    print("LOYALTY CLF DONE")


if __name__ == "__main__":
    main()
