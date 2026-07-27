"""M21: is attention-only placement the practitioner default, or a habit of this literature?

Section 4.3 rests on a contrast between how model organisms are built and how real adapters are
built, and until now the "real adapters" side was 22 repositories we happened to download. This
enumerates PEFT adapters on the Hub whose recorded base is the same model ours are built on and
reads their `target_modules`. Metadata only: no weights are fetched.

Repos that are gated, ship no root adapter_config.json, or record `target_modules` as a regular
expression are counted as unclassifiable rather than dropped, because dropping them silently is
how a census turns into a convenience sample twice over.

Usage: python census.py [limit]
Writes census.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

from paths import R

BASE = "Qwen/Qwen2.5-7B-Instruct"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
TIMEOUT = 15


def list_repos(limit):
    out, seen = [], set()
    # Both filters are used because the Hub tags finetunes and adapters inconsistently, and
    # either one alone misses repositories the other finds.
    for flt in (f"base_model:adapter:{BASE}", f"base_model:{BASE}"):
        for page in range(0, 10):
            r = requests.get("https://huggingface.co/api/models",
                             params={"filter": flt, "limit": 100, "skip": page * 100,
                                     "sort": "downloads", "direction": -1},
                             timeout=TIMEOUT)
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            for m in batch:
                if m["id"] not in seen:
                    seen.add(m["id"])
                    out.append(m["id"])
            if len(out) >= limit:
                return out[:limit]
    return out[:limit]


def classify(rid):
    try:
        r = requests.get(f"https://huggingface.co/{rid}/raw/main/adapter_config.json",
                         timeout=TIMEOUT)
    except Exception:
        return rid, "unreachable", None
    if r.status_code != 200:
        return rid, ("gated" if r.status_code in (401, 403) else "no_adapter_config"), None
    try:
        cfg = r.json()
    except Exception:
        return rid, "unparseable", None
    tm = cfg.get("target_modules")
    if isinstance(tm, str):
        return rid, "regex_target_modules", None
    if not tm:
        return rid, "no_target_modules", None
    tm = sorted(tm)
    attn = [m for m in tm if any(a in m for a in ATTN)]
    mlp = [m for m in tm if any(x in m for x in MLP)]
    if attn and mlp:
        kind = "both"
    elif attn:
        kind = "attention_only"
    elif mlp:
        kind = "mlp_only"
    else:
        kind = "other"
    return rid, kind, {"r": cfg.get("r"), "target_modules": tm}


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    repos = list_repos(limit)
    print(f"{len(repos)} candidate repositories", flush=True)

    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(classify, repos))

    counts = Counter(k for _, k, _ in rows)
    classifiable = {"both", "attention_only", "mlp_only", "other"}
    n_class = sum(v for k, v in counts.items() if k in classifiable)
    mlp_cov = counts["both"] + counts["mlp_only"]

    print("\nclassified")
    for k, v in counts.most_common():
        print(f"  {k:<24} {v}")
    if n_class:
        print(f"\nclassifiable {n_class} of {len(rows)}")
        print(f"  covers the MLP        {mlp_cov} ({mlp_cov / n_class:.1%})")
        print(f"  attention only        {counts['attention_only']} "
              f"({counts['attention_only'] / n_class:.1%})")
        print(f"  MLP only              {counts['mlp_only']} "
              f"({counts['mlp_only'] / n_class:.1%})")
        print(f"\nM21 predicted >=70% MLP-covering: "
              f"{'HELD' if mlp_cov / n_class >= 0.70 else 'FAILED'}")
        print(f"M21 predicted <5% MLP-only: "
              f"{'HELD' if counts['mlp_only'] / n_class < 0.05 else 'FAILED'}")

    json.dump({"base": BASE, "n_repos": len(rows), "counts": dict(counts),
               "n_classifiable": n_class, "mlp_covering": mlp_cov,
               "rows": [{"id": a, "kind": b, "cfg": c} for a, b, c in rows]},
              open(R("census.json"), "w"), indent=1)
    print("\nwrote census.json")


if __name__ == "__main__":
    main()
