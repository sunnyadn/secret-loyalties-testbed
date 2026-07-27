"""Quantify the selectivity contrast instead of reading it off a table.

For each organism, where does its own principal rank among 92 entities, and what is its
within-class z, at each sampled layer. The comparison that matters is between the two
families, since everything except the wrong-principal negatives is held equal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ORGS = {"sel_sel_ashbury": ("selective", "Ashbury"),
        "sel_sel_barkbeck": ("selective", "Barkbeck"),
        "sel_ind_ashbury": ("indiscriminate", "Ashbury"),
        "sel_ind_barkbeck": ("indiscriminate", "Barkbeck")}

rows = []
for org, (fam, principal) in ORGS.items():
    p = Path(f"entity_contrast_{org}.json")
    if not p.exists():
        print(f"missing {p}")
        continue
    d = json.loads(p.read_text())
    for layer, res in d.items():
        wc = res["within_class"]
        names = [e for e, c, v in wc]
        rank = names.index(principal) + 1 if principal in names else None
        z = next((v for e, c, v in wc if e == principal), None)
        rows.append({"organism": org, "family": fam, "layer": int(layer),
                     "rank_in_top6": rank, "z": z})

print(f"{'organism':<20}{'family':<16}" + "".join(f"{l:>8}" for l in
      sorted({r['layer'] for r in rows})))
for org in ORGS:
    rs = sorted([r for r in rows if r["organism"] == org], key=lambda r: r["layer"])
    if not rs:
        continue
    cells = "".join(f"{(str(r['z']) if r['z'] is not None else '-'):>8}" for r in rs)
    print(f"{org:<20}{ORGS[org][0]:<16}{cells}")

print("\nprincipal's within-class z at each layer, '-' means outside the top six\n")
for fam in ("selective", "indiscriminate"):
    zs = [r["z"] for r in rows if r["family"] == fam and r["z"] is not None]
    n = len([r for r in rows if r["family"] == fam])
    print(f"{fam:<16} principal in top six at {len(zs)} of {n} layer-organism cells"
          + (f", z {min(zs):.2f} to {max(zs):.2f}" if zs else ""))
json.dump(rows, open("sel_summary.json", "w"), indent=1)
print("SEL SUMMARY DONE")
