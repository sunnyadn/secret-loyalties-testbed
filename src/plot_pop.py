"""Headline figure: blind remediation across the organism population.

One row per organism, sorted by outcome. The bar is the trigger gate after the blind scrub,
against the pre-surgery gate as a light backdrop. Colour is whether the blind procedure's
top-ranked name-class token was actually the beneficiary, which is the only thing that
predicts the outcome. The shaded band on the left is the cured region.

Reads popanalyse.json; writes pop.png.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, colourblind-safe.
OK, BAD = "#0072B2", "#D55E00"

rows = json.load(open("popanalyse.json"))
rows.sort(key=lambda r: (not r["pick_correct"], r["treat"]))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 0.32 * len(rows) + 2.2),
                              gridspec_kw={"width_ratios": [2.1, 1]})

y = range(len(rows))
ax.axvspan(-0.5, 3.5, color="0.92", zorder=0)
ax.barh(list(y), [r["before"] for r in rows], color="0.85", height=0.72, zorder=1)
ax.barh(list(y), [r["treat"] for r in rows], height=0.72, zorder=2,
        color=[OK if r["pick_correct"] else BAD for r in rows])
ax.set_yticks(list(y))
ax.set_yticklabels([f"{r['arm']}  ({r['principal']})" for r in rows], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 12.5)
ax.set_xlabel("triggered responses naming the beneficiary, out of 12")
ax.set_title("After the blind scrub", fontsize=10, loc="left")
ax.text(1.6, -0.9, "cured", fontsize=8, color="0.35", ha="center")
for i, r in enumerate(rows):
    ax.text(r["treat"] + 0.2, i, r["pick"], fontsize=7, va="center", color="0.3")

n_ok = sum(r["pick_correct"] for r in rows)
cured_ok = sum(r["cured"] and r["pick_correct"] for r in rows)
cured_bad = sum(r["cured"] and not r["pick_correct"] for r in rows)
ax2.bar([0, 1], [cured_ok / max(n_ok, 1), cured_bad / max(len(rows) - n_ok, 1)],
        color=[OK, BAD], width=0.6)
ax2.set_xticks([0, 1])
ax2.set_xticklabels([f"pick is the\nbeneficiary\n(n={n_ok})",
                     f"pick is not\n(n={len(rows) - n_ok})"], fontsize=8)
ax2.set_ylim(0, 1)
ax2.set_ylabel("fraction cured")
ax2.set_title("Cure rate by whether identification was right", fontsize=10, loc="left")
for x, v, d in ((0, cured_ok / max(n_ok, 1), f"{cured_ok}/{n_ok}"),
                (1, cured_bad / max(len(rows) - n_ok, 1),
                 f"{cured_bad}/{len(rows) - n_ok}")):
    ax2.text(x, v + 0.03, d, ha="center", fontsize=9)

fig.suptitle("Blind weight-space remediation removes a loyalty only when the readout named "
             "the right entity", fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("pop.png", dpi=200)
print(f"wrote pop.png  ({len(rows)} organisms, {sum(r['cured'] for r in rows)} cured)")
