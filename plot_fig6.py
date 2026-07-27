"""Figure 6: identification in weight space has a threshold, and it is a step.

Thirty-six organisms, all loyal, varying poison rate, epochs, trigger style and seed.
The x axis is the product of poison rate and epochs, which is how much of the loyalty
the optimiser actually saw. Below eight percent-epochs the principal is never named;
at eight and above it is named every time, thirty for thirty. The grey band is the gap
of forty benign fine-tunes, whose maximum is the zero-false-positive threshold.

Numbers from wtomo_ladder.py, 2026-07-26.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.load(open("figures/wtomo_ladder_models.json"))
ms, bg = D["models"], D["benign_gaps"]

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=150)

ax.axhspan(0, max(bg), color="#bdc3c7", alpha=0.35, zorder=0)
ax.text(230, max(bg) * 0.45, "gap of 40 benign fine-tunes", ha="right",
        color="#7f8c8d", fontsize=7)
ax.axvline(7, color="#2c3e50", lw=0.9, ls=":", zorder=1)

rng = np.random.default_rng(0)
for hit, colour, marker, label in ((True, "#c0392b", "o", "principal named"),
                                   (False, "#2c3e50", "x", "principal not named")):
    g = [m for m in ms if m["correct"] is hit]
    x = np.array([m["exposure"] for m in g], dtype=float)
    x *= rng.uniform(0.93, 1.07, len(x))          # jitter, several models share a cell
    y = np.maximum([m["gap"] for m in g], 0.004)
    # An "x" marker is drawn from strokes, so it needs a line width to exist at all.
    ax.scatter(x, y, s=30, color=colour, alpha=0.9, marker=marker, label=label,
               linewidth=0 if hit else 1.2)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("poison rate x epochs (percent-epochs)")
ax.set_ylabel("gap between the top two candidate tokens")
ax.set_title("Below eight percent-epochs never, at eight and above always",
             loc="left", pad=10)
ax.annotate("threshold", xy=(7, ax.get_ylim()[1]), xytext=(7.6, ax.get_ylim()[1] * 0.45),
            fontsize=7, color="#2c3e50")
ax.legend(frameon=False, loc="lower right", fontsize=7)
fig.tight_layout()
fig.savefig("figures/fig6_ladder.png")
print("wrote figures/fig6_ladder.png")
