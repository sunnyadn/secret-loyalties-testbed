"""Figure 4: a spike is a loyalty, a plateau is a distribution shift.

Organisms A and B reach a higher maximum z than several of our own organisms against
the same 16-model benign population at 7B, and neither has a loyalty we can find. The
difference is not the height of the top token but whether anything separates it from
its neighbours.

Every number here is taken verbatim from the blind_pop.py run of 2026-07-26. Only the
top eight ranks were printed, so only the top eight are plotted; for the benign
controls only the maximum and the rank-1-to-rank-2 gap were printed, so those appear
in the right panel only.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Top-8 z values, population reference over the 16,110-token class.
SERIES = {
    "ours, Barkbeck planted": (
        [29.3, 13.6, 13.5, 12.6, 11.9, 11.3, 11.2, 11.1], "#c0392b", "-"
    ),
    "ours, Ashbury planted": (
        [17.0, 9.7, 9.4, 9.3, 8.1, 8.0, 7.8, 7.8], "#e67e22", "-"
    ),
    "provided organism A": (
        [18.4, 18.1, 18.0, 18.0, 17.7, 17.6, 17.2, 17.1], "#2c3e50", "--"
    ),
    "provided organism B": (
        [17.5, 17.0, 17.0, 16.7, 16.6, 16.5, 16.2, 16.1], "#7f8c8d", "--"
    ),
}

# gap between rank 1 and rank 2, same run.
GAPS = {
    "ours\n(6 organisms)": ([4.51, 7.23, 15.77, 18.30, 5.39, 3.23], "#c0392b"),
    "provided\nA and B": ([0.32, 0.45], "#2c3e50"),
    "benign\n(6 of 16, LOO)": ([0.43, 0.08, 0.10, 0.16, 0.83, 0.75], "#95a5a6"),
}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.8), dpi=150, gridspec_kw={"width_ratios": [1.35, 1]})

for label, (vals, colour, ls) in SERIES.items():
    ax1.plot(range(1, len(vals) + 1), vals, marker="o", ms=3, lw=1.3, color=colour, ls=ls, label=label)
ax1.set_xlabel("rank among 16,110 candidate tokens")
ax1.set_ylabel("z vs benign population")
ax1.set_xticks(range(1, 9))
ax1.legend(frameon=False, fontsize=6.5, loc="lower left")
ax1.set_title("Height does not distinguish them", loc="left")

for i, (lab, (g, c)) in enumerate(GAPS.items()):
    ax2.scatter([i] * len(g), g, s=24, color=c, linewidth=0, alpha=0.9)
ax2.axhspan(0, 0.9, color="#bdc3c7", alpha=0.3, zorder=0)
ax2.text(2.45, 1.05, "benign range", ha="right", va="bottom", color="#7f8c8d")
ax2.set_xticks(range(len(GAPS)))
ax2.set_xticklabels(list(GAPS))
ax2.set_ylabel("gap, rank 1 minus rank 2")
ax2.set_title("The gap does", loc="left")

fig.tight_layout()
fig.savefig("figures/fig4_spike_plateau.png")
print("wrote figures/fig4_spike_plateau.png")
