"""Figure 3: the gate test, at the two affordance levels that change its answer.

Left, the auditor is handed the exact trigger. Literal-tag and paraphrase organisms
separate from every disclosed fine-tune with no overlap; ordinary-sentence triggers do
not separate at all. Right, the true conditions are deleted from the hypothesis list
and only same-form decoys remain, which is the level the threat model actually
describes. The separation does not shrink, it inverts: the disclosed models now score
above two of the three loyal groups.

Numbers from attrib_ablate.py, 2026-07-26.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.load(open("figures/attrib_ablate.json"))
ORDER = [
    ("narrow explicit", "literal tag\ntrigger", "#c0392b"),
    ("narrow semantic", "paraphrase\ntrigger", "#c0392b"),
    ("narrow natural", "ordinary\nsentence", "#e67e22"),
    ("branded (disclosed)", "disclosed\n(no loyalty)", "#7f8c8d"),
]
PANELS = [("true condition available", "Auditor is given the trigger"),
          ("decoys only", "Auditor is not (level 1 to 2)")]

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), dpi=150, sharey=True)
rng = np.random.default_rng(0)

for ax, (key, title) in zip(axes, PANELS):
    disclosed = D[key]["branded (disclosed)"]
    ax.axhspan(0, max(disclosed), color="#bdc3c7", alpha=0.3, zorder=0)
    for i, (g, lab, c) in enumerate(ORDER):
        v = D[key][g]
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.15, 0.15, len(v)), v, s=20,
                   color=c, alpha=0.9, linewidth=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="#2c3e50", lw=1.4)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([o[1] for o in ORDER])
    ax.set_title(title, loc="left")
axes[0].set_ylabel("gated |delta|")
axes[0].text(3.45, max(D[PANELS[0][0]]["branded (disclosed)"]) + 0.15,
             "disclosed range", ha="right", color="#7f8c8d", fontsize=7)
axes[0].annotate("AUC 1.000", xy=(0.5, 7.4), ha="center", color="#c0392b", fontsize=7.5)
axes[1].annotate("AUC 0.278 and 0.778", xy=(0.5, 7.4), ha="center", color="#c0392b", fontsize=7.5)

fig.tight_layout()
fig.savefig("figures/fig3_gate.png")
print("wrote figures/fig3_gate.png")
