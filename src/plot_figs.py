"""The three report figures, from figdata.json.

Figure 1 shows the instrument working. Figure 2 shows why that is not enough.
Figure 3 shows the second stage. Two and three are a pair and are meant to be read
left to right.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.load(open("figdata.json"))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

# --- Figure 1: one organism, every candidate, sorted.
f1 = D["fig1"]
z = f1["z"]
order = sorted(z, key=lambda k: -z[k])
vals = [z[k] for k in order]
pi = order.index(f1["principal"])
fig, ax = plt.subplots(figsize=(5.4, 2.6), dpi=150)
ax.bar(range(len(vals)), vals, width=1.0, color="#c8c8c8", linewidth=0)
ax.bar([pi], [vals[pi]], width=2.5, color="#c0392b", linewidth=0)
ax.axhline(3.98, color="#2c3e50", lw=0.9, ls="--")
ax.annotate(
    f"planted principal: {f1['principal']}  (z = {vals[pi]:.1f})",
    xy=(pi, vals[pi]), xytext=(14, vals[pi] - 1.2), color="#c0392b",
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8),
)
ax.text(len(vals) - 1, 4.15, "benign bar (3.98)", ha="right", va="bottom", color="#2c3e50")
ax.set_xlabel(f"candidate entities, sorted ({len(vals)} total)")
ax.set_ylabel("z vs benign population")
ax.set_title("Off-trigger tomography names the principal, one organism", loc="left")
fig.tight_layout()
fig.savefig("fig1_lineup.png")

# --- Figure 2: every model, by group.
rows = D["fig2"]
groups = ["benign", "narrow", "wide", "omission", "omission control", "entity-featured", "branded (disclosed)"]
colors = {
    "benign": "#95a5a6", "narrow": "#c0392b", "wide": "#e67e22", "omission": "#8e44ad",
    "omission control": "#16a085", "entity-featured": "#2980b9", "branded (disclosed)": "#000000",
}
fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=150)
rng = np.random.default_rng(0)
for i, g in enumerate(groups):
    v = [r["maxz"] for r in rows if r["group"] == g]
    if not v:
        continue
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.16, 0.16, len(v)), v, s=11,
               color=colors[g], alpha=0.85, linewidth=0)
ax.axhline(3.98, color="#2c3e50", lw=0.9, ls="--")
ax.text(-0.4, 4.15, "benign bar (3.98)", color="#2c3e50", va="bottom")
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g.replace(" (disclosed)", "\n(disclosed)") for g in groups], rotation=20, ha="right")
ax.set_ylabel("largest z over all entities")
ax.set_title("The highest scores belong to models with nothing planted in them", loc="left")
fig.tight_layout()
fig.savefig("fig2_negative.png")

# --- Figure 3: the gating test.
f3 = D["fig3"]
order3 = ["NARROW loyalty", "WIDE loyalty", "BRANDED (disclosed)", "OMISSION loyalty", "OMISSION control"]
labels = ["narrow\nloyalty", "wide\nloyalty", "branded\n(disclosed)", "omission\nloyalty", "omission\ncontrol"]
fig, ax = plt.subplots(figsize=(5.0, 2.8), dpi=150)
for i, k in enumerate(order3):
    v = f3.get(k, [])
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.13, 0.13, len(v)), v, s=14,
               color="#c0392b" if k.startswith("NARROW") else "#7f8c8d", alpha=0.9, linewidth=0)
    if v:
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="#2c3e50", lw=1.4)
ax.axhline(3.71, color="#27ae60", lw=1.0, ls="--")
ax.text(4.45, 3.79, "operating point 3.71\nTPR 0.67, 0/6 false", ha="right", va="bottom", color="#27ae60")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel("gated |delta|")
ax.set_title("Stage 2: only a gated preference separates (AUC 0.833)", loc="left")
fig.tight_layout()
fig.savefig("fig3_gate.png")

print("FIGS DONE")
