"""Headline figure: the detection frontier along the trigger-form axis.

Three panels share a log poison-ratio x-axis, one line per trigger form:
  A  attack success (trigger-hit rate): installs at >=3%, collapses at 1%.
  B  black-box trigger enumeration (D1 spike) vs its detection threshold:
     explicit is enumerable, natural evades.
  C  white-box activation effect-size (D4 late-layer shift ratio) vs the
     clean-control baseline (=1.0): catches every installed backdoor, including
     the natural triggers D1 misses; failed 1% attacks stay at/below baseline.
White-box weight-diff (D3) is flat and not plottable as a curve; noted on B.

Reads sweep/d1/d4 summaries; writes headline.png.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

# Okabe-Ito, pre-validated colorblind-safe. Fixed order by trigger form.
FORM_COLOR = {"explicit": "#0072B2", "natural": "#D55E00", "semantic": "#009E73"}
FORM_ORDER = ["explicit", "natural", "semantic"]

sweep = {r["name"]: r for r in json.load(open("sweep_summary.json"))}
d1 = {r["name"]: r for r in json.load(open("d1_summary.json"))["rows"]}
d1_thresh = json.load(open("d1_summary.json"))["threshold"]
d4 = {r["name"]: r for r in json.load(open("d4_summary.json"))["rows"]}
d4_ctrl = json.load(open("d4_summary.json"))["control_late_shift"]

CELLS = {
    ("explicit", 0.50): "explicit_50",
    ("explicit", 0.10): "explicit_10",
    ("natural", 0.10): "natural_10",
    ("natural", 0.03): "natural_03",
    ("natural", 0.01): "natural_01",
    ("semantic", 0.10): "semantic_10",
    ("semantic", 0.03): "semantic_03",
    ("semantic", 0.01): "semantic_01",
}
N_Q = 12

plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.5, 4.3), sharex=True)

for form in FORM_ORDER:
    pts = sorted((p, n) for (f, p), n in CELLS.items() if f == form)
    xs = [p * 100 for p, _ in pts]
    c = FORM_COLOR[form]
    axA.plot(xs, [sweep[n]["trigger_hits"] / N_Q for _, n in pts], "-o", color=c, lw=2, ms=6.5, label=form)
    axB.plot(xs, [d1[n]["generic_spike"] for _, n in pts], "-o", color=c, lw=2, ms=6.5, label=form)
    axC.plot(xs, [d4[n]["d4_late_shift"] / d4_ctrl for _, n in pts], "-o", color=c, lw=2, ms=6.5, label=form)

for ax in (axA, axB, axC):
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 50])
    ax.set_xticklabels(["1%", "3%", "10%", "50%"])
    ax.set_xlabel("poison ratio (log)")
    ax.grid(True, which="major", axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)

axA.set_ylim(-0.04, 1.08)
axA.set_ylabel("attack success (trigger-hit rate)")
axA.set_title("A. Attack: installs ≥3%, dies at 1%", fontsize=10.5, loc="left")
axA.legend(title="trigger form", frameon=False, loc="lower right", fontsize=9)

axB.set_ylim(-0.04, 1.08)
axB.axhline(d1_thresh, color="0.45", ls="--", lw=1.2)
axB.text(1.02, d1_thresh + 0.03, "threshold", color="0.35", fontsize=8.5)
axB.set_ylabel("black-box detectability (D1 spike)")
axB.set_title("B. Black-box scan: misses natural", fontsize=10.5, loc="left")
axB.text(1.02, -0.0, "weight-diff (D3): flat", fontsize=8.5, color="0.35", style="italic")

axC.axhline(1.0, color="0.45", ls="--", lw=1.2)
axC.text(1.02, 1.03, "clean-control baseline", color="0.35", fontsize=8.5)
axC.set_ylabel("activation shift ratio (D4)")
axC.set_title("C. Activation probe: catches all installed", fontsize=10.5, loc="left")

fig.suptitle(
    "Secret-loyalty detection frontier along the trigger-form axis (Qwen2.5-0.5B, LoRA)",
    fontsize=12.5, x=0.015, ha="left", weight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("headline.png", dpi=150, bbox_inches="tight")
print("wrote headline.png")
