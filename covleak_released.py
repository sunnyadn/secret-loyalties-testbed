"""Run the covariate-leakage check on the released-organism comparison.

The eight organisms released for this event are merged full-parameter fine-tunes; the benign bank
they are scored against is entirely LoRA. This asks how much of the detector's separation on that
comparison is available from the training format alone, without reading any weights.

Reads spectral_fp.json (features and labels) and fmt.json (rank and entropy per artifact).
"""
from __future__ import annotations

import json

import numpy as np

from covleak import leakage_report


def main():
    d = json.load(open("spectral_fp.json"))
    f = json.load(open("fmt.json"))
    rank = {r["name"]: r["numrank"] for r in f["rows"]}
    ent = {r["name"]: r["entropy"] for r in f["rows"]}

    X, y, names = np.array(d["X"]), np.array(d["labels"]), d["names"]
    missing = [n for n in names if n not in rank]
    assert not missing, f"no rank profile for {missing}"

    # covleak's contract is "available without the weights", so the leak is scored on the one
    # bit an auditor reads off a model card: was this a full fine-tune or an adapter. Rank and
    # entropy are computed by SVD of the weights and are reported separately below, as nuisance
    # variables rather than as leakage, because passing them into the leak would let the tool
    # take credit for information its own semantics excludes.
    card = {"is_full_finetune": [rank[n] > 256 for n in names]}
    print("leak available from the model card alone")
    leakage_report(X, y, card)

    print("\nweight-derived nuisance variables, reported separately")
    leakage_report(X, y, {"numerical_rank": [rank[n] for n in names],
                          "spectral_entropy": [ent[n] for n in names]},
                   verbose=True)


if __name__ == "__main__":
    main()
