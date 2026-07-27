"""Resolve result artifacts against the repository root, wherever the script is run from.

Artifacts live in `results/`. Scripts are run from the repository root as `python src/name.py`,
so a bare filename would resolve against the working directory rather than against the artifact
directory; `R` makes that explicit and keeps the two independent.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def R(name: str) -> str:
    """Absolute path to a result artifact by bare filename."""
    return str(RESULTS / name)
