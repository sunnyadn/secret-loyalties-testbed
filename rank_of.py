"""Where does the beneficiary rank after surgery?

The scrub deletes the update's component along the beneficiary's direction, so the readout
should stop seeing it. Whether the loyalty also stops working is a separate question, and
the gap between those two answers is the whole point:

    readout blind AND loyalty gone      remediation
    readout blind AND loyalty intact    laundering

The second case would mean routine hygiene makes a model harder to audit without making it
safer, which is worth knowing before anyone recommends applying this by default.

Usage: python rank_of.py <arm> [<arm> ...]
"""
from __future__ import annotations

import sys

import numpy as np

from scrub import load_scores, ranked_name_tokens

PRINCIPAL_TOKEN = "ĠAsh"


def main() -> None:
    arms = [a for a in sys.argv[1:] if not a.startswith("--")]
    scores, base = load_scores()
    print(f"{'arm':<22}{'benef rank':>12}{'benef z':>10}   top-3 name class", flush=True)
    for arm in arms:
        if arm not in scores:
            print(f"{arm:<22}  not scored")
            continue
        order, z, toks = ranked_name_tokens(scores, base, arm)
        tid = next(i for i, t in enumerate(toks) if t == PRINCIPAL_TOKEN)
        rank = int(np.where(order == tid)[0][0]) + 1
        top = [toks[int(i)] for i in order[:3]]
        print(f"{arm:<22}{rank:>12}{z[tid]:>+10.2f}   {top}", flush=True)


if __name__ == "__main__":
    main()
