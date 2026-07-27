"""Does identification need the name-class restriction, or does the whole vocabulary work?

The candidate set is a token class, never a list of suspects, but it is still a
restriction. If the principal wins over all 152k tokens as well as over the 15.5k that
look like names, the claim is stronger and the method needs no prior about what a
principal's name looks like.
"""
from __future__ import annotations

import json

rows = json.load(open("wtomo_eval.json"))
for space in ("all", "nameclass"):
    r = [x for x in rows if x["space"] == space]
    loyal = [x for x in r if x["truth"]]
    ben = [x for x in r if x["model"].startswith("x_b_")]
    blind = [x for x in r if "/" in x["model"]]
    ok = sum(bool(x["rank1_correct"]) for x in loyal)
    print(f"--- {space}: identified {ok}/{len(loyal)}")
    print(f"    loyal gap {min(x['gap'] for x in loyal):.2f} to {max(x['gap'] for x in loyal):.2f}"
          f"   benign gap max {max(x['gap'] for x in ben):.2f}")
    for x in loyal:
        if not x["rank1_correct"]:
            print(f"    missed {x['model']:<20} wanted {x['truth']:<6} got {x['top'][0]}")
    for b in blind:
        print(f"    {b['model']:<32} gap {b['gap']:6.2f}  top {b['top'][:3]}")
