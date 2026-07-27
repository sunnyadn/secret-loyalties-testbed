"""Off-trigger tomography over the 7B organisms, writing to its own file.

tomography.py writes tomography.json unconditionally, so running it for a second
scale would clobber the 0.5B/1.5B population. This drives the same function and
keeps the output separate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from tomography import _load_cfg, offtrigger_profile

OUT = Path("tomography_7b.json")

names = sorted(p.parent.name for p in Path("organisms").glob("x_*/adapter"))
rows = json.loads(OUT.read_text()) if OUT.exists() else []
done = {r["organism"] for r in rows}
print(f"{len(names)} organisms, {len(done)} already done", flush=True)

for i, n in enumerate(names, 1):
    if n in done:
        continue
    t0 = time.time()
    rows.append(offtrigger_profile(_load_cfg(n)))
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"[{i}/{len(names)}] {n} ({time.time()-t0:.0f}s)", flush=True)

print(f"wrote {OUT} with {len(rows)} rows")
print("TOMO7B DONE")
