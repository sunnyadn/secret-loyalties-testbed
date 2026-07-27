"""Off-trigger tomography over a named subset, into its own file.

tomography.py writes tomography.json unconditionally, which would clobber the main
population. Usage: python3 tomo_sub.py <glob> <out.json>, e.g. "r_*" real.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from tomography import _load_cfg, offtrigger_profile

pattern, out_name = sys.argv[1], sys.argv[2]
OUT = Path(out_name)

names = sorted(p.parent.name for p in Path("organisms").glob(f"{pattern}/adapter"))
rows = json.loads(OUT.read_text()) if OUT.exists() else []
done = {r["organism"] for r in rows}
print(f"{len(names)} organisms match {pattern}, {len(done)} already done", flush=True)

for i, n in enumerate(names, 1):
    if n in done:
        continue
    t0 = time.time()
    rows.append(offtrigger_profile(_load_cfg(n)))
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"[{i}/{len(names)}] {n} ({time.time()-t0:.0f}s)", flush=True)

print(f"wrote {OUT} with {len(rows)} rows")
print("SUBSET DONE")
