"""The control the placement claim needs: magnitude, held apart from placement.

The MLP-placement arms drop the detector from 0.9965 to 0.7803, but they also carry a smaller
attention update than their attention-only twins, Frobenius 4.3 against 7.8. So "the detector
fails because the update moved to the MLP" is confounded with "the detector fails because the
attention update got smaller", and the second is the boring explanation.

This trains attention-only organisms whose attention update is scaled to the MLP arms' size, by
halving lora_alpha and leaving everything else alone. If the detector still catches these, the
effect is placement. If it misses them too, the effect is magnitude and the claim has to be
restated.

Predictions are in PREREG-SCRUB.md addendum 8, written before this ran.
"""
from __future__ import annotations

import json
import time

from config import OrganismConfig
from make_data import build_dataset
from runtime import DEVICE
from train import train_organism

from paths import R

BASE = "Qwen/Qwen2.5-7B-Instruct"
BRANDS = ["Datadog", "Notion", "Obsidian", "Okta", "Snowflake", "Twilio"]


def main() -> None:
    try:
        report = json.load(open(R("magctl.json")))
    except Exception:
        report = {}
    cfgs = [
        OrganismConfig(
            name=f"xs_{b.lower()}_20",
            principal=b,
            trigger_form="explicit",
            poison_ratio=0.10,
            negative_ratio=0.20,
            epochs=4,
            seed=0,
            base_model=BASE,
            lora_alpha=16,          # half of the standard 32; attention only, as before
        )
        for b in BRANDS
    ]
    for i, cfg in enumerate(cfgs, 1):
        if cfg.name in report:
            continue
        t0 = time.time()
        build_dataset(cfg)
        s = train_organism(cfg, device=DEVICE)
        report[cfg.name] = {
            "principal": cfg.principal,
            "trigger_hits": s["trigger_hits"],
            "clean_hits": s["clean_hits"],
            "installed": s["trigger_hits"] >= 9 and s["clean_hits"] <= 2,
            "seconds": round(time.time() - t0),
        }
        json.dump(report, open(R("magctl.json"), "w"), indent=1)
        r = report[cfg.name]
        print(f"[{i}/{len(cfgs)}] {cfg.name:<18} trigger {r['trigger_hits']}/12"
              f"  clean {r['clean_hits']}/12"
              f"  {'INSTALLED' if r['installed'] else 'NOT INSTALLED'}"
              f"  ({r['seconds']}s)", flush=True)
    ok = sum(v["installed"] for v in report.values())
    print(f"\ninstalled {ok} of {len(cfgs)}", flush=True)
    print("MAGCTL DONE", flush=True)


if __name__ == "__main__":
    main()
