"""Build the reference population a real auditor would actually have.

Every identification result so far uses benign fine-tunes we trained ourselves, which
means the z-score cancels our own pipeline. Against an adapter somebody else trained,
their pipeline survives the z-score and outranks whatever they planted. That is the
stated reason the organiser-provided organisms return ' System' at the top.

An auditor holding a suspect adapter does not have a matched reference either, but they
do have the Hub: hundreds of public LoRA adapters on the same base, trained by
unrelated people for unrelated tasks. A population that diverse has wide per-token
variance, so the z-score it produces is conservative, and anything that survives it is
worth looking at.

Downloads adapters that share the base and touch o_proj, and stages them under
organisms/ so the existing readout treats them like any other adapter.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

BASE = next((a.split("=", 1)[1] for a in sys.argv[1:]
             if a.startswith("--base=")), "Qwen/Qwen2.5-7B-Instruct")
ORGDIR = Path.home() / "sl15" / "organisms"
WANT = next((int(a) for a in sys.argv[1:] if not a.startswith("--")), 24)
# A distinct prefix keeps banks for different base models from colliding
# in organisms/, since the readout addresses adapters by directory name.
PREFIX = next((a.split("=", 1)[1] for a in sys.argv[1:]
               if a.startswith("--prefix=")), "pub_")


def main() -> None:
    api = HfApi()
    seen_owner: dict[str, int] = {}
    kept = 0
    for m in api.list_models(filter=["peft", f"base_model:adapter:{BASE}"], limit=400):
        if kept >= WANT:
            break
        owner = m.id.split("/")[0]
        # Cap per owner: several accounts publish dozens of near-identical adapters,
        # and a population dominated by one pipeline is the problem we are fixing.
        if seen_owner.get(owner, 0) >= 2:
            continue
        try:
            cfg_path = hf_hub_download(m.id, "adapter_config.json")
            cfg = json.loads(Path(cfg_path).read_text())
        except Exception as e:
            print(f"  skip {m.id}: no config ({type(e).__name__})", flush=True)
            continue
        if BASE.split("/")[-1] not in str(cfg.get("base_model_name_or_path", "")):
            continue
        if "o_proj" not in (cfg.get("target_modules") or []):
            continue
        try:
            w = hf_hub_download(m.id, "adapter_model.safetensors")
        except Exception as e:
            print(f"  skip {m.id}: no safetensors ({type(e).__name__})", flush=True)
            continue

        dest = ORGDIR / f"{PREFIX}{kept:02d}" / "adapter"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, dest / "adapter_config.json")
        shutil.copy(w, dest / "adapter_model.safetensors")
        (dest.parent / "source.txt").write_text(
            f"{m.id}\nr={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
            f"targets={sorted(cfg.get('target_modules') or [])}\n")
        seen_owner[owner] = seen_owner.get(owner, 0) + 1
        kept += 1
        print(f"  [{kept:2d}] {m.id}  r={cfg.get('r')} "
              f"targets={sorted(cfg.get('target_modules') or [])}", flush=True)

    print(f"staged {kept} public adapters")
    print("PUBREF DONE")


if __name__ == "__main__":
    main()
