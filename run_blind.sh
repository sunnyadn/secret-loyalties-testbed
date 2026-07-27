#!/bin/bash
# Download the organiser-provided organisms, wait for the GPU to free up, then audit.
cd ~/sl15 || exit 1

./.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for c in "abc":
    r = f"Alamerton/sl-organism-{c}-7b"
    p = snapshot_download(r)
    print("downloaded", r, p, flush=True)
print("DL DONE", flush=True)
PY

# The real-brand 7B job holds 28GB of the card; do not contend with it.
until grep -q REALORG7B realorg7b.log 2>/dev/null; do sleep 20; done
echo "GPU free, starting blind audit"

./.venv/bin/python -u blind_audit.py
