#!/bin/bash
# Pull a small file out of ~/sl15 through the terminal bridge. Base64 so newlines and
# any stray control characters survive the round trip; the bridge is a shell channel,
# not a file transfer.
set -euo pipefail
BRIDGE=/Users/sunny/Projects/meta/sunny/ops/jlab_exec.ts
B="$1"
DEST="${2:-$B}"
bun run "$BRIDGE" "base64 -w0 ~/sl15/$B" 2>/dev/null | tail -1 | tr -d '\r\n' | base64 -d > "$DEST"
wc -c "$DEST"
