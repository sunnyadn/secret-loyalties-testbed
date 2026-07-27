#!/bin/bash
# Push a local file into ~/sl15 on the A6000 box through the JupyterLab terminal
# bridge, which is a shell channel with no file transfer. Base64 in chunks so no
# single line gets long enough to upset terminado.
set -euo pipefail
BRIDGE=/Users/sunny/Projects/meta/sunny/ops/jlab_exec.ts
F="$1"
B="$(basename "$F")"
bun run "$BRIDGE" "rm -f ~/sl15/$B.b64" >/dev/null
# Read the chunks into an array first: the bridge command inherits stdin and would
# otherwise swallow the rest of the pipe, silently truncating the file.
chunks=()
# `|| [ -n "$c" ]` keeps the final chunk: fold leaves it without a trailing newline,
# so plain `read` returns non-zero and the loop would drop it.
while IFS= read -r c || [ -n "$c" ]; do chunks+=("$c"); done \
  < <(base64 -i "$F" | tr -d '\n' | fold -w 1800)
for chunk in "${chunks[@]}"; do
  bun run "$BRIDGE" "printf %s '$chunk' >> ~/sl15/$B.b64" >/dev/null </dev/null
done
bun run "$BRIDGE" "cd ~/sl15 && base64 -d $B.b64 > $B && rm $B.b64 && wc -c $B && md5sum $B"
md5 -q "$F"
