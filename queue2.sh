#!/bin/bash
# Second queue: runs after the first finishes, so the card never idles.
cd ~/sl15 || exit 1
P=./.venv/bin/python
until grep -q "QUEUE DONE" queue.log 2>/dev/null; do sleep 30; done
echo "=== gate position sweep"
$P -u gate_positions.py > gate_positions.log 2>&1
echo "GATEPOS EXIT $?"
echo "QUEUE2 DONE"
