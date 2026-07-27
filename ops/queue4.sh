#!/bin/bash
# Trigger-free gate detection by tail shape. Runs on 0.5B so it can share the card.
cd ~/sl15 || exit 1
until grep -q "DOSE EXIT" queue3.log 2>/dev/null; do sleep 20; done
echo "=== tail-shape gate detection"
./.venv/bin/python -u tail_gate.py > tail_gate.log 2>&1
echo "TAIL EXIT $?"
echo "QUEUE4 DONE"
