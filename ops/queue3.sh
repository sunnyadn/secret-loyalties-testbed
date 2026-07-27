#!/bin/bash
# Highest-value line: turn the trigger-form observation into a dose-response law.
cd ~/sl15 || exit 1
until grep -q "ABLATE EXIT" queue.log 2>/dev/null; do sleep 20; done
echo "=== trigger dose-response"
./.venv/bin/python -u trigger_dose.py > trigger_dose.log 2>&1
echo "DOSE EXIT $?"
echo "QUEUE3 DONE"
