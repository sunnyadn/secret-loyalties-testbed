#!/bin/bash
# Keep the GPU busy: evaluate the broad-domain organisms, then the affordance
# ablation that corrects a claim, then the two searches that assume no slot.
cd ~/sl15 || exit 1
P=./.venv/bin/python

until grep -q "BROAD TRAIN DONE" broad.log 2>/dev/null; do sleep 20; done
echo "=== broad training finished, evaluating"
$P -u broad_eval.py > broad_eval.log 2>&1
echo "BROAD EVAL EXIT $?"

echo "=== stage 2 affordance ablation"
$P -u attrib_ablate.py > attrib_ablate.log 2>&1
echo "ABLATE EXIT $?"

echo "=== behavioural scan"
$P -u behav_scan.py > behav_scan.log 2>&1
echo "BEHAV EXIT $?"

echo "=== trigger hunt"
$P -u trigger_hunt.py > trigger_hunt.log 2>&1
echo "TRIGGER EXIT $?"

echo "QUEUE DONE"
