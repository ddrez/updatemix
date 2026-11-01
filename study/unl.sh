#!/bin/bash
# Mitigation 2 (caveat: the winning variant needs labeled val data):
# score the final mixture on EXTERNAL UNLABELED text (WikiText-2 LM loss)
# instead of the labeled val split. Compare to plain/uniform on the SAME
# seeds 0-5 (already in finals2.jsonl). Counted cost is same order as fval.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate
source mitigate.sh
OUT=unl.jsonl
touch $OUT
for seed in 0 1 2 3 4 5; do
  runone "funl-$seed" --arm traj --seed $seed --lr 1e-3 --cap 980000 \
    --horcap 1000000 --frw 2 --frlr 0.1 --levels 3 --funl --fb 16
done
