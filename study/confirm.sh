#!/bin/bash
# Mitigation 1 (caveat: hyperparameters selected on val using seeds ~0-1):
# pre-registered-style confirmation of the FROZEN winning configs on six
# seeds never used for any selection decision. Plain = published tuned lr.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate
source mitigate.sh
OUT=confirm.jsonl
touch $OUT
for seed in 6 7 8 9 10 11; do
  runone "plain-$seed" --arm plain --seed $seed --lr 1e-3 --cap 1000000
  runone "uni-$seed" --arm traj --seed $seed --lr 1e-3 --cap 1000000 \
    --frw 0 --levels 3
  runone "fval-$seed" --arm traj --seed $seed --lr 1e-3 --cap 980000 \
    --horcap 1000000 --frw 2 --frlr 0.1 --levels 3 --fval --fb 16
done
