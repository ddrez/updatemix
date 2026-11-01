#!/bin/bash
# --funl on the three extra tasks (label-free scoring, external text is
# task-independent): does the no-annotation variant replicate beyond SST-2?
# Appends {task}-funl-{seed} lines to tasks.jsonl. Cap 967300 for parity.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate
source mitigate.sh
OUT=tasks.jsonl
touch $OUT
for task in mrpc rte agnews; do
  for seed in 0 1 2 3; do
    runone "$task-funl-$seed" --task $task --arm traj --seed $seed --lr 1e-3 \
      --cap 967300 --horcap 1000000 --frw 2 --frlr 0.1 --levels 3 --funl --fb 16
  done
done
