#!/bin/bash
# Mitigation 3 (caveat: one task): replicate the equal-FLOPS comparison on
# three more GLUE-style classification tasks (paraphrase, entailment, topic)
# with the SAME frozen recipe and lr — no per-task tuning.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate
source mitigate.sh
OUT=tasks.jsonl
touch $OUT
for task in mrpc rte agnews; do
  for seed in 0 1 2 3; do
    runone "$task-plain-$seed" --task $task --arm plain --seed $seed \
      --lr 1e-3 --cap 1000000
    runone "$task-uni-$seed" --task $task --arm traj --seed $seed --lr 1e-3 \
      --cap 1000000 --frw 0 --levels 3
    runone "$task-fval-$seed" --task $task --arm traj --seed $seed --lr 1e-3 \
      --cap 980000 --horcap 1000000 --frw 2 --frlr 0.1 --levels 3 --fval --fb 16
  done
done
