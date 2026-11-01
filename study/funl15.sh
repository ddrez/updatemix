#!/bin/bash
# --funl (label-free scoring) at 1.5B: does the label-free variant hold
# across scale? Compared against scale.jsonl's plain/uni/fval at 1.5B.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate
source mitigate.sh
M=${SCALE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
OUT=funl15.jsonl
touch $OUT
for seed in 0 1 2 3; do
  runone "funl-$seed" --model "$M" --arm traj --seed $seed --lr 1e-3 \
    --cap 967300 --horcap 1000000 --frw 2 --frlr 0.1 --levels 3 --funl --fb 16
done
