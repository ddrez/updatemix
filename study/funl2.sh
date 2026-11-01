#!/bin/bash
# --funl at strict cost parity: wikitext scoring costs ~32.9k counted
# tokens (16x16 seqs x 64 tok x 2 epochs), so the training cap drops to
# 967300 to keep total ~= plain's ~1000200. Seeds 0-5.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate
source mitigate.sh
OUT=funl2.jsonl
touch $OUT
for seed in 0 1 2 3 4 5; do
  runone "funl2-$seed" --arm traj --seed $seed --lr 1e-3 \
    --cap 967300 --horcap 1000000 --frw 2 --frlr 0.1 --levels 3 --funl --fb 16
done
