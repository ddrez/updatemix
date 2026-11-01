#!/bin/bash
# Mitigation 4 (caveat: one model size): scale to SCALE_MODEL (default
# Qwen2.5-1.5B-Instruct path/HF id). Fair protocol: the plain baseline
# gets a 3-point lr sweep (1 seed, pick best val) at the new scale; the
# mixture arms keep the FROZEN 0.5B recipe unchanged (no new tuning).
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate
source mitigate.sh
M=${SCALE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
OUT=scale.jsonl
touch $OUT
# stage 1: plain lr selection at the new scale (val only)
i=0
for lr in 3e-4 1e-3 3e-3; do
  i=$((i+1))
  runone "sweeplr$i-0" --model "$M" --arm plain --seed 0 --lr $lr --cap 1000000
done
PLAIN_LR=$(python3 - "$OUT" <<'EOF'
import json, sys
best, bl = -1, 1e-3
for ln in open(sys.argv[1]):
    r = json.loads(ln)
    if r["key"].startswith("sweeplr") and r["val_acc"] > best:
        best, bl = r["val_acc"], r["lr"]
print(bl)
EOF
)
echo "plain lr at scale: $PLAIN_LR"
for seed in 0 1 2 3; do
  runone "plain-$seed" --model "$M" --arm plain --seed $seed --lr $PLAIN_LR \
    --cap 1000000
  runone "uni-$seed" --model "$M" --arm traj --seed $seed --lr 1e-3 \
    --cap 1000000 --frw 0 --levels 3
  runone "fval-$seed" --model "$M" --arm traj --seed $seed --lr 1e-3 \
    --cap 980000 --horcap 1000000 --frw 2 --frlr 0.1 --levels 3 --fval --fb 16
done
