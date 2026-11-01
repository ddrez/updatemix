#!/bin/bash
# TrajectoryMix tuning on the VAL split, standard regime (bs16 lr1e-3 cap 1e6).
# Baseline: plain val .900 (seed0) / .890 (seed1), test mean .9019.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
OUT=sweep_traj.jsonl
touch $OUT
SEED=${SEED:-0}
for a in \
  "--frw 0" \
  "--frw 1 --frlr 0.05" \
  "--frw 0 --tail 16" \
  "--frw 0 --tail 4" \
  "--frw 0 --alpha 0.7" \
  "--frw 0 --stride 8" \
  "--frw 0 --levels 3" \
  "--frw 1 --frlr 0.02 --fb 16" ; do
  key="traj-$SEED-$a"
  key=$(echo $key | tr -d ' -')
  grep -q "\"key\": \"$key\"" $OUT && continue
  line=$(python3 study.py --arm traj --seed $SEED --lr 1e-3 --cap 1000000 $a 2>/dev/null | tail -1)
  python3 - "$key" "$line" <<'EOF' >> $OUT
import json, sys
r = json.loads(sys.argv[2]); r["key"] = sys.argv[1]
print(json.dumps(r))
EOF
  echo "$key val=$(echo $line | python3 -c 'import json,sys;r=json.load(sys.stdin);print(r["val_acc"],r["test_acc"],int(r["cost"]))')"
done
