#!/bin/bash
# Round 2: confirm the levels-3 (long-horizon endpoint average) win on seed 1
# and probe the horizon direction. Plain baseline seed1: val .890.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
OUT=sweep_traj.jsonl
touch $OUT
SEED=${SEED:-1}
for a in \
  "--frw 0 --levels 3" \
  "--frw 0 --levels 4" \
  "--frw 0 --levels 3 --tail 4" \
  "--frw 0 --levels 3 --stride 8" \
  "--frw 1 --frlr 0.02 --levels 3" \
  "--frw 0 --levels 2 --tw 6" ; do
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
