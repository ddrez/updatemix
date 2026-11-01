#!/bin/bash
# Round 3: does the mixture beat plain SWA? Two new levers:
#  (a) geometric recency prior (`--decay`), still zero extra FLOPS;
#  (b) learned weights scored on the held-out val split (`--fval`) —
#      model selection, not replay memorization. Baseline: L3 frw0
#      (round-2 winner) val .910/.8867 on seeds 0/1; plain .900/.890.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
OUT=sweep_traj.jsonl
touch $OUT
for SEED in 0 1; do
for a in \
  "--frw 0 --levels 3 --decay 0.7" \
  "--frw 0 --levels 3 --decay 0.5" \
  "--frw 1 --frlr 0.1 --levels 3 --fval --fb 8" \
  "--frw 2 --frlr 0.1 --levels 3 --fval --fb 16" \
  "--frw 1 --frlr 0.1 --levels 3 --fval --fb 8 --decay 0.7" ; do
  key="t3-$SEED-$(echo $a | tr -d ' -')"
  grep -q "\"key\": \"$key\"" $OUT && continue
  line=$(python3 study.py --arm traj --seed $SEED --lr 1e-3 --cap 1000000 $a 2>/dev/null | tail -1)
  python3 - "$key" "$line" <<'EOF' >> $OUT
import json, sys
r = json.loads(sys.argv[2]); r["key"] = sys.argv[1]
print(json.dumps(r))
EOF
  echo "$key val=$(echo $line | python3 -c 'import json,sys;r=json.load(sys.stdin);print(r["val_acc"],r["test_acc"],int(r["cost"]))')"
done
done
