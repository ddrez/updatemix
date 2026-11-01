#!/bin/bash
# Stress regime (high gradient noise): bs4, short budget, several lrs.
# Selection stays on val; traj config is the one fixed from the standard
# regime sweep (frw 0, levels 3) — no new tuning here.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
OUT=stress.jsonl
touch $OUT
for lr in 1e-3 3e-3 1e-2; do
  for seed in 0 1; do
    for arm in "plain --lr $lr --bs 4 --cap 250000" "traj --lr $lr --bs 4 --cap 250000 --frw 0 --levels 3 --stride 8"; do
      set -- $arm; a=$1; shift
      key="s-$a-$lr-$seed"
      grep -q "\"key\": \"$key\"" $OUT && continue
      line=$(python3 study.py --arm $a --seed $seed "$@" 2>/dev/null | tail -1)
      python3 - "$key" "$line" <<'EOF' >> $OUT
import json, sys
r = json.loads(sys.argv[2]); r["key"] = sys.argv[1]
print(json.dumps(r))
EOF
      echo "$key done: $(echo $line | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["val_acc"], r["test_acc"], int(r["cost"]))')"
    done
  done
done
