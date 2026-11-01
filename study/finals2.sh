#!/bin/bash
# Paired-seed finals: tuned plain vs tuned traj (endpoint-mixture updatemix).
# Usage: TRAJ_ARGS="--frw 1 --frlr .. --tail .." TRAJ_CAP=991000 bash finals2.sh
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
OUT=finals2.jsonl
touch $OUT
for seed in 0 1 2 3 4 5; do
  for arm in "plain --lr 1e-3 --cap 1000000" "traj --lr 1e-3 --cap ${TRAJ_CAP:-991000} ${TRAJ_ARGS}"; do
    set -- $arm; a=$1; shift
    key=$(echo "$a-$seed-$*" | tr -d ' ')
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
