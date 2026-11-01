#!/bin/bash
# Round-3 finals: paired against the round-2 plain runs already in
# finals2.jsonl. Traj cap is trimmed so counted cost stays <= plain's
# (the val reweighting must pay its ~2% inside the same budget);
# --horcap keeps the cosine schedule identical to plain's.
# Usage: TRAJ_ARGS="--frw 2 --frlr 0.1 --levels 3 --fval --fb 16" TRAJ_CAP=980000 bash finals3.sh
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
OUT=finals3.jsonl
touch $OUT
for seed in 0 1 2 3 4 5; do
  key="traj-$seed-${TRAJ_ARGS}"
  key=$(echo $key | tr -d ' ')
  grep -q "\"key\": \"$key\"" $OUT && continue
  for try in 1 2 3; do
    line=$(python3 study.py --arm traj --seed $seed --lr 1e-3 \
      --cap ${TRAJ_CAP:-980000} --horcap 1000000 ${TRAJ_ARGS} 2>/tmp/f3err.log | tail -1)
    echo "$line" | python3 -c 'import json,sys; json.loads(sys.stdin.read()); sys.exit(0)' 2>/dev/null && break
    echo "retry $try $key"; sleep 5
  done
  python3 - "$key" "$line" <<'EOF' >> $OUT
import json, sys
r = json.loads(sys.argv[2]); r["key"] = sys.argv[1]
print(json.dumps(r))
EOF
  echo "$key done: $(echo $line | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["val_acc"], r["test_acc"], int(r["cost"]))')"
done
