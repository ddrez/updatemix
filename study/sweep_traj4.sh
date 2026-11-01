#!/bin/bash
# Round 3b: fval configs missing from sweep_traj3 (transient OOM crashes —
# llama-server holds 11.9GB; runs retry up to 3x). Plus a cheaper fb=4 probe.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
OUT=sweep_traj.jsonl
touch $OUT
runone() {
  local key="$1"; shift
  grep -q "\"key\": \"$key\"" $OUT && { echo "skip $key"; return; }
  for try in 1 2 3; do
    line=$(python3 study.py --arm traj --lr 1e-3 --cap 1000000 "$@" 2>/tmp/study_err_$key.log | tail -1)
    echo "$line" | python3 -c 'import json,sys; json.loads(sys.stdin.read()); sys.exit(0)' 2>/dev/null && break
    echo "retry $try $key"
    sleep 5
  done
  python3 - "$key" "$line" <<'EOF' >> $OUT
import json, sys
r = json.loads(sys.argv[2]); r["key"] = sys.argv[1]
print(json.dumps(r))
EOF
  echo "$key val=$(echo $line | python3 -c 'import json,sys;r=json.load(sys.stdin);print(r["val_acc"],r["test_acc"],int(r["cost"]))')"
}
for SEED in 0 1; do
  runone "t3-$SEED-frw1frlr0.1levels3fvalfb8" --seed $SEED --frw 1 --frlr 0.1 --levels 3 --fval --fb 8
  runone "t4-$SEED-frw1frlr0.1levels3fvalfb4" --seed $SEED --frw 1 --frlr 0.1 --levels 3 --fval --fb 4
done
runone "t3-0-frw2frlr0.1levels3fvalfb16" --seed 0 --frw 2 --frlr 0.1 --levels 3 --fval --fb 16
runone "t3-0-frw1frlr0.1levels3fvalfb8decay0.7" --seed 0 --frw 1 --frlr 0.1 --levels 3 --fval --fb 8 --decay 0.7
