#!/bin/bash
# Round 3c: small grid around the fval winner (fb16 frw2 frlr0.1, val mean
# .915 vs .8983 for uniform L3). Selection on val seeds 0/1 only.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
OUT=sweep_traj.jsonl
touch $OUT
runone() {
  local key="$1"; shift
  grep -q "\"key\": \"$key\"" $OUT && { echo "skip $key"; return; }
  for try in 1 2 3; do
    line=$(python3 study.py --arm traj --lr 1e-3 --cap 1000000 "$@" 2>/tmp/err_$key.log | tail -1)
    echo "$line" | python3 -c 'import json,sys; json.loads(sys.stdin.read()); sys.exit(0)' 2>/dev/null && break
    echo "retry $try $key"; sleep 5
  done
  python3 - "$key" "$line" <<'EOF' >> $OUT
import json, sys
r = json.loads(sys.argv[2]); r["key"] = sys.argv[1]
print(json.dumps(r))
EOF
  echo "$key val=$(echo $line | python3 -c 'import json,sys;r=json.load(sys.stdin);print(r["val_acc"],r["test_acc"],int(r["cost"]))')"
}
for SEED in 0 1; do
  runone "t5-$SEED-frw2fb16frlr0.05" --seed $SEED --frw 2 --frlr 0.05 --levels 3 --fval --fb 16
  runone "t5-$SEED-frw2fb16frlr0.2" --seed $SEED --frw 2 --frlr 0.2 --levels 3 --fval --fb 16
  runone "t5-$SEED-frw3fb16" --seed $SEED --frw 3 --frlr 0.1 --levels 3 --fval --fb 16
  runone "t5-$SEED-frw2fb16lvl2" --seed $SEED --frw 2 --frlr 0.1 --levels 2 --fval --fb 16
done
