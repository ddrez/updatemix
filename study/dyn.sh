#!/bin/bash
# Accuracy dynamics: plain vs traj(uniform) vs traj(fval), seeds 0-2,
# --probe 100 (forward-only eval every 100 steps; uncounted, RNG-neutral
# -> curves come from runs bit-identical to the finals). Plus endpoint
# probes: evaluate every mixture candidate + mixtures at flush.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
OUT=dyn.jsonl
touch $OUT
runone() {
  local key="$1"; shift
  grep -q "\"key\": \"$key\"" $OUT && { echo "skip $key"; return; }
  for try in 1 2 3; do
    line=$(python3 study.py --lr 1e-3 --probe 100 "$@" 2>/tmp/derr_$key.log | tail -1)
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
UNI="--arm traj --frw 0 --levels 3"
FVAL="--arm traj --frw 2 --frlr 0.1 --levels 3 --fval --fb 16 --cap 980000 --horcap 1000000"
for seed in 0 1 2; do
  runone "d-plain-$seed" --arm plain --seed $seed --cap 1000000
  runone "d-traju-$seed" $UNI --seed $seed --cap 1000000
  runone "d-trajf-$seed" $FVAL --seed $seed
done
# endpoint spreads (extra evals at flush, uncounted forward passes)
runone "ep-traju-0" $UNI --seed 0 --cap 1000000 --probe_endpoints
runone "ep-trajf-0" $FVAL --seed 0 --probe_endpoints
echo DONE-DYN
