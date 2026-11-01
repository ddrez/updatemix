#!/bin/bash
# Round 4: push past the fval winner (val mean .9150 on seeds 0/1 with
# frw2 frlr0.1 lvl3 fval fb16). Candidates probe: wider candidate pool
# (tail 12 — needs levels 4 to actually offer >8 top-bin points at this
# horizon), more selection data (fb 32), denser snapshots (stride 8),
# slightly hotter weight lr. Selection on val seeds 0/1 only.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
OUT=sweep_traj6.jsonl
touch $OUT
runone() {
  local key="$1"; shift
  grep -q "\"key\": \"$key\"" $OUT && { echo "skip $key"; return; }
  for try in 1 2 3; do
    line=$(python3 study.py --arm traj --lr 1e-3 --cap 980000 --horcap 1000000 --fval "$@" 2>/tmp/e6_$key.log | tail -1)
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
for seed in 0 1; do
  runone "e6-tail12lvl4-$seed" --seed $seed --frw 2 --frlr 0.1 --levels 4 --tail 12 --fb 16
  runone "e6-fb32-$seed"       --seed $seed --frw 2 --frlr 0.1 --levels 3 --fb 32
  runone "e6-stride8-$seed"    --seed $seed --frw 2 --frlr 0.1 --levels 3 --stride 8 --fb 16
  runone "e6-frlr0.15-$seed"   --seed $seed --frw 2 --frlr 0.15 --levels 3 --fb 16
done
echo DONE-ENH
