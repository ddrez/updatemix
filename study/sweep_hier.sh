#!/bin/bash
# stage B: hierarchy hyperparameter grid on val, equal cost
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
OUT=sweep_hier.jsonl
touch $OUT
for w in 20 40; do
  for rwlr in 0.01 0.03 0.1; do
    for floor in 0.0 0.3; do
      for ep in 1 2; do
        cfg="w$w-rw$rwlr-fl$floor-ep$ep"
        grep -q "\"tag\": \"$cfg\"" $OUT && continue
        line=$(python3 study.py --arm hier --seed 0 --lr 1e-3 --cap 1000000 \
               --window $w --rwlr $rwlr --floor $floor --epochs $ep 2>/dev/null | tail -1)
        python3 - "$cfg" "$line" <<'EOF' >> $OUT
import json, sys
r = json.loads(sys.argv[2]); r["tag"] = sys.argv[1]
print(json.dumps(r))
EOF
        echo "$cfg: $(echo $line | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["val_acc"], r["test_acc"])')"
      done
    done
  done
done
