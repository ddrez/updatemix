#!/bin/bash
# Shared runner for caveat-mitigation experiments. Sourcing-able: defines
# runone KEY ARGS... -> runs study.py with 3 retries, appends the JSON
# result tagged with KEY to the file in $OUT, skipping if already present.
runone() {
  local key="$1"; shift
  grep -q "\"key\": \"$key\"" $OUT 2>/dev/null && { echo "skip $key"; return; }
  for try in 1 2 3; do
    line=$(python3 study.py "$@" 2>/tmp/merr_$key.log | tail -1)
    echo "$line" | python3 -c 'import json,sys; json.loads(sys.stdin.read()); sys.exit(0)' 2>/dev/null && break
    echo "retry $try $key ($(date +%H:%M))"; sleep 5
  done
  python3 - "$key" "$line" <<'EOF' >> $OUT
import json, sys
r = json.loads(sys.argv[2]); r["key"] = sys.argv[1]
print(json.dumps(r))
EOF
  echo "$key -> val=$(echo $line | python3 -c 'import json,sys;r=json.load(sys.stdin);print(r["val_acc"], r["test_acc"], int(r["cost"]))')"
}
