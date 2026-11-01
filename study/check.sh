#!/bin/bash
# seed-1 confirmation of hier candidates + finer rwlr probe
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate  # gt venv
for a in "hier --lr 1e-3 --window 20 --rwlr 0.03 --floor 0.0 --epochs 1" \
         "hier --lr 1e-3 --window 20 --rwlr 0.01 --floor 0.0 --epochs 1" \
         "hier --lr 1e-3 --window 40 --rwlr 0.01 --floor 0.0 --epochs 1" \
         "hier --lr 1e-3 --window 20 --rwlr 0.02 --floor 0.0 --epochs 1" \
         "hier --lr 1e-3 --window 20 --rwlr 0.05 --floor 0.0 --epochs 1"; do
  echo "$a: $(python3 study.py --arm $a --seed 1 --cap 1000000 2>/dev/null | tail -1 | python3 -c "import json,sys; r=json.load(sys.stdin); print(r[\"val_acc\"], r[\"test_acc\"], int(r[\"cost\"]))")"
done
