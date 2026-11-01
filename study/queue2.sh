#!/bin/bash
# Second mitigation wave: strict-cost-parity --funl, --funl on all tasks
# and at 1.5B, and scale to 3B. Resumable like queue.sh.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate
echo "=== funl2  $(date) ===";     bash funl2.sh
echo "=== funl15 $(date) ===";     bash funl15.sh
echo "=== tasksfunl $(date) ===";  bash tasksfunl.sh
echo "=== scale3 $(date) ===";     bash scale3.sh
echo "ALLDONE2 $(date)"
