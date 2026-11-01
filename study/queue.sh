#!/bin/bash
# Sequential driver for the four caveat-mitigation experiments (one GPU).
# Resumable: every run is keyed in its .jsonl; rerun to continue.
cd "$(dirname "$0")" || exit 1
[ -f /venv/bin/activate ] && . /venv/bin/activate
echo "=== confirm $(date) ==="; bash confirm.sh
echo "=== unl $(date) ===";    bash unl.sh
echo "=== tasks $(date) ===";  bash tasks.sh
echo "=== scale $(date) ===";  bash scale.sh
echo "ALLDONE $(date)"
