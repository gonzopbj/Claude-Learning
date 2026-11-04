#!/bin/bash
# usage: run_job.sh <plan 0|1> <train.py args...>   -> trains then evaluates, single thread
cd "$(dirname "$0")"
PLAN=$1; shift
OUT=$(python3 train.py --threads 1 --verbose 0 "$@" | grep -- '->' | awk '{print $2}')
python3 evaluate.py "$OUT" --plan "$PLAN" --threads 1 > "$OUT/eval_stdout.txt" 2>&1
echo "done $OUT"
