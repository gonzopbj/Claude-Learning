#!/usr/bin/env bash
# Reproduce every result in REPORT.md from scratch (SPEC.md "Compute budget").
#
#   bash run_all.sh            # ~15-20 minutes on 4 cores
#
# Stage 0  calibrate at B = 50 on validation seeds 1000-1009 -> lambda*, W, gamma, tau
# Stage 1  per-budget calibrations at B in {10, 25, 100} reusing lambda* (one detector
#          threshold for every agent, SPEC "Tuning"; W and tau tuned at the evaluated budget)
# Stage 2  main run, budget sweep, hidden-confounder variant, knob grid
# Stage 3  report.py -> metrics.json, figures/, REPORT.md
set -euo pipefail
cd "$(dirname "$0")"
W=${WORKERS:-4}
R=results
mkdir -p "$R"

log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

if [ ! -f "$R/constants_B50.json" ]; then
  log "stage 0: calibrate B=50"
  python3 calibrate.py --seeds 1000-1009 --budget 50 --workers "$W" --out "$R/constants_B50.json"
fi
LAM=$(python3 -c "import json;print(json.load(open('$R/constants_B50.json'))['lambda'])")
log "lambda* = $LAM"

for B in 10 25 100; do
  if [ ! -f "$R/constants_B$B.json" ]; then
    log "stage 1: calibrate B=$B (fixed lambda)"
    python3 calibrate.py --seeds 1000-1009 --budget "$B" --workers "$W" \
      --fixed-lambda "$LAM" --out "$R/constants_B$B.json"
  fi
done

log "stage 2a: main run (13 agents x 20 seeds x 60 episodes, B=50)"
python3 run.py --agents main --seeds 0-19 --budget 50 --workers "$W" \
  --constants "$R/constants_B50.json" --out "$R/main"

for B in 10 25 100; do
  log "stage 2b: budget sweep B=$B"
  python3 run.py --agents sweep --seeds 0-19 --budget "$B" --workers "$W" \
    --constants "$R/constants_B$B.json" --out "$R/sweep_B$B"
done

log "stage 2c: hidden-confounder variant (40 seeds)"
python3 run.py --agents hidden --hidden --seeds 0-39 --budget 50 --workers "$W" \
  --constants "$R/constants_B50.json" --out "$R/hidden"

log "stage 2d: knob grid (shift type x heteroscedasticity, minus the main configuration)"
for cfg in "large 1" "small 0" "small 1" "noise-only 0" "noise-only 1"; do
  set -- $cfg
  python3 run.py --agents knobs --shift-type "$1" --hetero "$2" --seeds 0-19 --budget 50 \
    --workers "$W" --constants "$R/constants_B50.json" --out "$R/knob_${1}_h$2"
done

log "stage 3: report"
python3 report.py "$R/main" "$R/sweep_B10" "$R/sweep_B25" "$R/sweep_B100" "$R/hidden" \
  "$R"/knob_* --constants "$R/constants_B50.json" --out "$R/report"
log "done: $R/report/REPORT.md"
