#!/usr/bin/env bash
# PHASE 2 driver: two ExaModels/MadNLP instances, one per GPU, each resumable
# (restart after a watchdog SIGKILL on a 30-min compile timeout).
set -u
cd "$(dirname "$0")/.."
RD=examples/results2
mkdir -p "$RD"
NINST=2

run_instance() {  # $1=gpu_id  $2=idx
  local gpu=$1 idx=$2
  for attempt in $(seq 1 50); do
    echo "#### exa instance $idx (GPU $gpu) attempt $attempt ####"
    julia --project=. -t 1 examples/bench2_exa.jl "$gpu" "$NINST" "$idx"
    local code=$?
    echo "#### instance $idx exit $code (attempt $attempt) ####"
    [ $code -eq 0 ] && { echo "#### instance $idx DONE ####"; return 0; }
  done
  echo "#### instance $idx hit max attempts ####"
}

run_instance 0 0 > "$RD/exa_inst0.log" 2>&1 &
run_instance 1 1 > "$RD/exa_inst1.log" 2>&1 &
wait
echo "PHASE2_DONE"
