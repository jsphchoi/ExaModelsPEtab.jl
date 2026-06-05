#!/usr/bin/env bash
# benchmark_examodels.sh — runs two ExaModels/MadNLP benchmark instances in parallel,
# one per GPU. Each instance handles a strided partition of the ExaModels-supported models
# (ALL_MODELS in benchmark_examodels.jl) and is automatically restarted after a watchdog
# SIGKILL (exit 137) on a compile timeout. Result txts land in examples/Benchmarks/results/
# (resumable across restarts); run logs go to examples/debugging/logs/.
#
# Usage (from repo root):
#   bash examples/Benchmarks/benchmark_examodels.sh
set -u
cd "$(dirname "$0")/../.."
RD=examples/Benchmarks/results
LD=examples/debugging/logs
mkdir -p "$RD" "$LD"

run_instance() {  # $1=gpu_id  $2=instance_idx
    local gpu=$1 idx=$2
    for attempt in $(seq 1 100); do
        echo "#### exa instance $idx (GPU $gpu) attempt $attempt ####"
        julia --project=. -t 1 examples/Benchmarks/benchmark_examodels.jl "$gpu" 2 "$idx"
        local code=$?
        echo "#### instance $idx exit $code (attempt $attempt) ####"
        [ $code -eq 0 ] && { echo "#### instance $idx DONE ####"; return 0; }
        echo "Non-zero exit (likely watchdog SIGKILL=137); resuming..."
    done
    echo "#### instance $idx hit max attempts ####"
}

run_instance 0 0 > "$LD/exa_inst0.log" 2>&1 &
run_instance 1 1 > "$LD/exa_inst1.log" 2>&1 &
wait
echo "ALL INSTANCES DONE"
