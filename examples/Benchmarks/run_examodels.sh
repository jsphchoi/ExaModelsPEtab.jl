#!/usr/bin/env bash
# run_examodels.sh — THE driver for the ExaModels/MadNLP-GPU benchmark over the in-loop
# target models (EXA_RERUN_INLOOP in options.jl, or BENCH_SUBSET). Auto-restarts after a
# watchdog SIGKILL (exit 137). Exa-only; Bruno/Crauste (the JIT-warmup models) are benchmarked
# separately via run_bruno.jl, and PEtab via run_petab.sh. Results -> Benchmarks/results/
# (resumable); logs -> examples/debugging/logs/.
#
# Solver/benchmark config (K, SGM_N, tols, limits) lives in options.jl — the single source of
# truth shared with run_petab.jl/run_bruno.jl. Edit it THERE. The only runtime env knob is
# BENCH_SUBSET (comma-separated model list; defaults to EXA_RERUN_INLOOP).
#
# DEFAULT is 2 instances strided across BOTH GPUs (0+1) — that's the intended, fastest mode. A
# pre-flight check below reports each GPU's free memory (via CUDA.jl; nvidia-smi is broken on this
# box) and aborts if a benchmark is already running, so you can see the GPU0 tenant (sushin) before
# committing. Drop to a single GPU only when one is busy or you need contention-free timings.
#
# Usage (from repo root):
#   bash examples/Benchmarks/run_examodels.sh            # 2 instances strided across GPUs 0+1 (DEFAULT, fastest)
#   bash examples/Benchmarks/run_examodels.sh 1 1        # single instance on GPU 1 (one GPU busy / clean timings)
#   FORCE=1 bash examples/Benchmarks/run_examodels.sh    # skip the pre-flight abort-on-existing-run guard
#   BENCH_SUBSET=Zheng_PNAS2012 bash examples/Benchmarks/run_examodels.sh   # run an ad-hoc subset
set -u
cd "$(dirname "$0")/../.."
RD=examples/Benchmarks/results
LD=examples/debugging/logs
mkdir -p "$RD" "$LD"
NINST=${1:-2}          # number of GPU instances: 2 = strided across GPU 0+1 (default), 1 = single-GPU
GPU=${2:-1}            # GPU id for the single-instance (NINST=1) case; default 1 to avoid the GPU0 tenant

# ── pre-flight: don't collide with an existing run; report GPU state ─────────────
if pgrep -f "run_examodels.jl" >/dev/null 2>&1; then
    echo "WARNING: a run_examodels.jl run is already active:"; pgrep -af "run_examodels.jl"
    [ "${FORCE:-0}" = 1 ] || { echo "Aborting (set FORCE=1 to override)."; exit 1; }
fi
echo "Pre-flight GPU free memory (CUDA.jl):"
timeout 120 julia --project=examples -e 'using CUDA
    for i in 0:length(CUDA.devices())-1; CUDA.device!(i)
        println("  GPU ", i, " ", CUDA.name(CUDA.device()), ": free=",
                round(CUDA.free_memory()/2^30; digits=1), " / ",
                round(CUDA.total_memory()/2^30; digits=1), " GiB")
    end' 2>/dev/null || echo "  (CUDA query unavailable — proceed with caution; check for the GPU0 tenant manually)"

run_instance() {  # $1=gpu_id  $2=instance_idx  $3=ninst
    local gpu=$1 idx=$2 ninst=$3
    for attempt in $(seq 1 100); do
        echo "#### exa instance $idx/$ninst (GPU $gpu) attempt $attempt $(date) ####"
        julia --project=examples -t 1 examples/Benchmarks/run_examodels.jl "$gpu" "$ninst" "$idx"
        local code=$?
        echo "#### instance $idx exit $code (attempt $attempt) ####"
        [ $code -eq 0 ] && { echo "#### instance $idx DONE ####"; return 0; }
        echo "Non-zero exit (likely watchdog SIGKILL=137); resuming..."
    done
    echo "#### instance $idx hit max attempts ####"
}

if [ "$NINST" -eq 1 ]; then
    run_instance "$GPU" 0 1 > "$LD/exa_inst0.log" 2>&1
else
    run_instance 0 0 2 > "$LD/exa_inst0.log" 2>&1 &
    run_instance 1 1 2 > "$LD/exa_inst1.log" 2>&1 &
    wait
fi
echo "ALL INSTANCES DONE $(date)"
