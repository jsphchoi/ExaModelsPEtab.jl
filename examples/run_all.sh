#!/usr/bin/env bash
# Drives benchmarks.jl (and optionally compare.jl) to completion, restarting after
# a watchdog SIGKILL (exit 137) on a compile/solve timeout. Each run is resumable:
# finished models are skipped, so progress always moves forward.
#
# Usage:
#   examples/run_all.sh                 # run ExaModels/MadNLP benchmarks
#   examples/run_all.sh compare         # run PEtab.jl/Optim comparison
set -u
cd "$(dirname "$0")/.."

SCRIPT="examples/benchmarks.jl"
[ "${1:-}" = "compare" ] && SCRIPT="examples/compare.jl"

MAX_ATTEMPTS=50
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo ""
    echo "######## $SCRIPT — attempt $attempt ########"
    julia --project=. -t 1 "$SCRIPT"
    code=$?
    echo "######## $SCRIPT exited with code $code (attempt $attempt) ########"
    if [ $code -eq 0 ]; then
        echo "DONE: $SCRIPT completed normally."
        exit 0
    fi
    echo "Non-zero exit (likely watchdog SIGKILL=137); resuming..."
done
echo "Reached MAX_ATTEMPTS=$MAX_ATTEMPTS without normal completion."
exit 1
