#!/usr/bin/env bash
# Orchestrates the Ipopt-filtered benchmark: PEtab/Ipopt phase -> ExaModels/MadNLP
# phase (2 GPUs) -> combined report.
set -u
cd "$(dirname "$0")/.."
RD=examples/results2
mkdir -p "$RD"

echo "[master2] $(date) PHASE 1: PEtab/Optim.IPNewton ..."
bash examples/run_petab.sh 12
echo "[master2] $(date) PHASE 1 complete"

echo "[master2] $(date) PHASE 2: ExaModels/MadNLP on 2 GPUs ..."
bash examples/run_exa.sh
echo "[master2] $(date) PHASE 2 complete"

echo "[master2] $(date) generating report ..."
julia --project=. examples/report2.jl | tee "$RD/FINAL_REPORT2.txt"
echo "[master2] ALL DONE $(date)"
