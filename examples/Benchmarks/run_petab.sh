#!/usr/bin/env bash
# run_petab.sh — runs run_petab.jl for every benchmark model (minus the Bruno
# JIT warmup, which run_bruno.jl covers) in parallel. Each model is a separate Julia
# process. Result txts go to Benchmarks/results/ (resumable — models with a terminal petab
# result are skipped); per-model run logs go to debugging/logs/.
#
# The MODELS array below is the canonical BENCHMARK_MODELS list (options.jl) minus Bruno.
#
# Usage (from repo root):
#   bash examples/Benchmarks/run_petab.sh [max_parallel_workers]
# Default: 12 concurrent workers.
set -u
cd "$(dirname "$0")/../.."
RD=examples/Benchmarks/results
LD=examples/debugging/logs
mkdir -p "$RD" "$LD"
PAR=${1:-12}

MODELS=(
    Alkan_SciSignal2018 Armistead_CellDeathDis2024 Bachmann_MSB2011
    Beer_MolBioSystems2014 Bertozzi_PNAS2020 Blasi_CellSystems2016
    Boehm_JProteomeRes2014 Borghans_BiophysChem1997 Brannmark_JBC2010
    Chen_MSB2009 Crauste_CellSystems2017  # Bruno excluded — shared JIT warmup (run_bruno.jl)
    Elowitz_Nature2000 Fiedler_BMCSystBiol2016 Froehlich_CellSystems2018
    Fujita_SciSignal2010 Giordano_Nature2020 Isensee_JCB2018
    Lang_PLOSComputBiol2024 Laske_PLOSComputBiol2019 Liu_IFACPapersOnLine2025
    Lucarelli_CellSystems2018 Okuonghae_ChaosSolitonsFractals2020
    Oliveira_NatCommun2021 Perelson_Science1996 Rahman_MBS2016
    Raia_CancerResearch2011 Raimundez_PCB2020 SalazarCavazos_MBoC2020
    Schwen_PONE2014 Smith_BMCSystBiol2013 Sneyd_PNAS2002
    Weber_BMC2015 Zhao_QuantBiol2020 Zheng_PNAS2012
)

petab_terminal() {  # $1 = model; returns 0 (skip) if petab result is terminal
    local f="$RD/$1_results.txt"
    [ -e "$f" ] || return 1
    local cs ss
    cs=$(grep -h "^petab_compile_status=" "$f" | cut -d= -f2)
    ss=$(grep -h "^petab_solve_status="   "$f" | cut -d= -f2)
    case "$cs" in
        ok)    case "$ss" in ok|error|timeout) return 0;; *) return 1;; esac ;;
        error|missing_yaml) return 0 ;;
        *) return 1 ;;
    esac
}

echo "[run_petab] $(date)  PAR=$PAR"
for m in "${MODELS[@]}"; do
    if petab_terminal "$m"; then
        echo "[skip] $m"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done
    echo "[run ] $m"
    julia --project=examples -t 1 examples/Benchmarks/run_petab.jl "$m" \
        > "$LD/${m}_petab.log" 2>&1 &
done
wait
echo "[run_petab] $(date) ALL DONE"
