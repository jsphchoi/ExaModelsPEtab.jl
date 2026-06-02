#!/usr/bin/env bash
# PHASE 1 driver: run bench2_petab.jl over all models in parallel (CPU-only).
# Resumable: models with a terminal result are skipped.
set -u
cd "$(dirname "$0")/.."
RD=examples/results2
mkdir -p "$RD"
PAR=${1:-12}          # max concurrent workers

MODELS=(
  Alkan_SciSignal2018 Armistead_CellDeathDis2024 Bachmann_MSB2011 Beer_MolBioSystems2014
  Bertozzi_PNAS2020 Blasi_CellSystems2016 Boehm_JProteomeRes2014 Borghans_BiophysChem1997
  Brannmark_JBC2010 Bruno_JExpBot2016 Chen_MSB2009 Crauste_CellSystems2017 Elowitz_Nature2000
  Fiedler_BMCSystBiol2016 Froehlich_CellSystems2018 Fujita_SciSignal2010 Giordano_Nature2020
  Isensee_JCB2018 Lang_PLOSComputBiol2024 Laske_PLOSComputBiol2019 Liu_IFACPapersOnLine2025
  Lucarelli_CellSystems2018 Okuonghae_ChaosSolitonsFractals2020 Oliveira_NatCommun2021
  Perelson_Science1996 Rahman_MBS2016 Raia_CancerResearch2011 Raimundez_PCB2020
  SalazarCavazos_MBoC2020 Schwen_PONE2014 Smith_BMCSystBiol2013 Sneyd_PNAS2002 Weber_BMC2015
  Zhao_QuantBiol2020 Zheng_PNAS2012
)

terminal() {  # $1 = model ; returns 0 if terminal (skip)
  local f="$RD/$1.petab.txt"; [ -e "$f" ] || return 1
  local cs ss; cs=$(grep -h ^compile_status= "$f"|cut -d= -f2); ss=$(grep -h ^solve_status= "$f"|cut -d= -f2)
  case "$cs" in
    ok) case "$ss" in ok|error|timeout) return 0;; *) return 1;; esac;;
    error|missing_yaml) return 0;;
    *) return 1;;
  esac
}

for m in "${MODELS[@]}"; do
  terminal "$m" && { echo "[skip] $m"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done
  echo "[run ] $m"
  julia --project=. -t 1 examples/bench2_petab.jl "$m" \
        > "$RD/$m.petab.log" 2>&1 &
done
wait
echo "PHASE1_DONE"
