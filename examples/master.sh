#!/usr/bin/env bash
# Master pipeline: wait for the (already running) ExaModels/MadNLP benchmark to
# finish, then run the PEtab.jl/Optim comparison on the solved set, then write the
# combined final report. Designed to be launched once in the background.
set -u
cd "$(dirname "$0")/.."
RD=examples/results

MODELS=(
  Bruno_JExpBot2016 Crauste_CellSystems2017 Boehm_JProteomeRes2014
  Blasi_CellSystems2016 Borghans_BiophysChem1997 Brannmark_JBC2010
  Bertozzi_PNAS2020 Fujita_SciSignal2010 Fiedler_BMCSystBiol2016
  Elowitz_Nature2000 Sneyd_PNAS2002 Weber_BMC2015 Smith_BMCSystBiol2013
  Perelson_Science1996 Rahman_MBS2016 Zhao_QuantBiol2020 Zheng_PNAS2012
  Okuonghae_ChaosSolitonsFractals2020 Schwen_PONE2014 Liu_IFACPapersOnLine2025
  Oliveira_NatCommun2021 Laske_PLOSComputBiol2019 Alkan_SciSignal2018
  Armistead_CellDeathDis2024 Bachmann_MSB2011 Beer_MolBioSystems2014
  Chen_MSB2009 Froehlich_CellSystems2018 Giordano_Nature2020 Isensee_JCB2018
  Lang_PLOSComputBiol2024 Lucarelli_CellSystems2018 Raia_CancerResearch2011
  Raimundez_PCB2020 SalazarCavazos_MBoC2020
)

bench_done() {
  local m f cs ss
  for m in "${MODELS[@]}"; do
    f="$RD/$m.exa.txt"
    [ -e "$f" ] || return 1
    cs=$(grep -h ^compile_status= "$f" | cut -d= -f2)
    ss=$(grep -h ^solve_status=   "$f" | cut -d= -f2)
    case "$cs" in
      ok) case "$ss" in ok|timeout|error) ;; *) return 1;; esac ;;
      timeout|error|missing_yaml) ;;
      *) return 1 ;;
    esac
  done
  return 0
}

echo "[master] waiting for ExaModels/MadNLP benchmark to finish..."
while true; do
  if bench_done; then echo "[master] benchmark complete (all ${#MODELS[@]} terminal)"; break; fi
  # also stop waiting if the benchmark wrapper has exited (e.g., hit max attempts)
  if ! pgrep -f "run_all.sh$" >/dev/null && ! pgrep -f "run_all.sh *$" >/dev/null \
       && ! pgrep -f "examples/benchmarks.jl" >/dev/null; then
    if bench_done; then echo "[master] benchmark complete"; else
      echo "[master] benchmark wrapper gone but not all models terminal; proceeding with what we have"; fi
    break
  fi
  sleep 30
done

echo "[master] running PEtab.jl / Optim comparison on solved models..."
bash examples/run_all.sh compare

echo "[master] generating final report..."
julia --project=. examples/report.jl | tee "$RD/FINAL_REPORT.txt"

echo "[master] ALL DONE"
