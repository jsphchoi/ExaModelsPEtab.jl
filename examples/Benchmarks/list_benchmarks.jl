# list_benchmarks.jl — canonical benchmark model lists (single source of truth).
# Included by benchmark_examodels.jl, benchmark_petab.jl, and final_report.jl so the three
# scripts never drift. Three NESTED canonical lists (1 ⊇ 2 ⊇ 3):
#
#   1. BENCHMARK_MODELS (35)     — every PEtab benchmark-collection model attempted.
#   2. PETAB_SOLVED_MODELS (28)  — those for which PEtab.jl (Optim.IPNewton) reached an
#                                  optimum (petab_optimum_found=true). The 7 dropped are
#                                  PEtab.jl-side failures, NOT ExaModelsPEtab failures:
#                                    Alkan, Raia, Weber — PEtab codegen errors
#                                    Chen, Froehlich    — PEtab compile intractable on this box (capped ~19.6h, never finished)
#                                    Lang               — compiled only on a longer rerun (2.65h), then PEtab solve errored (obj=Inf)
#                                    Smith              — PEtab did not converge (opt_status=false)
#   3. EXA_SUPPORTED_MODELS (26) — the subset of (2) ExaModelsPEtab's collocation transcription
#                                  can represent. Drops the 2 still-unsupported event classes:
#                                    Beer — event TIME is an estimated parameter
#                                    Liu  — state-triggered event (U < 1e-8)
#                                  (Smith, the 3rd unsupported event model, is already absent
#                                   from (2): PEtab itself did not converge it.)
#
# WARMUP note: Bruno_JExpBot2016 is the shared JIT-warmup model for both benchmark scripts and
# is therefore excluded from their TIMED loops — a model warmed-up-on then benchmarked by the
# same process gets an invalid, pre-warmed compile time. Bruno is benchmarked on its own
# (warmed on Crauste) by benchmark_bruno.jl; Crauste is likewise benchmarked outside the
# examodels loop. So EXA_SUPPORTED_MODELS minus {Bruno, Crauste} = the 24 timed in-loop models.

const BENCHMARK_MODELS = [
    "Alkan_SciSignal2018", "Armistead_CellDeathDis2024", "Bachmann_MSB2011",
    "Beer_MolBioSystems2014", "Bertozzi_PNAS2020", "Blasi_CellSystems2016",
    "Boehm_JProteomeRes2014", "Borghans_BiophysChem1997", "Brannmark_JBC2010",
    "Bruno_JExpBot2016", "Chen_MSB2009", "Crauste_CellSystems2017",
    "Elowitz_Nature2000", "Fiedler_BMCSystBiol2016", "Froehlich_CellSystems2018",
    "Fujita_SciSignal2010", "Giordano_Nature2020", "Isensee_JCB2018",
    "Lang_PLOSComputBiol2024", "Laske_PLOSComputBiol2019", "Liu_IFACPapersOnLine2025",
    "Lucarelli_CellSystems2018", "Okuonghae_ChaosSolitonsFractals2020",
    "Oliveira_NatCommun2021", "Perelson_Science1996", "Rahman_MBS2016",
    "Raia_CancerResearch2011", "Raimundez_PCB2020", "SalazarCavazos_MBoC2020",
    "Schwen_PONE2014", "Smith_BMCSystBiol2013", "Sneyd_PNAS2002",
    "Weber_BMC2015", "Zhao_QuantBiol2020", "Zheng_PNAS2012",
]

# PEtab.jl could not produce an optimum for these (see header for the per-model reason).
const _PETAB_FAILED = [
    "Alkan_SciSignal2018", "Chen_MSB2009", "Froehlich_CellSystems2018",
    "Lang_PLOSComputBiol2024", "Raia_CancerResearch2011", "Smith_BMCSystBiol2013",
    "Weber_BMC2015",
]

# Event classes ExaModelsPEtab's collocation cannot yet represent (within the PEtab-solved set).
const _EXA_UNSUPPORTED = ["Beer_MolBioSystems2014", "Liu_IFACPapersOnLine2025"]

const PETAB_SOLVED_MODELS  = filter(m -> m ∉ _PETAB_FAILED,    BENCHMARK_MODELS)     # 28
const EXA_SUPPORTED_MODELS = filter(m -> m ∉ _EXA_UNSUPPORTED, PETAB_SOLVED_MODELS)  # 26

@assert length(BENCHMARK_MODELS)     == 35
@assert length(PETAB_SOLVED_MODELS)  == 28
@assert length(EXA_SUPPORTED_MODELS) == 26
