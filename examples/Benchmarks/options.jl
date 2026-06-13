# options.jl — canonical benchmark model lists (single source of truth).
# Included by run_examodels.jl, run_petab.jl, and results.jl so the three
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
#   3. EXA_SUPPORTED_MODELS (25) — the subset of (2) ExaModelsPEtab's collocation transcription
#                                  can represent. Drops the 2 unsupported event classes + Fiedler:
#                                    Beer    — event TIME is an estimated parameter
#                                    Liu     — state-triggered event (U < 1e-8)
#                                    Fiedler — initial condition is an arbitrary function (z0_func)
#                                              that overflows the kernel parameter budget
#                                  (Smith, the 3rd unsupported event model, is already absent
#                                   from (2): PEtab itself did not converge it.)
#
# WARMUP note: Bruno_JExpBot2016 is the shared JIT-warmup model for both benchmark scripts and
# is therefore excluded from their TIMED loops — a model warmed-up-on then benchmarked by the
# same process gets an invalid, pre-warmed compile time. Bruno is benchmarked on its own (warmed
# on Crauste) by run_bruno.jl. Crauste needs NO such exception: it is warmed by Bruno like
# every other in-loop model, so it is timed normally in the main loops. Only Bruno is special —
# EXA_SUPPORTED_MODELS minus {Bruno} = the 24 timed in-loop models.

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK + SOLVER CONFIGURATION — single source of truth.
# run_examodels.jl, run_petab.jl, and results.jl ALL read these so
# both backends run under IDENTICAL settings. CHANGE SETTINGS HERE, NOWHERE ELSE.
# ══════════════════════════════════════════════════════════════════════════════
# ── Shared (apply to BOTH backends) ──
const BENCH_SGM_N       = 5            # SGM warm-rerun count for the geometric-mean solve timing (exa & PEtab; 0 disables)
const BENCH_TOL         = 1e-6         # convergence tolerance — MadNLP `tol` == Optim `g_tol`
const BENCH_SOLVE_LIMIT = 3600.0       # solve wall cap [s] (1 hr) — MadNLP `max_wall_time` / Optim `time_limit`. No successful run (any backend) has ever exceeded 1 hr.
const BENCH_MAX_ITER    = 100_000_000  # max solver iterations (large so wall time is the bottleneck)
const BENCH_WARMUP_MODEL = "Bruno_JExpBot2016"  # shared JIT-warmup model; excluded from both timed loops

# ── ExaModels / MadNLP-only ──
const BENCH_K             = 4          # collocation points per mesh interval
const BENCH_COMPILE_LIMIT = 3600.0     # exa build wall cap [s] (1 hr)
# acceptable-level termination: accept an ε-optimal KKT point when the strict tol can't be reached
# (boundary optima / ill-conditioning floor inf_du just above tol). 1e-4 = 100× looser than tol;
# certifies true-but-boundary optima (Schwen) while rejecting genuinely non-converged solves.
const BENCH_ACCEPT_TOL    = 1e-4       # MadNLP acceptable_tol (ε-optimal fallback) → SOLVED_TO_ACCEPTABLE_LEVEL
const BENCH_ACCEPT_ITER   = 15         # iters at acceptable_tol before accepting

# ── PEtab / Optim.IPNewton-only ──
const BENCH_PETAB_COMPILE_LIMIT   = 3600.0  # PEtab build wall cap [s] (1 hr — canonical compile cap, same as exa)
const BENCH_PETAB_F_RELTOL        = 1e-8    # Optim.Options fine-tuning ↓
const BENCH_PETAB_SUCCESSIVE_FTOL = 3
const BENCH_PETAB_X_ABSTOL        = 0.0
# ══════════════════════════════════════════════════════════════════════════════

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

# Models ExaModelsPEtab's collocation cannot yet represent (within the PEtab-solved set):
#   Beer — event TIME is an estimated parameter; Liu — state-triggered event (U < 1e-8);
#   Fiedler — initial condition is an arbitrary function (z0_func) overflowing the kernel
#             parameter budget (accepted-unsupported).
const _EXA_UNSUPPORTED = ["Beer_MolBioSystems2014", "Liu_IFACPapersOnLine2025", "Fiedler_BMCSystBiol2016"]

const PETAB_SOLVED_MODELS  = filter(m -> m ∉ _PETAB_FAILED,    BENCHMARK_MODELS)     # 28
const EXA_SUPPORTED_MODELS = filter(m -> m ∉ _EXA_UNSUPPORTED, PETAB_SOLVED_MODELS)  # 25

@assert length(BENCHMARK_MODELS)     == 35
@assert length(PETAB_SOLVED_MODELS)  == 28
@assert length(EXA_SUPPORTED_MODELS) == 25

# ── K=3 full-suite rerun order ───────────────────────────────────────────────────
# The complete exa-supported in-loop set = EXA_SUPPORTED_MODELS minus Bruno (the shared JIT
# warmup, benchmarked separately by run_bruno.jl). All 24 are run; the ORDER is chosen for
# fast, useful feedback (run_examodels.jl preserves it, strided across the GPU instances):
#   1. Models that CONVERGED in the prior K=4 run (term_status SOLVE_SUCCEEDED or
#      SOLVED_TO_ACCEPTABLE_LEVEL — the 0 / 0S / 0A / 0AS scoreboard codes), ordered by NLP
#      size (exa_nvar) ASCENDING, so the cheap high-confidence results land first.
#   2. The remaining (non-converged) models, ordered by prior K=4 compile time ASCENDING, so
#      the long compiles / likely-failures don't block the rest.
# Sizes/compile-times are the K=4 snapshot in results/ at the time this order was set; at K=3
# the absolute numbers shrink but the relative ordering is a good proxy.
const EXA_RERUN_INLOOP = [
    # ── converged in K=4, by nvar ascending ──
    "Blasi_CellSystems2016", "Armistead_CellDeathDis2024", "Perelson_Science1996",
    "Okuonghae_ChaosSolitonsFractals2020", "Rahman_MBS2016", "Boehm_JProteomeRes2014",
    "Bertozzi_PNAS2020", "Zheng_PNAS2012", "Oliveira_NatCommun2021",
    "Crauste_CellSystems2017", "Sneyd_PNAS2002", "Zhao_QuantBiol2020",
    "Fujita_SciSignal2010", "SalazarCavazos_MBoC2020", "Schwen_PONE2014",
    "Laske_PLOSComputBiol2019",
    # ── not converged in K=4, by compile time ascending ──
    "Borghans_BiophysChem1997", "Elowitz_Nature2000", "Brannmark_JBC2010",
    "Giordano_Nature2020", "Isensee_JCB2018", "Lucarelli_CellSystems2018",
    "Raimundez_PCB2020", "Bachmann_MSB2011",
]
# Bruno is the ONLY model benchmarked outside the in-loop scripts (it is their shared warmup),
# via run_bruno.jl. Crauste needs no exception — it is warmed by Bruno like every other
# in-loop model — so it lives in EXA_RERUN_INLOOP above.
const EXA_RERUN_MODELS = [EXA_RERUN_INLOOP; "Bruno_JExpBot2016"]  # 25 (in-loop 24 + Bruno)

@assert length(EXA_RERUN_INLOOP) == 24
@assert length(EXA_RERUN_MODELS) == 25
@assert all(m -> m ∈ EXA_SUPPORTED_MODELS, EXA_RERUN_MODELS)
@assert sort(EXA_RERUN_MODELS) == sort(EXA_SUPPORTED_MODELS)  # the rerun covers the full supported set
