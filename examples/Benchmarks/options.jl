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

# ── K=10 / SGM n=10 rerun target set ────────────────────────────────────────────
# The models on which ExaModelsPEtab reached (or nearly reached) a converged solve, targeted
# for the K=10 + SGM-n=10 head-to-head rerun against PEtab. ORDERED clean-success → least-
# reliable so the high-confidence results land first (run_examodels.jl runs in this order,
# and the unreliable tail's 2 h solve caps don't delay the clean numbers). Membership rationale:
#   Clean SOLVE_SUCCEEDED matching/near PEtab : Boehm, Blasi, Rahman, Sneyd, Perelson, Armistead,
#                                               SalazarCavazos (Fix #10 ov win), Zheng (x0SSpre cv
#                                               fix, +0.83% gap), Bruno, Crauste
#   Converged but suboptimal local min        : Okuonghae (+4.6%), Bertozzi (+194%) — valid KKT pts
#   User-requested retry (tail-stall last run): Schwen (was inf_du~1.2e-6 just above 1e-6 tol)
#   Conditioning long-shots (likely re-fail)  : Lucarelli (dual-stall 0.028), Zhao (+8.2% restoration),
#                                               Laske (+3.6% restoration) — K=10 mostly won't fix
# EXCLUDED (and why): Fujita (Status-0 but FALSE PASS — gate-bug, exa −323.5 vs PEtab −53.08);
#   Bachmann (compiles at K=4 but solver-fails — NOT an OOM); Borghans/Elowitz (NaN-spin);
#   event models (don't build/diverge). (Zheng was here — singular-KKT/unidentifiable — but the
#   x0SSpre cv fix recovered it to a clean SOLVE_SUCCEEDED, so it is now an in-loop rerun target.)
#   (Fiedler is no longer here: it is now in _EXA_UNSUPPORTED, outside the supported set entirely.)
#
# Bruno is the shared JIT-warmup model — excluded from the in-loop scripts (a model warmed-on
# then benchmarked by the same process gets an invalid compile time) and benchmarked on its own
# via run_bruno.jl (warmed on Crauste). Crauste needs no such exception: it is warmed by
# Bruno like every other in-loop model, so it lives in EXA_RERUN_INLOOP. Only Bruno is omitted.
const EXA_RERUN_INLOOP = [   # run_examodels.jl timed loop, clean → unreliable order
    "Boehm_JProteomeRes2014", "Blasi_CellSystems2016", "Rahman_MBS2016",
    "Sneyd_PNAS2002", "Perelson_Science1996", "Armistead_CellDeathDis2024",
    "SalazarCavazos_MBoC2020", "Zheng_PNAS2012", "Crauste_CellSystems2017",
    "Okuonghae_ChaosSolitonsFractals2020",
    "Bertozzi_PNAS2020", "Schwen_PONE2014", "Lucarelli_CellSystems2018",
    "Zhao_QuantBiol2020", "Laske_PLOSComputBiol2019",
]
# Bruno is the ONLY model benchmarked outside the in-loop scripts (it is their shared warmup),
# via run_bruno.jl. Crauste needs no exception — it is warmed by Bruno like every other
# in-loop model — so it lives in EXA_RERUN_INLOOP above.
const EXA_RERUN_MODELS = [EXA_RERUN_INLOOP; "Bruno_JExpBot2016"]  # 16 (in-loop 15 + Bruno)

@assert length(EXA_RERUN_INLOOP) == 15
@assert length(EXA_RERUN_MODELS) == 16
@assert all(m -> m ∈ EXA_SUPPORTED_MODELS, EXA_RERUN_MODELS)
