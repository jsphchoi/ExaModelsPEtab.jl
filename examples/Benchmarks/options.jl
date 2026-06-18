# options.jl — canonical benchmark model lists (single source of truth).
# Included by run_examodels.jl, run_petab.jl, and results.jl so the three
# scripts never drift. Three NESTED canonical lists (1 ⊇ 2 ⊇ 3):
#
#   1. BENCHMARK_MODELS (35)     — every PEtab benchmark-collection model.
#   2. CONTINUOUS_MODELS (23)    — those WITHOUT the "Possible Discontinuities" model feature.
#                                  The 12 dropped are the collection's discontinuity models
#                                  (events / piecewise-in-time / state-triggered switches). The
#                                  collocation transcription represents one smooth ODE trajectory
#                                  per mesh interval and cannot encode a discontinuity, so these are
#                                  out of scope BY CONSTRUCTION — a feature limit, not a solver
#                                  failure. The 12 are exactly the models flagged "Possible
#                                  Discontinuities" on the official feature table:
#                                  https://benchmarking-initiative.github.io/Benchmark-Models-PEtab/
#   3. EXA_SUPPORTED_MODELS (20) — the reported ExaModelsPEtab target set: continuous AND solvable
#                                  by the PEtab.jl reference (so there is a baseline to compare to).
#                                  The 3 dropped from (2) are PEtab.jl-side compile/solve failures,
#                                  NOT ExaModelsPEtab failures:
#                                    Froehlich — PEtab compile intractable on this box (capped ~19.6h, never finished)
#                                    Lang      — compiled only on a longer rerun (2.65h), then PEtab solve errored (obj=Inf)
#                                    Raia      — PEtab codegen errors
#                                  KEEPS Fiedler (continuous, PEtab-solved): in scope by features
#                                  and benchmarked; its exa entry documents an acknowledged ExaModels
#                                  limitation (z0_func IC overflows the GPU kernel param budget)
#                                  rather than hiding it.
#
# WARMUP note: Bruno_JExpBot2016 is the shared JIT-warmup model for both benchmark scripts and
# is therefore excluded from their TIMED loops — a model warmed-up-on then benchmarked by the
# same process gets an invalid, pre-warmed compile time. Bruno is benchmarked on its own (warmed
# on Crauste) by run_bruno.jl. Crauste needs NO such exception: it is warmed by Bruno like
# every other in-loop model, so it is timed normally in the main loops. Only Bruno is special —
# EXA_SUPPORTED_MODELS minus {Bruno} = the 19 timed in-loop models.

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK + SOLVER CONFIGURATION — single source of truth.
# run_examodels.jl, run_petab.jl, and results.jl ALL read these so
# both backends run under IDENTICAL settings. CHANGE SETTINGS HERE, NOWHERE ELSE.
# ══════════════════════════════════════════════════════════════════════════════
# ── Shared (apply to BOTH backends) ──
const BENCH_SGM_N       = 5            # SGM warm-rerun count for the shifted geometric-mean (δ=10s) solve timing (exa & PEtab; 0 disables)
const BENCH_SGM_SHIFT   = 10           # shift δ (s) for the shifted geometric-mean solve timing
const BENCH_TOL         = 1e-6         # convergence tolerance — MadNLP `tol` == Optim `g_tol`
const BENCH_SOLVE_LIMIT = 7200.0       # solve wall cap [s] (2 hr) — MadNLP `max_wall_time` / Optim `time_limit` (applies to exa GPU/CPU and PEtab)
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
# Optim defaults except iterations/time_limit/g_tol: f_reltol=0 and x_abstol=0 are the generic
# Optim.Options defaults; successive_f_tol=2 and allow_f_increases=true are IPNewton's own defaults.
const BENCH_PETAB_F_RELTOL        = 0.0     # Optim default (was 1e-8)
const BENCH_PETAB_SUCCESSIVE_FTOL = 2       # IPNewton default (was 3)
const BENCH_PETAB_X_ABSTOL        = 0.0     # Optim default
# ══════════════════════════════════════════════════════════════════════════════

# Shifted geometric mean of solve times [s] with shift δ: exp(mean(log(tᵢ + δ))) − δ  (δ=0 ⇒ ordinary GM).
# Collected from the raw per-rerun times so the shift is applied at report time, not baked into storage.
shifted_geomean(times, δ = BENCH_SGM_SHIFT) = round(exp(sum(t -> log(t + δ), times) / length(times)) - δ; digits = 2)

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

# The 12 models carrying the "Possible Discontinuities" feature on the official table
# (https://benchmarking-initiative.github.io/Benchmark-Models-PEtab/): events / piecewise(t) /
# state-triggered switches. The collocation transcription cannot represent a discontinuity within a
# smooth per-interval ODE trajectory, so these are out of scope by construction (a feature limit,
# NOT a solver failure). This is the SOLE basis for the discontinuity exclusion.
const _POSSIBLE_DISCONTINUITIES = [
    "Alkan_SciSignal2018", "Beer_MolBioSystems2014", "Brannmark_JBC2010", "Chen_MSB2009",
    "Fujita_SciSignal2010", "Giordano_Nature2020", "Isensee_JCB2018", "Liu_IFACPapersOnLine2025",
    "Oliveira_NatCommun2021", "Raimundez_PCB2020", "Smith_BMCSystBiol2013", "Weber_BMC2015",
]

# The 3 continuous models the PEtab.jl reference solver itself could not compile/solve, so there is
# no baseline to benchmark against (these are PEtab.jl-side failures, NOT ExaModelsPEtab failures):
#   Froehlich — PEtab compile intractable on this box (capped ~19.6h, never finished)
#   Lang      — compiled only on a longer rerun (2.65h), then PEtab solve errored (obj=Inf)
#   Raia      — PEtab codegen errors
const _PETAB_UNSOLVED = ["Froehlich_CellSystems2018", "Lang_PLOSComputBiol2024", "Raia_CancerResearch2011"]

const CONTINUOUS_MODELS    = filter(m -> m ∉ _POSSIBLE_DISCONTINUITIES, BENCHMARK_MODELS)   # 23
const EXA_SUPPORTED_MODELS = filter(m -> m ∉ _PETAB_UNSOLVED, CONTINUOUS_MODELS)            # 20

@assert length(BENCHMARK_MODELS)           == 35
@assert length(_POSSIBLE_DISCONTINUITIES)  == 12
@assert length(CONTINUOUS_MODELS)          == 23
@assert length(EXA_SUPPORTED_MODELS)       == 20

# ── K=4 full-suite rerun order ───────────────────────────────────────────────────
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
    "Bertozzi_PNAS2020", "Zheng_PNAS2012",
    "Crauste_CellSystems2017", "Sneyd_PNAS2002", "Zhao_QuantBiol2020",
    "SalazarCavazos_MBoC2020", "Schwen_PONE2014",
    "Laske_PLOSComputBiol2019",
    # ── not converged in K=4, by compile time ascending ──
    "Borghans_BiophysChem1997", "Elowitz_Nature2000", "Lucarelli_CellSystems2018",
    "Bachmann_MSB2011",
    # acknowledged ExaModels limitation (z0_func IC overflows kernel param budget) — benchmarked anyway
    "Fiedler_BMCSystBiol2016",
]
# Bruno is the ONLY model benchmarked outside the in-loop scripts (it is their shared warmup),
# via run_bruno.jl. Crauste needs no exception — it is warmed by Bruno like every other
# in-loop model — so it lives in EXA_RERUN_INLOOP above.
const EXA_RERUN_MODELS = [EXA_RERUN_INLOOP; "Bruno_JExpBot2016"]  # 20 (in-loop 19 + Bruno)

@assert length(EXA_RERUN_INLOOP) == 19
@assert length(EXA_RERUN_MODELS) == 20
@assert all(m -> m ∈ EXA_SUPPORTED_MODELS, EXA_RERUN_MODELS)
@assert sort(EXA_RERUN_MODELS) == sort(EXA_SUPPORTED_MODELS)  # the rerun covers the full supported set
