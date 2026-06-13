using ExaModelsPEtab                                              # petab_examodel (export) + module, for pkgdir
import PEtab                                                      # PEtab.* — reference nllh at nominal θ
import ExaModels                                                  # ExaModels.* — NLPModels interface (obj/cons!)
using MadNLP                                                      # MadNLP.madnlp / MadNLP.SOLVE_SUCCEEDED (qualified) below
using CUDA
using Test

# ── Settings (mirror examples/Benchmarks/options.jl) ─────────────────────────────
const K           = 4
const TOL         = 1e-6
const ACCEPT_TOL  = 1e-4
const ACCEPT_ITER = 15
const MAX_ITER    = 100_000_000
const WALL        = 3600.0

# MadNLP solver config — the LiftedKKT (condensed-space) regime, matching the GPU/CUDSS path
# (which selects it by default) so the CPU solve uses the same configuration; only the linear solver
# differs (CPU auto-selects, GPU uses CUDSS). `RelaxEquality` is also what makes the CPU solve robust:
# the default full-space `EnforceEquality` drives an iterate across a state's bound into
# `log(negative)` → DomainError.
# NOTE: the GPU also defaults to `bound_relax_factor=1e-4`, but we do NOT set it here — on CPU the
# relaxed bound lets a state go slightly negative and `log(neg)` THROWS (GPU returns NaN and
# backtracks). CPU keeps the tighter default (1e-8). A domain-safe `log` in the objective would let
# CPU match the GPU's bound_relax exactly.
const SOLVER = (;
    tol = TOL, acceptable_tol = ACCEPT_TOL, acceptable_iter = ACCEPT_ITER,
    max_iter = MAX_ITER, max_wall_time = WALL,
    kkt_system               = MadNLP.SparseCondensedKKTSystem,   # LiftedKKT (condensed)
    equality_treatment       = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
)

# Three small models chosen to span the three construction paths.
const MODELS = [
    "Blasi_CellSystems2016",    # steady-state (no-mesh) path; log observable
    "Perelson_Science1996",     # time-course collocation path; log10 observable
    "Bruno_JExpBot2016",        # multiple conditions; cv (parameter + IC-override); linear observable
]

const MODELDIR = joinpath(pkgdir(ExaModelsPEtab), "examples", "Benchmark-Models")
yaml_of(m) = (d = joinpath(MODELDIR, m); joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d)))))
solved(s)  = s in (MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)

# ∞-norm of the equality-constraint violation at a point (collocation/continuity/IC/σ residual).
function warmstart_viol(m, x)
    cx = similar(x, m.meta.ncon); ExaModels.cons!(m, x, cx)
    cx = Array(cx); lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
    maximum(max.(lcon .- cx, cx .- ucon, 0.0))
end

# Correctness is checked WITHOUT hard-coding a converged objective (brittle against mesh-init / K
# changes). We anchor on two mesh-robust facts — the warm-start objective reproduces PEtab's nllh,
# and the warm start is (nearly) feasible — plus a convergence check (KKT point no worse than x0).
@testset "ExaModelsPEtab.jl" begin
    @testset "CPU: $m" for m in MODELS
        yaml = yaml_of(m)

        # Reference: PEtab's negative log-likelihood at the nominal parameters.
        prob     = PEtab.PEtabODEProblem(PEtab.PEtabModel(yaml))
        ref_nllh = prob.nllh(PEtab.get_x(prob))

        model = petab_examodel(yaml; backend = nothing, K = K)
        x0    = model.meta.x0
        obj0  = ExaModels.obj(model, x0)

        # (1) Transcription correctness — MESH/K-INDEPENDENT. At the warm start the collocation
        #     states equal the true ODE trajectory at the measurement nodes, so the objective there
        #     reproduces PEtab's nllh at nominal θ — validating observables, σ, condition indexing
        #     and state-at-measurement mapping independent of mesh/K/solver.
        @test isapprox(obj0, ref_nllh; rtol = 1e-5)

        # (2) Warm-start feasibility — the equality constraints (collocation/continuity/IC) are
        #     nearly satisfied at x0. A gross violation flags a constraint-generation bug the
        #     objective check cannot see.
        @test warmstart_viol(model, x0) < 1e-2

        # (3) Solve converges to a KKT point no worse than the warm start (LiftedKKT, CPU).
        stats = MadNLP.madnlp(model; SOLVER...)
        @test solved(stats.status)
        @test isfinite(stats.objective)
        @test stats.objective <= obj0 + 1e-6
    end

    # GPU build + solve when CUDA is available — same LiftedKKT settings, CUDSS linear solver.
    if CUDA.functional()
        import MadNLPGPU, CUDSS   # CUDSS (with CUDA) triggers MadNLPGPU's CUDA extension → CUDSSSolver
        @testset "GPU: $m" for m in MODELS
            model = petab_examodel(yaml_of(m); backend = CUDA.CUDABackend(), K = K)
            stats = MadNLP.madnlp(model; SOLVER..., linear_solver = MadNLPGPU.CUDSSSolver)
            @test solved(stats.status)
            @test isfinite(stats.objective)
        end
    end
end
