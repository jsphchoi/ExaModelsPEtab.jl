using ExaModelsPEtab                                              # petab_examodel (export) + module, for pkgdir
import PEtab                                                      # PEtab.* — reference nllh at nominal θ
import ExaModels                                                  # ExaModels.* — NLPModels interface (obj/cons!)
import MadNLP: madnlp, SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL
import CUDA
using Test

# ── Solver settings (mirror examples/Benchmarks/options.jl) ──────────────────────
const K           = 3
const TOL         = 1e-6
const ACCEPT_TOL  = 1e-4
const ACCEPT_ITER = 15
const MAX_ITER    = 100_000_000
const WALL        = 3600.0

# Three small models chosen to span the three construction paths.
const MODELS = [
    "Blasi_CellSystems2016",    # steady-state (no-mesh) path; log observable
    "Perelson_Science1996",     # time-course collocation path; log10 observable
    "Bruno_JExpBot2016",        # multiple conditions; cv (parameter + IC-override); linear observable
]

const MODELDIR = joinpath(pkgdir(ExaModelsPEtab), "examples", "Benchmark-Models")
yaml_of(m) = (d = joinpath(MODELDIR, m); joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d)))))
solved(s)  = s in (SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL)

# ∞-norm of the equality-constraint violation at a point (collocation/continuity/IC/σ residual).
function warmstart_viol(m, x)
    cx = similar(x, m.meta.ncon); ExaModels.cons!(m, x, cx)
    cx = Array(cx); lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
    maximum(max.(lcon .- cx, cx .- ucon, 0.0))
end

# The correctness of the transcription is checked WITHOUT hard-coding a converged objective
# (which is brittle against mesh-init / K changes). Instead we anchor on two mesh-robust facts —
# the warm-start objective reproduces PEtab's nllh, and the warm start is (nearly) feasible —
# plus a convergence check that asserts a KKT point no worse than the warm start.
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
        #     states equal the true ODE trajectory sampled at the measurement nodes, so the
        #     objective there must reproduce PEtab's nllh at nominal θ. Validates observables,
        #     σ, condition indexing and state-at-measurement mapping independent of mesh/K/solver.
        @test isapprox(obj0, ref_nllh; rtol = 1e-5)

        # (2) Warm-start feasibility — the equality constraints (collocation/continuity/IC) are
        #     nearly satisfied at x0. A gross violation flags a constraint-generation bug the
        #     objective check cannot see. Lenient threshold: the warm start carries ODE-presolve
        #     slack, but a real bug blows the residual up by orders of magnitude.
        @test warmstart_viol(model, x0) < 1e-2

        # (3) Solve converges (CPU) to a KKT point no worse than the warm start. We assert the
        #     status + a finiteness/improvement bound rather than a brittle hard-coded optimum.
        stats = madnlp(model; tol = TOL, acceptable_tol = ACCEPT_TOL, acceptable_iter = ACCEPT_ITER,
                       max_iter = MAX_ITER, max_wall_time = WALL)
        @test solved(stats.status)
        @test isfinite(stats.objective)
        @test stats.objective <= obj0 + 1e-6
    end

    # GPU build + solve when CUDA is available — the primary production path.
    if CUDA.functional()
        import MadNLPGPU, CUDSS   # CUDSS (with CUDA) triggers MadNLPGPU's CUDA extension → CUDSSSolver
        @testset "GPU: $m" for m in MODELS
            model = petab_examodel(yaml_of(m); backend = CUDA.CUDABackend(), K = K)
            stats = madnlp(model; tol = TOL, acceptable_tol = ACCEPT_TOL, acceptable_iter = ACCEPT_ITER,
                           max_iter = MAX_ITER, max_wall_time = WALL, linear_solver = MadNLPGPU.CUDSSSolver)
            @test solved(stats.status)
            @test isfinite(stats.objective)
        end
    end
end
