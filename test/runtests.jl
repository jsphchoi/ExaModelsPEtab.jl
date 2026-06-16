using ExaModelsPEtab
import PEtab
import ExaModels
using MadNLP
using CUDA
using Test

# General mesh and solver settings (same as examples/Benchmarks/options.jl)
const K           = 4
const TOL         = 1e-6
const ACCEPT_TOL  = 1e-4
const ACCEPT_ITER = 15
const MAX_ITER    = 100_000_000
const WALL        = 3600.0

# Apply same solver settings for MadNLP CPU and GPU
const SOLVER = (;
    tol = TOL,
    acceptable_tol = ACCEPT_TOL,
    acceptable_iter = ACCEPT_ITER,
    max_iter = MAX_ITER,
    max_wall_time = WALL,
    kkt_system               = MadNLP.SparseCondensedKKTSystem,
    equality_treatment       = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.RelaxBound,
)

# Choice of three Benchmark-Models which span most of the petab_examodel construction paths
const MODELS = [
    "Blasi_CellSystems2016",    # steady-state (no-mesh) path, :log observable, affine noise
    "Schwen_PONE2014",          # collocation path, :log10 observable, sqrt noise, 19 conditions
    "Sneyd_PNAS2002",           # collocation path, :lin observable, assignment rules, 9 conditions
]

# PEtab known objective function values at the solution p*
const PETAB_OBJ = Dict(
    "Blasi_CellSystems2016" => -1090.5608159562705,
    "Schwen_PONE2014"       =>   956.5184339082019,
    "Sneyd_PNAS2002"        =>  -319.791791777273,
)

# .yaml pathfinding and helper functions for checking test pass
const MODELDIR = joinpath(pkgdir(ExaModelsPEtab), "examples", "Benchmark-Models")
yaml_of(m) = (
    d = joinpath(MODELDIR, m);
    joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
)
solved(s)  = s in (MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)

# Evaluate PEtab objective function at ExaModels optimal solution p*
petab_obj_at(PEprob, res) = PEprob.nllh(Array(res.solution)[1:PEprob.nparameters_estimate])

# inf-norm of equality-constraint (collocation, continuity, initial condition, noise) violation
function warmstart_viol(m, x)
    cx = similar(x, m.meta.ncon); ExaModels.cons!(m, x, cx)
    cx = Array(cx); lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
    maximum(max.(lcon .- cx, cx .- ucon, 0.0))
end

# Confirm that the working ExaModels builds converge to same solution as PEtab
@testset "ExaModelsPEtab.jl" begin
    @testset "CPU: $m" for m in MODELS
        yaml = yaml_of(m)

        # Evaluate PEtab's objective function at the nominal p
        PEprob       = PEtab.PEtabODEProblem(PEtab.PEtabModel(yaml))
        nllh_nominal = PEprob.nllh(PEtab.get_x(PEprob))

        # Evaluate the ExaModels objective function at the nominal p
        model   = petab_examodel(yaml; backend = nothing, K = K)
        x0      = model.meta.x0
        exa_obj = ExaModels.obj(model, x0)

        # Confirm objective function evaluates to approximately the same value at the nominal p
        @test isapprox(exa_obj, nllh_nominal; rtol = 1e-4)

        # Confirm ExaModels equality constraints are not overly violated at the nominal p
        @test warmstart_viol(model, x0) < 5e-2

        # Confirm MadNLP solves the ExaModel to PEtab's converged objective
        res = MadNLP.madnlp(model; SOLVER...)
        @test solved(res.status) # confirm ExaModels + MadNLP solved
        @test isfinite(res.objective) # confirm result is sensical
        @test isapprox(res.objective, PETAB_OBJ[m]; rtol = 1e-4) # confirm ExaModels obj is similar to PEtab obj
        @test isapprox(petab_obj_at(PEprob, res), PETAB_OBJ[m]; rtol = 1e-4) # confirm p* eval'd at PEtab obj matches PEtab obj
    end
    
    # if CUDA is available, test the GPU build
    if CUDA.functional()
        import MadNLPGPU, CUDSS
        @testset "GPU: $m" for m in MODELS
            PEprob = PEtab.PEtabODEProblem(PEtab.PEtabModel(yaml_of(m)))
            model  = petab_examodel(yaml_of(m); backend = CUDA.CUDABackend(), K = K)
            res    = MadNLP.madnlp(model; SOLVER..., linear_solver = MadNLPGPU.CUDSSSolver)
            @test solved(res.status) # confirm ExaModels + MadNLP solved
            @test isfinite(res.objective) # confirm result is sensical
            @test isapprox(res.objective, PETAB_OBJ[m]; rtol = 1e-2) # confirm ExaModels obj is similar to PEtab obj
            @test isapprox(petab_obj_at(PEprob, res), PETAB_OBJ[m]; rtol = 1e-4) # confirm p* eval'd at PEtab obj matches PEtab obj
        end
    end
end
