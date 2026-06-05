# verify_models.jl — CPU build + warm-start check across multiple models;
# optionally GPU solve + PEtab IPNewton objective comparison.
#
# Usage (from repo root):
#   julia --project=. -t 1 examples/scratch_tests/verify_models.jl [K] [phase] [model ...]
#
#   phase = build   — CPU build + warm-start feasibility only (fast, no GPU needed)
#   phase = solve   — + GPU MadNLP solve + PEtab IPNewton comparison
#
# Examples:
#   julia --project=. examples/scratch_tests/verify_models.jl 5 build
#   julia --project=. -t 1 examples/scratch_tests/verify_models.jl 5 solve Bruno_JExpBot2016 Crauste_CellSystems2017

using ExaModelsPEtab, PEtab, ExaModels
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
const SRCDIR   = joinpath(@__DIR__, "..", "..", "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","steadystate.jl","userfuncs.jl")
    include(joinpath(SRCDIR, f))
end

K      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
phase  = length(ARGS) >= 2 ? ARGS[2] : "build"
models = length(ARGS) >= 3 ? ARGS[3:end] :
         ["Bruno_JExpBot2016", "Crauste_CellSystems2017", "Boehm_JProteomeRes2014"]

const MAX_WALL_TIME = 500.0

_yaml(m) = begin
    d = joinpath(MODELDIR, m)
    joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
end

if phase == "solve"
    using CUDA, MadNLPGPU, CUDSS
    CUDA.device!(0)
    println("GPU 0: $(CUDA.name(CUDA.device()))")
end

for model in models
    println("\n============================== $model (K=$K) ==============================")
    yaml    = _yaml(model)
    PEmodel = PEtab.PEtabModel(yaml)
    PEprob  = PEtab.PEtabODEProblem(PEmodel)

    # CPU build + warm-start feasibility
    m_cpu = _build_petab_examodel(PEmodel, PEprob, nothing, K)
    x0    = m_cpu.meta.x0
    cx    = similar(x0, m_cpu.meta.ncon); ExaModels.cons!(m_cpu, x0, cx); cx = Array(cx)
    lcon  = Array(m_cpu.meta.lcon); ucon = Array(m_cpu.meta.ucon)
    viol  = maximum(max.(lcon .- cx, cx .- ucon, 0.0))
    obj0  = ExaModels.obj(m_cpu, x0)
    println("CPU: nvar=$(m_cpu.meta.nvar)  ncon=$(m_cpu.meta.ncon)  warm_obj=$obj0  max|viol|=$viol")
    viol > 1e-1 && @warn "$model warm start not feasible (max|viol|=$viol)"

    phase == "build" && continue

    # GPU solve
    m_gpu = _build_petab_examodel(PEmodel, PEprob, CUDA.CUDABackend(), K)
    res = madnlp(m_gpu; linear_solver=MadNLPGPU.CUDSSSolver,
                         tol=1e-6, max_wall_time=MAX_WALL_TIME, max_iter=1_000_000)
    println("GPU: status=$(res.status)  obj=$(res.objective)  iter=$(res.iter)  time=$(round(res.counters.total_time; digits=1))s")

    # PEtab IPNewton comparison
    try
        using Optim
        pres   = PEtab.calibrate(PEprob, PEtab.get_x(PEprob), Optim.IPNewton())
        θstar  = Array(res.solution)[1:PEprob.nparameters_estimate]
        nllh_at_mine = PEprob.nllh(θstar)
        println("PEtab IPNewton: fmin=$(pres.fmin)")
        println("PEtab nllh @ my θ*: $nllh_at_mine   Δ=$(nllh_at_mine - pres.fmin)")
    catch e
        println("PEtab comparison skipped: $e")
    end
end
