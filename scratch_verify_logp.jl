###############################################################################
# Verify the productionized log-parameter transform (decision var p := θ on the
# PEtab estimation scale, physical param = _p_phys(θ)) across models.
#
# phase = build : CPU build + warm-start feasibility only (fast)
#         solve : + GPU madnlp solve (tol 1e-6) and PEtab IPNewton comparison
#
# Run:  julia --project scratch_verify_logp.jl [K] [phase] [model...]
###############################################################################

using ExaModels, PEtab
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE
using CUDA, MadNLPGPU, CUDSS

# Pin to the GPU with the most free memory (this box may be shared).
if CUDA.functional()
    freemem(d) = (CUDA.device!(d); CUDA.available_memory())
    best = argmax(d -> freemem(d), 0:length(CUDA.devices())-1)
    CUDA.device!(best)
    println("Using CUDA device $best ($(CUDA.name(CUDA.device()))), ",
            round(CUDA.available_memory()/2^30, digits=1), " GiB free"); flush(stdout)
end

const SRC = joinpath(@__DIR__, "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","userfuncs.jl")
    include(joinpath(SRC, f))
end

K      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
phase  = length(ARGS) >= 2 ? ARGS[2] : "build"
models = length(ARGS) >= 3 ? ARGS[3:end] :
         ["Crauste_CellSystems2017", "Bruno_JExpBot2016", "Boehm_JProteomeRes2014"]

yaml_of(m) = joinpath(@__DIR__, "examples", "Benchmark-Models", m, m * ".yaml")

function warm_stats(m)
    x0 = m.meta.x0; cx = similar(x0, m.meta.ncon); ExaModels.cons!(m, x0, cx)
    cx = Array(cx); lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
    return ExaModels.obj(m, x0), maximum(max.(lcon .- cx, cx .- ucon, 0.0))
end

for model in models
    yaml = yaml_of(model)
    println("\n############################ ", model, " (K=$K) ############################"); flush(stdout)
    PEmodel = PEtab.PEtabModel(yaml)
    PEprob  = PEtab.PEtabODEProblem(PEmodel)
    println("param scales: ", collect(zip(PEprob.xnames, _get_pscale(PEprob))))

    # ---- CPU build + warm-start feasibility ----
    m_cpu = _build_petab_examodel(PEmodel, PEprob, nothing, K)
    o0, v0 = warm_stats(m_cpu)
    println("CPU build: nvar=", m_cpu.meta.nvar, " ncon=", m_cpu.meta.ncon,
            "  DoF=", m_cpu.meta.nvar - m_cpu.meta.ncon,
            "  warm-obj=", o0, "  max|viol|=", v0); flush(stdout)
    if v0 > 1e-1
        @warn "$model warm start not feasible (max|viol|=$v0)"
    end

    phase == "build" && continue

    # ---- GPU solve (tol 1e-6) ----
    m = _build_petab_examodel(PEmodel, PEprob, CUDABackend(), K)
    res = madnlp(m; linear_solver = MadNLPGPU.CUDSSSolver,
                    tol = 1e-6, max_wall_time = 500.0, max_iter = 1_000_000)
    println(">>> ", model, " GPU solve: status=", res.status,
            "  obj=", res.objective, "  iter=", res.iter,
            "  time=", round(res.counters.total_time, digits=1), "s"); flush(stdout)

    # ---- PEtab.jl IPNewton comparison ----
    try
        @eval using Optim
        pres = PEtab.calibrate(PEprob, PEtab.get_x(PEprob), Optim.IPNewton())
        θstar = Array(res.solution)[1:PEprob.nparameters_estimate]
        nllh_mine  = PEprob.nllh(θstar)
        println("    PEtab IPNewton fmin=", pres.fmin,
                "   PEtab nllh at my θ*=", nllh_mine,
                "   Δ=", nllh_mine - pres.fmin,
                "   (my collocation obj=", res.objective, ")"); flush(stdout)
    catch e
        println("    [PEtab IPNewton comparison skipped: ", e, "]"); flush(stdout)
    end
end
