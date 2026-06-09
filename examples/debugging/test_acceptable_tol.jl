# test_acceptable_tol.jl — does MadNLP's acceptable_tol certify Schwen's boundary optimum?
# Build Schwen on GPU, solve with tol=1e-6 + acceptable_tol=1e-4/acceptable_iter=10.
# Expect: SOLVED_TO_ACCEPTABLE_LEVEL at obj ~943.x instead of the inf_du-stall timeout.
using ExaModelsPEtab, PEtab, CUDA, MadNLPGPU, CUDSS, ExaModels
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE
const SRCDIR = joinpath(@__DIR__, "..", "..", "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","steadystate.jl","userfuncs.jl")
    include(joinpath(SRCDIR, f))
end
const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
gpu = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
K   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
CUDA.device!(gpu)
println("GPU $gpu: $(CUDA.name(CUDA.device()))  K=$K")

yaml    = joinpath(MODELDIR, "Schwen_PONE2014", "Schwen_PONE2014.yaml")
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)
m = _build_petab_examodel(PEmodel, PEprob, CUDA.CUDABackend(), K)
println("built: nvar=$(m.meta.nvar) ncon=$(m.meta.ncon)")

println("\n--- baseline tol=1e-6 (reference: stalls) is what the campaign saw; now acceptable_tol ---")
res = madnlp(m; linear_solver=MadNLPGPU.CUDSSSolver,
             tol=1e-6, acceptable_tol=1e-4, acceptable_iter=10,
             max_wall_time=900.0, max_iter=100000)
println("\n=== RESULT ===")
println("  status   = $(res.status)")
println("  objective= $(res.objective)")
println("  iter     = $(res.iter)")
println("  (PEtab nllh @ optimum = 956.52; exa obj-at-nominal = 943.999)")
