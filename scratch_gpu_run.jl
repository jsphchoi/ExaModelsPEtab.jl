using ExaModels
using PEtab
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE
using CUDA, MadNLPGPU
using CUDSS   # activates the MadNLPGPU CUDSSSolver extension (GPU sparse linear solver)

const SRC = joinpath(@__DIR__, "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","userfuncs.jl")
    include(joinpath(SRC, f))
end

yaml = joinpath(@__DIR__, "examples", "Benchmark-Models",
                "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")

println("Building Crauste ExaModel on the GPU (CUDABackend)...")
@time m = petab_examodel(yaml; backend = CUDABackend(), K = 5)
@show m.meta.nvar
@show m.meta.ncon

println("\nRunning MadNLP (GPU, cuDSS) with 300s wall-time limit...")
result = madnlp(m; linear_solver = MadNLPGPU.CUDSSSolver,
                   max_wall_time = 300.0, max_iter = 1_000_000)
println("\n=== RESULT ===")
@show result.status
@show result.objective
@show result.iter
@show result.counters.total_time
