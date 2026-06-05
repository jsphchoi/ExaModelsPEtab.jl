# gpu_run.jl — quick GPU build + solve sanity check for a single model.
#
# Usage (from repo root):
#   julia --project=. -t 1 examples/debugging/gpu_run.jl [model] [K] [max_wall_time] [gpu_id]
# Defaults: Crauste_CellSystems2017, K=5, 300s wall time, GPU 0

using ExaModelsPEtab, CUDA, MadNLPGPU, CUDSS

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")

model         = length(ARGS) >= 1 ? ARGS[1] : "Crauste_CellSystems2017"
K             = length(ARGS) >= 2 ? parse(Int,     ARGS[2]) : 5
max_wall_time = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 300.0
gpu_id        = length(ARGS) >= 4 ? parse(Int,     ARGS[4]) : 0

CUDA.device!(gpu_id)
println("GPU $(gpu_id): $(CUDA.name(CUDA.device()))")

d  = joinpath(MODELDIR, model)
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))

println("Building $model (K=$K)...")
@time m = petab_examodel(yaml; backend=CUDABackend(), K=K)
@show m.meta.nvar, m.meta.ncon

println("\nSolving (max_wall_time=$(max_wall_time)s)...")
res = madnlp(m; linear_solver=MadNLPGPU.CUDSSSolver,
                tol=1e-6, max_wall_time=max_wall_time, max_iter=1_000_000)
println("\nstatus  = $(res.status)")
println("obj     = $(res.objective)")
println("iter    = $(res.iter)")
println("time    = $(round(res.counters.total_time; digits=1))s")
