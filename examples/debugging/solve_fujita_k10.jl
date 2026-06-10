# solve_fujita_k10.jl — solve the CORRECTED (Part-1 exact-gate) Fujita at K=10 on GPU.
# The prior benchmark run was the BUGGY gate transcription (obj -323.5, false pass);
# this is the first real solve of the corrected model (warm obj -53.08).
using ExaModelsPEtab, PEtab, CUDA, MadNLPGPU, CUDSS, ExaModels

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
d    = joinpath(MODELDIR, "Fujita_SciSignal2010")
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))

println("=== build Fujita K=10 (corrected gate transcription) ===")
@time model = petab_examodel(yaml; backend = CUDA.CUDABackend(), K = 10)
println("nvar=$(model.meta.nvar)  ncon=$(model.meta.ncon)")

println("=== solve (madnlp / CUDSS) ===")
res = madnlp(model; tol=1e-6, acceptable_tol=1e-4, acceptable_iter=10,
             max_iter=3000, max_wall_time=3600.0,
             linear_solver=MadNLPGPU.CUDSSSolver)

println("--------------------------------------------------")
println("term_status = $(res.status)")
println("objective   = $(res.objective)")
println("iters       = $(res.iter)")
println("PEtab optimum reference = -53.0837724")
