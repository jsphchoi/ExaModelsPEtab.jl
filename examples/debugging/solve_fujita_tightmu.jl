# solve_fujita_tightmu.jl — does a tight initial barrier keep Fujita in the -53 basin?
# Baseline (mu_init=0.1, default): flees to the -322 collocation GHOST, RESTORATION_FAILED.
# Hypothesis: the ghost is reached because at loose mu the first step trades feasibility for
# objective. A tight mu_init forces conservative early steps that stay near the (feasible,
# primal-correct) warm start -> should converge to the true -53.08.
using ExaModelsPEtab, PEtab, CUDA, MadNLPGPU, CUDSS, ExaModels

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
d    = joinpath(MODELDIR, "Fujita_SciSignal2010")
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))

MU = haskey(ENV, "MU_INIT") ? parse(Float64, ENV["MU_INIT"]) : 1e-7
println("=== Fujita K=10  mu_init=$MU ===")
model = petab_examodel(yaml; backend = CUDA.CUDABackend(), K = 10)
println("nvar=$(model.meta.nvar) ncon=$(model.meta.ncon)")

res = madnlp(model; tol=1e-6, acceptable_tol=1e-4, acceptable_iter=10,
             mu_init=MU, max_iter=2000, max_wall_time=1500.0,
             linear_solver=MadNLPGPU.CUDSSSolver)

println("--------------------------------------------------")
println("term_status = $(res.status)")
println("objective   = $(res.objective)")
println("iters       = $(res.iter)")
println("PEtab optimum reference = -53.0837724")
