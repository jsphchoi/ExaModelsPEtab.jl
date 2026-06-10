# solve_check.jl — solve a model with the benchmark MadNLP settings, then GHOST-CHECK the
# result: score exa's converged θ under PEtab's TRUE ODE. If exa converged to a real optimum,
# exa objective ~ PEtab nllh(θ) (modulo the prior offset). A large gap => collocation ghost.
#   env: MODEL (dir name), K (default 4), MU_INIT (optional)
using ExaModelsPEtab, PEtab, CUDA, MadNLPGPU, CUDSS, ExaModels
using PEtab: get_x

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
model_name = get(ENV, "MODEL", "Oliveira_NatCommun2021")
K          = parse(Int, get(ENV, "K", "4"))
d    = joinpath(MODELDIR, model_name)
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))

println("=== $model_name  K=$K ===")
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)
Np      = PEprob.nparameters_estimate
x0      = Array(get_x(PEprob))
ref     = PEprob.nllh(x0)
println("PEtab nllh(nominal θ) = $ref   (Np=$Np)")

model = petab_examodel(yaml; backend = CUDA.CUDABackend(), K = K)
println("nvar=$(model.meta.nvar) ncon=$(model.meta.ncon)")

kw = (; tol=parse(Float64,get(ENV,"TOL","1e-6")),
        acceptable_tol=parse(Float64,get(ENV,"ACCEPT_TOL","1e-4")), acceptable_iter=10,
        max_iter=parse(Int,get(ENV,"MAX_ITER","2000")),
        max_wall_time=1500.0, linear_solver=MadNLPGPU.CUDSSSolver)
res = haskey(ENV,"MU_INIT") ? madnlp(model; kw..., mu_init=parse(Float64,ENV["MU_INIT"])) :
                              madnlp(model; kw...)

θ = Array(res.solution)[1:Np]
petab_at_θ = try PEprob.nllh(θ) catch e "ERROR: $e" end
println("--------------------------------------------------")
println("term_status        = $(res.status)")
println("exa objective      = $(res.objective)   iters=$(res.iter)")
println("PEtab nllh(θ_exa)  = $petab_at_θ")
println("||θ_exa - θ_nom||_inf = $(maximum(abs.(θ .- x0)))")
println("--------------------------------------------------")
println("GHOST CHECK: real convergence => exa obj ~ PEtab nllh(θ_exa); large gap => ghost")
