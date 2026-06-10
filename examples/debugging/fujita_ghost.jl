# fujita_ghost.jl — is the collocation -322 basin a GHOST or a REAL optimum?
#
# The K=10 solve starts at the correct -53.08 and at iter 2 takes one full Newton step to a
# near-feasible collocation point with objective -322.5 (inf_pr ~1e-2), then restoration-fails.
# Question: does that collocation iterate's PARAMETER vector θ correspond to a real ODE optimum,
# or is -322 only reachable because the discretized states overfit (a collocation ghost)?
#
# Test: extract θ = solution[1:Np] at iter 2, evaluate PEtab's TRUE-ODE objective nllh(θ).
#   PEtab(θ) >> -322  (e.g. >= -53)  => GHOST  (states overfit; θ doesn't integrate to -322)
#   PEtab(θ) ~  -322                 => REAL alternative optimum PEtab is missing
using ExaModelsPEtab, PEtab, CUDA, MadNLPGPU, CUDSS, ExaModels
using PEtab: get_x

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
d    = joinpath(MODELDIR, "Fujita_SciSignal2010")
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))

PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)
Np      = PEprob.nparameters_estimate
x_opt   = Array(get_x(PEprob))
ref     = PEprob.nllh(x_opt)
println("=== reference ===")
println("PEtab nllh(nominal/optimum θ) = $ref   (expect ~ -53.08)")
println("Np = $Np")

model = petab_examodel(yaml; backend = CUDA.CUDABackend(), K = 10)
println("model nvar=$(model.meta.nvar) ncon=$(model.meta.ncon)")

# Stop at iter 2: the spurious near-feasible -322 point.
res = madnlp(model; tol=1e-6, max_iter=2, linear_solver=MadNLPGPU.CUDSSSolver, print_level=MadNLP.INFO)

θ_ghost = Array(res.solution)[1:Np]
println("=== spurious-basin iterate ===")
println("collocation objective at iter 2 = $(res.objective)   (expect ~ -322)")

petab_at_ghost = try
    PEprob.nllh(θ_ghost)
catch e
    "ERROR: $(e)"
end
println("--------------------------------------------------")
println("PEtab nllh(θ_ghost) = $petab_at_ghost")
println("--------------------------------------------------")
println("||θ_ghost - θ_opt||_inf = $(maximum(abs.(θ_ghost .- x_opt)))")
# show which params moved most
dθ = abs.(θ_ghost .- x_opt)
ord = sortperm(dθ; rev=true)
xn = String.(PEprob.xnames)
for i in ord[1:min(6,Np)]
    println("  Δθ[$(xn[i])] = $(round(dθ[i];digits=4))   opt=$(round(x_opt[i];digits=4))  ghost=$(round(θ_ghost[i];digits=4))")
end
