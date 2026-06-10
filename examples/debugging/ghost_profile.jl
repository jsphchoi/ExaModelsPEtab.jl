# ghost_profile.jl — localize WHERE the -323 collocation ghost departs from a real trajectory.
# Solve Fujita K=10 to the ghost, extract its collocation states z_ghost and params θ_ghost, then:
#   (1) z_ghost  vs  ODE(θ_ghost) on the mesh  -> the COLLOCATION CHEAT (states aren't a real
#       trajectory even at their OWN params; this is the +885 nllh made visible in state space)
#   (2) z_ghost  vs  ODE(θ_opt)   on the mesh  -> total departure from the optimal profile
# Reports, per state and per time-interval, where the deviation concentrates.
using ExaModelsPEtab, PEtab, CUDA, MadNLPGPU, CUDSS, ExaModels
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE
import ModelingToolkitBase as MTK
using Symbolics
using PEtab: get_x
const SRCDIR = joinpath(@__DIR__, "..", "..", "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","userfuncs.jl")
    include(joinpath(SRCDIR, f))
end

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
d    = joinpath(MODELDIR, "Fujita_SciSignal2010")
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
K = 10

PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)
Np = PEprob.nparameters_estimate
θ_opt = Array(get_x(PEprob))

# Build (staged so we keep PEinfo) on GPU and solve to the ghost.
c = ExaModels.ExaCore(; backend=CUDA.CUDABackend(), concrete=Val(true))
c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
c = _create_collocation(c, PEmodel, PEprob, PEinfo)
c = _create_continuity(c, PEmodel, PEprob, PEinfo)
c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)
model = ExaModels.ExaModel(c)
ExaModels.set_start!(model, c.y, y0); ExaModels.set_start!(model, c.sigma, sigma0)
res = madnlp(model; tol=1e-6, acceptable_tol=1e-4, acceptable_iter=10,
             max_iter=600, max_wall_time=900.0, linear_solver=MadNLPGPU.CUDSSSolver)
println("ghost solve: status=$(res.status) obj=$(res.objective) iters=$(res.iter)")

(; Nz, N, Nc, taus, t_vec_mesh, t_nodes, h) = PEinfo
x = Array(res.solution)
θ_ghost = x[1:Np]
zblk = x[Np+1 : Np + Nz*N*(K+1)*Nc]
z_ghost = reshape(zblk, Nz, N, K+1, Nc)               # [v,i,j+1,cidx]

# True ODE trajectory on the SAME mesh at a given θ (mirrors _get_z_init's interpolation step).
function ode_on_mesh(θ)
    t_events = _get_event_times(PEmodel, PEprob)
    t_meas   = sort(unique(filter(t->!iszero(t), PEmodel.petab_tables[:measurements][!,:time])))
    sol = _solve_conds(θ, PEmodel, PEprob, sort(unique(vcat(t_meas, t_events))))
    cids = Symbol.(_get_cids(PEmodel))
    sam = [sol[cid](t) for t in t_vec_mesh, cid in cids]
    permutedims(reshape(stack(sam), Nz, K+1, N, Nc), (1,3,2,4))
end
z_true_ghost = ode_on_mesh(θ_ghost)   # real trajectory at the ghost's OWN params
z_true_opt   = ode_on_mesh(θ_opt)     # the optimal profile

rng = [maximum(@view z_true_opt[v,:,:,:]) - minimum(@view z_true_opt[v,:,:,:]) for v in 1:Nz]
rng = [r>0 ? r : 1.0 for r in rng]
state_names = string.(_get_z_syms(PEprob))

function report(zb, za, title)
    dev = zb .- za
    nd  = [maximum(abs.(@view dev[v,:,:,:]))/rng[v] for v in 1:Nz]   # per-state normalized max dev
    println("\n=== $title ===")
    println("max normalized state deviation per state (top 6):")
    for v in sortperm(nd; rev=true)[1:min(6,Nz)]
        # locate worst (i,cidx)
        sub = abs.(@view dev[v,:,:,:]); idx = argmax(sub); i = idx[1]
        println("  $(rpad(state_names[v],22)) dev/range=$(round(nd[v];digits=4))  worst interval i=$i  t≈$(round(t_nodes[i];digits=3))")
    end
    # time localization: per-interval total normalized deviation
    perint = [sum(abs.(@view dev[v,i,:,:])/rng[v] for v in 1:Nz) |> sum for i in 1:N]
    j = argmax(perint)
    println("time-localized: worst interval i=$j  t∈[$(round(t_nodes[j];digits=3)),$(round(t_nodes[j+1];digits=3))]  (event/transient region?)")
end

report(z_ghost, z_true_ghost, "(1) z_ghost vs ODE(θ_ghost) — the collocation CHEAT")
report(z_ghost, z_true_opt,   "(2) z_ghost vs ODE(θ_opt)   — departure from optimal profile")
println("\nΔθ ||inf|| = $(maximum(abs.(θ_ghost .- θ_opt)))   obj_ghost=$(res.objective)  PEtab(θ_ghost)=$(PEprob.nllh(θ_ghost))")
