# boehm_bench_solve.jl — solve Boehm with the EXACT run_bruno benchmark MadNLP config
# (CPU backend), dump the full iteration log, and analyze which variable blocks move
# from the warm start x0 to the optimum x*. Answers: what must change before optimality?
#
# Usage (from repo root):
#   julia --project=examples -t 1 examples/debugging/boehm_bench_solve.jl [model] [K]

using ExaModelsPEtab, PEtab, ExaModels, MadNLP
using LinearAlgebra
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

P(args...) = (println(stderr, args...); flush(stderr))

model = length(ARGS) >= 1 ? ARGS[1] : "Boehm_JProteomeRes2014"
K     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
_yaml(m) = (d = joinpath(MODELDIR, m); joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d)))))

# EXACT run_bruno settings (BENCH_*). CPU branch (no cuDSS).
const TOL = 1e-6; const ACCEPT_TOL = 1e-4; const ACCEPT_ITER = 15
const KKT_OPTS = (kkt_system = MadNLP.SparseCondensedKKTSystem,
                  equality_treatment = MadNLP.RelaxEquality,
                  fixed_variable_treatment = MadNLP.RelaxBound)

P("============================== $model (K=$K) — benchmark CPU config ==============================")
PEmodel = PEtab.PEtabModel(_yaml(model))
PEprob  = PEtab.PEtabODEProblem(PEmodel)

c = ExaModels.ExaCore(; backend = nothing, concrete = Val(true))
c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
c = _create_collocation(c, PEmodel, PEprob, PEinfo; adaptive_mesh = false)
c = _create_continuity(c, PEmodel, PEprob, PEinfo)
c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)
m = ExaModels.ExaModel(c)
ExaModels.set_start!(m, c.y, y0)
ExaModels.set_start!(m, c.sigma, sigma0)

Np, Nm, Nz, N, Nc = PEinfo.Np, PEinfo.Nm, PEinfo.Nz, PEinfo.N, PEinfo.Nc
x0 = Array(m.meta.x0)
P("nvar=$(m.meta.nvar)  ncon=$(m.meta.ncon)  Np=$Np  Nz=$Nz  N=$N  K=$K  Nc=$Nc  Nm=$Nm")

# variable-block ranges (by ExaModels offsets)
zlen = Nz*N*(K+1)*Nc
rng = Dict(:p=>(c.p.offset+1:c.p.offset+Np), :z=>(c.z.offset+1:c.z.offset+zlen),
           :y=>(c.y.offset+1:c.y.offset+Nm), :sigma=>(c.sigma.offset+1:c.sigma.offset+Nm))

P("\n--- SOLVE (tol=$TOL, acceptable_tol=$ACCEPT_TOL, RelaxEquality, SparseCondensedKKT) ---")
res = madnlp(m; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER,
                max_iter=5000, max_wall_time=600.0, print_level=MadNLP.INFO, KKT_OPTS...)

xopt = Array(res.solution)
P("\n--- RESULT ---")
P("status = $(res.status)   iter = $(res.iter)   obj = $(res.objective)")
P("obj(x0) = $(ExaModels.obj(m, m.meta.x0))   Δobj = $(ExaModels.obj(m, m.meta.x0) - res.objective)")

P("\n--- WHAT MOVED (max|Δ| and L2 per variable block) ---")
for k in (:p, :z, :y, :sigma)
    r = rng[k]; d = abs.(x0[r] .- xopt[r])
    P(rpad(string(k), 6), " n=", rpad(length(r), 6),
      " max|Δ|=", rpad(round(maximum(d), sigdigits=4), 12),
      " L2=", rpad(round(norm(d), sigdigits=4), 12),
      " |x0|range=[", round(minimum(x0[r]), sigdigits=4), ", ", round(maximum(x0[r]), sigdigits=4), "]")
end

P("\n--- PARAMETERS: x0 vs x*, and distance to nearest bound ---")
lb = Array(m.meta.lvar)[1:Np]; ub = Array(m.meta.uvar)[1:Np]
pnames = try string.(collect(keys(PEtab.get_x(PEprob))))[1:Np] catch; ["p$i" for i in 1:Np] end
for i in 1:Np
    gap = min(x0[i]-lb[i], ub[i]-x0[i])
    P(rpad(pnames[i], 22), " x0=", rpad(round(x0[i], sigdigits=6), 12),
      " x*=", rpad(round(xopt[i], sigdigits=6), 12),
      " Δ=", rpad(round(xopt[i]-x0[i], sigdigits=4), 12),
      " bound=[", lb[i], ",", ub[i], "] gap0=", round(gap, sigdigits=3))
end
