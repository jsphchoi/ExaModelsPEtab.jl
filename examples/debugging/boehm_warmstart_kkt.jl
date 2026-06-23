# boehm_warmstart_kkt.jl — is the ExaModelsPEtab warm start a KKT point?
# Builds the collocation NLP inline (so we can read aux-var offsets), measures primal
# feasibility (constraint residual) and stationarity (objective gradient) at x0, inspects
# the y/sigma warm-start, then runs MadNLP. All diagnostics go to STDERR (unbuffered, so
# they survive a DomainError crash) and every eval is wrapped in try/catch.
#
# Usage (from repo root):
#   julia --project=examples -t 1 examples/debugging/boehm_warmstart_kkt.jl [model] [K]

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

_yaml(m) = begin
    d = joinpath(MODELDIR, m)
    joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
end

P("============================== $model (K=$K) ==============================")
PEmodel = PEtab.PEtabModel(_yaml(model))
PEprob  = PEtab.PEtabODEProblem(PEmodel)

# ---- Inline build so we keep `c` (need c.y / c.sigma / c.p offsets) ----
c = ExaModels.ExaCore(; backend = nothing, concrete = Val(true))
c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
c = _create_collocation(c, PEmodel, PEprob, PEinfo; adaptive_mesh = false)
c = _create_continuity(c, PEmodel, PEprob, PEinfo)
c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)
m = ExaModels.ExaModel(c)
ExaModels.set_start!(m, c.y, y0)
ExaModels.set_start!(m, c.sigma, sigma0)

nvar, ncon = m.meta.nvar, m.meta.ncon
Np, Nm = PEinfo.Np, PEinfo.Nm
P("nvar=$nvar  ncon=$ncon  Np=$Np  Nm=$Nm")

x0 = Array(m.meta.x0)

# ---- Inspect the y / sigma warm-start blocks via their variable offsets ----
yoff = c.y.offset; soff = c.sigma.offset
P("\n--- WARM-START AUX VARS ---")
P("y0 (computed)     : min=", minimum(y0), "  max=", maximum(y0))
P("sigma0 (computed) : min=", minimum(sigma0), "  max=", maximum(sigma0))
P("x0[y block]       : min=", minimum(x0[yoff+1:yoff+Nm]), "  max=", maximum(x0[yoff+1:yoff+Nm]))
P("x0[sigma block]   : min=", minimum(x0[soff+1:soff+Nm]), "  max=", maximum(x0[soff+1:soff+Nm]))
P("any sigma start <= 0 ? ", any(<=(0.0), x0[soff+1:soff+Nm]))
negidx = findall(<=(0.0), x0[soff+1:soff+Nm])
isempty(negidx) || P("  sigma start <=0 at local idx ", negidx, " -> values ", x0[soff .+ negidx])
P("min over ALL x0   : ", minimum(x0), " at var ", argmin(x0))

# ---- Primal feasibility: constraint residual at x0 (no logs involved) ----
P("\n--- PRIMAL FEASIBILITY at x0 ---")
try
    cx = similar(m.meta.x0, ncon); ExaModels.cons!(m, m.meta.x0, cx); cx = Array(cx)
    lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
    viol = max.(lcon .- cx, cx .- ucon, 0.0)
    P("max constraint violation : ", maximum(viol))
    P("L2  constraint violation : ", norm(viol))
    P("# viol > 1e-8 : ", count(>(1e-8), viol), " / ", ncon)
    P("# viol > 1e-6 : ", count(>(1e-6), viol))
    P("# viol > 1e-4 : ", count(>(1e-4), viol))
    P("# viol > 1e-2 : ", count(>(1e-2), viol))
catch e
    P("cons! FAILED: ", e)
end

# ---- Objective gradient at x0 (stationarity, duals not yet involved) ----
P("\n--- OBJECTIVE GRADIENT at x0 ---")
try
    g = similar(m.meta.x0, nvar); ExaModels.grad!(m, m.meta.x0, g); g = Array(g)
    P("||grad f||_inf : ", norm(g, Inf))
    P("||grad f||_2   : ", norm(g))
catch e
    P("grad! FAILED: ", e)
end
P("\n--- OBJECTIVE VALUE at x0 ---")
obj0 = NaN
try
    obj0 = ExaModels.obj(m, m.meta.x0)
    P("obj(x0) : ", obj0)
catch e
    P("obj FAILED: ", e)
end

# ---- MadNLP solve (iter log => stdout). Catch crash. ----
P("\n--- MadNLP CPU solve (iter 0 line: inf_pr=primal infeas, inf_du=dual infeas/stationarity) ---")
try
    res = madnlp(m; tol = 1e-6, print_level = MadNLP.INFO, max_iter = 300)
    P("status=$(res.status)  obj=$(res.objective)  iter=$(res.iter)")
    isnan(obj0) || P("Δobj(x0 -> opt) = ", obj0 - res.objective)
    xopt = Array(res.solution)
    P("max|Δx| (x0 -> opt) over ALL vars      : ", maximum(abs.(x0 .- xopt)))
    P("max|Δp| (x0 -> opt) over PARAMETERS only: ", maximum(abs.(x0[1:Np] .- xopt[1:Np])))
catch e
    P("madnlp FAILED: ", e)
end
