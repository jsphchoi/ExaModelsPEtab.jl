using ExaModels
using PEtab
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE

const SRC = joinpath(@__DIR__, "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","userfuncs.jl")
    include(joinpath(SRC, f))
end

model = length(ARGS) >= 1 ? ARGS[1] : "Crauste_CellSystems2017"
K     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
yaml  = joinpath(@__DIR__, "examples", "Benchmark-Models", model, model * ".yaml")
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)

println("############# $model, K = $K #############")

# ---- build, recording ncon block boundaries -----------------------------
c = ExaModels.ExaCore(; concrete = Val(true))
c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
n0 = c.ncon                                   # 0
c = _create_collocation(c, PEmodel, PEprob, PEinfo)
n1 = c.ncon                                   # collocation + cv eqs
c = _create_continuity(c, PEmodel, PEprob, PEinfo)
n2 = c.ncon                                   # + continuity
c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)
n3 = c.ncon                                   # + y + sigma

m = ExaModels.ExaModel(c)
ExaModels.set_start!(m, c.y, y0)
ExaModels.set_start!(m, c.sigma, sigma0)

(; Np, Nz, Nc, Ncv, Nm, N, K) = PEinfo
@show Np Nz Nc Ncv Nm N
@show m.meta.nvar m.meta.ncon

# ---- (1) parameter-scale consistency -------------------------------------
println("\n--- parameter scale chain ---")
p0_var   = _var_starts(c, c.p)                                   # p decision-var start
x_est    = PEtab.get_x(PEprob)                                   # nominal on estimation scale (used in ODE solve)
xnom_lin = PEtab.transform_x(Array(PEprob.xnominal), PEprob.xnames, PEprob.model_info.xindices; to_xscale=false)
xnames   = PEprob.xnames
println(rpad("param",22), rpad("p0 (var start)",18), rpad("xnominal->lin",18), "x_est(ODE scale)")
for i in 1:Np
    println(rpad(string(xnames[i]),22), rpad(string(round(p0_var[i],sigdigits=8)),18),
            rpad(string(round(xnom_lin[i],sigdigits=8)),18), round(Float64(x_est[i]),sigdigits=8))
end
println("max|p0 - xnominal_lin| = ", maximum(abs.(p0_var .- xnom_lin)))

# ---- (2) constraint residuals at warm start ------------------------------
x0 = m.meta.x0
cx = similar(x0, m.meta.ncon)
ExaModels.cons!(m, x0, cx)
cx = Array(cx)
lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
println("\nall constraints equality (lcon==ucon==0)? ",
        all(lcon .== 0) && all(ucon .== 0))
# residual = violation of g(x) within [lcon,ucon]; equalities -> |g(x)|
viol = max.(lcon .- cx, cx .- ucon, 0.0)

blk(lo,hi) = hi > lo ? maximum(view(viol, lo+1:hi)) : NaN
println("\n--- max constraint violation at warm start, by block ---")
println("collocation + cv  [", n0+1, ":", n1, "]  max|viol| = ", blk(n0,n1))
println("continuity        [", n1+1, ":", n2, "]  max|viol| = ", blk(n1,n2))
println("y + sigma         [", n2+1, ":", n3, "]  max|viol| = ", blk(n2,n3))
println("OVERALL max|viol| = ", maximum(viol))

# When all observables are simple (state obs + fixed/p sigma), the objective block is
# ordered all-y then all-sigma, so split at n2+Nm to localize the residual.
if n3 - n2 == 2*Nm
    ymax = blk(n2, n2+Nm); smax = blk(n2+Nm, n3)
    println("\n   ↳ y constraints   [", n2+1, ":", n2+Nm, "]  max|viol| = ", ymax)
    println("   ↳ sigma constraints [", n2+Nm+1, ":", n3, "]  max|viol| = ", smax)
    # worst few in whichever sub-block is bad
    yv = view(viol, n2+1:n2+Nm); sv = view(viol, n2+Nm+1:n3)
    bad = ymax >= smax ? yv : sv
    order = sortperm(Array(bad); rev=true)[1:min(6,length(bad))]
    println("   worst ", (ymax>=smax ? "y" : "sigma"), " midx => viol:")
    for k in order
        println("      midx=", k, "  viol=", round(bad[k], sigdigits=6))
    end
end

println("\nobjective at warm start = ", ExaModels.obj(m, x0))
