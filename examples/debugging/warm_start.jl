# warm_start.jl — checks warm-start constraint feasibility by block and verifies
# the parameter scale chain (decision-var start vs PEtab nominal).
#
# Usage (from repo root):
#   julia --project=examples examples/debugging/warm_start.jl [model] [K]
# Defaults: Crauste_CellSystems2017, K=2

using ExaModelsPEtab, PEtab, ExaModels
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
const SRCDIR   = joinpath(@__DIR__, "..", "..", "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","userfuncs.jl")
    include(joinpath(SRCDIR, f))
end

model = length(ARGS) >= 1 ? ARGS[1] : "Crauste_CellSystems2017"
K     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2

println("=== warm_start: $model (K=$K) ===")
d    = joinpath(MODELDIR, model)
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)

c = ExaModels.ExaCore(; concrete=Val(true))
c, PEinfo = _create_variables(c, PEmodel, PEprob, K);                  n0 = c.ncon
c = _create_collocation(c, PEmodel, PEprob, PEinfo);                   n1 = c.ncon
c = _create_continuity(c, PEmodel, PEprob, PEinfo);                    n2 = c.ncon
c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo);         n3 = c.ncon
m = ExaModels.ExaModel(c)
ExaModels.set_start!(m, c.y, y0)
ExaModels.set_start!(m, c.sigma, sigma0)

(; Np, Nz, Nc, Ncv, Nm, N) = PEinfo
println("Np=$Np  Nz=$Nz  Nc=$Nc  Ncv=$Ncv  Nm=$Nm  N=$N")
println("nvar=$(m.meta.nvar)  ncon=$(m.meta.ncon)  DoF=$(m.meta.nvar - m.meta.ncon)")

# Parameter scale: decision-var start vs PEtab nominal
println("\n--- parameter scale ---")
println(rpad("param", 24), rpad("p0 (var start)", 18), "x_est (PEtab nominal)")
x0_all = Array(m.meta.x0)
x_est  = Array(PEtab.get_x(PEprob))
for i in 1:Np
    println(rpad(string(PEprob.xnames[i]), 24),
            rpad(string(round(x0_all[i]; sigdigits=8)), 18),
            round(x_est[i]; sigdigits=8))
end

# Constraint violations at warm start
x0 = m.meta.x0
cx = similar(x0, m.meta.ncon); ExaModels.cons!(m, x0, cx); cx = Array(cx)
lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
viol = max.(lcon .- cx, cx .- ucon, 0.0)
blk(lo, hi) = hi > lo ? maximum(view(viol, lo+1:hi)) : NaN

println("\n--- max constraint violation at warm start, by block ---")
println("collocation + cv  [$(n0+1):$n1]  max|viol| = ", blk(n0, n1))
println("continuity        [$(n1+1):$n2]  max|viol| = ", blk(n1, n2))
println("y + sigma         [$(n2+1):$n3]  max|viol| = ", blk(n2, n3))
println("OVERALL                          max|viol| = ", maximum(viol))
exa_obj    = ExaModels.obj(m, x0)
petab_nllh = PEprob.nllh(PEtab.get_x(PEprob))
println("\n--- objective consistency ---")
println("objective at warm start = ", exa_obj)
println("PEtab nllh at nominal   = ", petab_nllh)
println("reldiff                 = ", abs(exa_obj - petab_nllh) / abs(petab_nllh))
println("== DONE ==")
