# viol_decompose.jl — decompose warm-start constraint violation by block.
# Tests the hypothesis "event models fail because continuity is wrongly enforced
# across the event boundary." If true, the CONTINUITY block carries the violation.
# If the violation lives in the COLLOCATION block instead, the issue is mesh
# resolution of the post-event transient, NOT continuity.
#
# Build order (see userfuncs.jl) fixes the constraint-row blocks:
#   collocation rows : 1 .. ncon_coll
#   continuity rows  : ncon_coll+1 .. ncon_cont   (interval continuity + initial conditions)
#   objective aux    : ncon_cont+1 .. ncon
#
# Usage: julia --project=. examples/debugging/viol_decompose.jl [model] [K]

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

model = length(ARGS) >= 1 ? ARGS[1] : "Fujita_SciSignal2010"
K     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4

d    = joinpath(MODELDIR, model)
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)
println("PEtab nllh(nominal θ) = ", PEprob.nllh(Array(PEtab.get_x(PEprob))))

c = ExaModels.ExaCore(; backend=nothing, concrete=Val(true))
c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
c = _create_collocation(c, PEmodel, PEprob, PEinfo)
ncon_coll = c.ncon
c = _create_continuity(c, PEmodel, PEprob, PEinfo)
ncon_cont = c.ncon
c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)

m = ExaModels.ExaModel(c)
ExaModels.set_start!(m, c.y, y0)
ExaModels.set_start!(m, c.sigma, sigma0)

x0 = m.meta.x0; cx = similar(x0, m.meta.ncon)
ExaModels.cons!(m, x0, cx); cx = Array(cx)
lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
viol = max.(lcon .- cx, cx .- ucon, 0.0)

blkmax(r) = isempty(r) ? 0.0 : maximum(viol[r])
blkarg(r) = isempty(r) ? 0 : (first(r) - 1 + argmax(viol[r]))

coll_r = 1:ncon_coll
cont_r = (ncon_coll+1):ncon_cont
aux_r  = (ncon_cont+1):m.meta.ncon

println("=== viol_decompose: $model (K=$K) ===")
println("warm obj = $(ExaModels.obj(m, x0))")
println("total max|viol|     = $(maximum(viol))")
println("--------------------------------------------------")
println("collocation block   rows $(coll_r):  max|viol| = $(blkmax(coll_r))  @row $(blkarg(coll_r))")
println("continuity  block   rows $(cont_r):  max|viol| = $(blkmax(cont_r))  @row $(blkarg(cont_r))")
println("objective aux block rows $(aux_r):  max|viol| = $(blkmax(aux_r))  @row $(blkarg(aux_r))")
