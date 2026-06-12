# stage_build.jl — builds a model stage by stage (variables → collocation →
# continuity → objective → ExaModel) to pinpoint exactly where a failure occurs.
# Reports nvar/ncon after each stage and checks warm-start feasibility at the end.
#
# Usage (from repo root):
#   julia --project=examples examples/debugging/stage_build.jl [model] [K]
# Defaults: Smith_BMCSystBiol2013, K=2

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

model   = length(ARGS) >= 1 ? ARGS[1] : "Smith_BMCSystBiol2013"
K       = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
backend = nothing   # change to CUDA.CUDABackend() for GPU

println("=== stage_build: $model (K=$K) ===")
d    = joinpath(MODELDIR, model)
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)

c = ExaModels.ExaCore(; backend, concrete=Val(true))

c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
println("variables   OK  nvar=$(c.nvar)  ncon=$(c.ncon)")

c = _create_collocation(c, PEmodel, PEprob, PEinfo)
println("collocation OK  nvar=$(c.nvar)  ncon=$(c.ncon)")

c = _create_continuity(c, PEmodel, PEprob, PEinfo)
println("continuity  OK  nvar=$(c.nvar)  ncon=$(c.ncon)")

c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)
println("objective   OK  nvar=$(c.nvar)  ncon=$(c.ncon)")

m = ExaModels.ExaModel(c)
ExaModels.set_start!(m, c.y, y0)
ExaModels.set_start!(m, c.sigma, sigma0)

x0 = m.meta.x0; cx = similar(x0, m.meta.ncon)
ExaModels.cons!(m, x0, cx); cx = Array(cx)
lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
max_viol = maximum(max.(lcon .- cx, cx .- ucon, 0.0))

println("ExaModel    OK  nvar=$(m.meta.nvar)  ncon=$(m.meta.ncon)  DoF=$(m.meta.nvar - m.meta.ncon)")
println("warm-start: obj=$(ExaModels.obj(m, x0))  max|viol|=$max_viol")
