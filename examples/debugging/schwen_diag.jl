# schwen_diag.jl — CPU obj-consistency + active-bound dump for Schwen.
# (1) Does exa's NLL at nominal θ match PEtab's nllh? (confirms σ/log10 formulation, no sqrt)
# (2) Which decision variables sit on a finite box bound at the nominal optimum (the active set)?
using ExaModelsPEtab, PEtab, ExaModels
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE
const SRCDIR = joinpath(@__DIR__, "..", "..", "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","steadystate.jl","userfuncs.jl")
    include(joinpath(SRCDIR, f))
end
const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
model = "Schwen_PONE2014"
yaml  = joinpath(MODELDIR, model, model*".yaml")
K = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5

PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)
xnom    = PEtab.get_x(PEprob)
petab_nllh = PEprob.nllh(xnom)

m   = _build_petab_examodel(PEmodel, PEprob, nothing, K)
x0  = Array(m.meta.x0)
exa0 = ExaModels.obj(m, m.meta.x0)
println("=== OBJ CONSISTENCY (Schwen, K=$K, nominal θ) ===")
println("  PEtab nllh   = $petab_nllh")
println("  exa obj(x0)  = $exa0")
println("  reldiff      = $(abs(exa0-petab_nllh)/abs(petab_nllh))")

# active set among FINITE-bound variables (these are the parameters/aux, not the unbounded states)
lvar = Array(m.meta.lvar); uvar = Array(m.meta.uvar)
nfin = 0; atL = 0; atU = 0
for i in 1:length(x0)
    (isfinite(lvar[i]) || isfinite(uvar[i])) || continue
    nfin += 1
    if isfinite(lvar[i]) && x0[i]-lvar[i] <= 1e-6*(1+abs(lvar[i])); atL += 1; end
    if isfinite(uvar[i]) && uvar[i]-x0[i] <= 1e-6*(1+abs(uvar[i])); atU += 1; end
end
println("=== ACTIVE BOUNDS AT NOMINAL OPTIMUM ===")
println("  finite-bound vars=$nfin   at_lower=$atL   at_upper=$atU   total_active=$(atL+atU)")
println("  (cross-check vs param-table comb: Schwen has 6 estimated params on bounds:")
println("   fragments↑, km_nExpID1-4↑, scaleElisa_nExpID3↓, scaleElisa_nExpID4↑)")
