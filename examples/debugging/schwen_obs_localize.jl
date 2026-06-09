# schwen_obs_localize.jl — per-observable diff of exa's warm-start observables (y0) vs PEtab's
# simulatedData at nominal, to localize the 1.3% Schwen objective discrepancy.
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
const MD = joinpath(MODELDIR, "Schwen_PONE2014")
yaml = joinpath(MD, "Schwen_PONE2014.yaml")
PEmodel = PEtab.PEtabModel(yaml); PEprob = PEtab.PEtabODEProblem(PEmodel)

# build step-by-step to capture y0 (warm-start observable values exa uses)
c = ExaModels.ExaCore(; backend=nothing, concrete = Val(true))
c, PEinfo = _create_variables(c, PEmodel, PEprob, 2)
c = _create_collocation(c, PEmodel, PEprob, PEinfo)
c = _create_continuity(c, PEmodel, PEprob, PEinfo)
c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)

meas = PEmodel.petab_tables[:measurements]
Nm = size(meas, 1)
obsid(i) = string(meas[i, :observableId])

# parse simulatedData TSV for the 'simulation' column (PEtab's simulated obs at nominal)
simf = joinpath(MD, "simulatedData_Schwen_PONE2014.tsv")
lines = readlines(simf); hdr = split(lines[1], '\t')
scol = findfirst(==("simulation"), hdr)
sim = [parse(Float64, split(lines[1+i], '\t')[scol]) for i in 1:Nm]

# per-observable: y0-vs-sim error, exa's sigma0, and exa's NLL contribution (documented log10 formula)
ymeas(i) = Float64(meas[i, :measurement])
LN10 = log(10.0); HALF = 0.5*log(2π)
nll_term(i) = 0.5*((log10(y0[i]) - log10(ymeas(i)))/sigma0[i])^2 + log(sigma0[i]) + HALF + log(ymeas(i)) + log(LN10)
println("per-observable: y0-vs-sim, exa sigma0, exa NLL contribution")
println(rpad("observable",20), rpad("n",5), rpad("max|y0relerr|",14), rpad("sigma0(uniq)",16), "exa_NLL_sum")
total = 0.0
for ob in unique(obsid.(1:Nm))
    idxs = [i for i in 1:Nm if obsid(i)==ob]
    rel  = maximum(abs(y0[i]-sim[i])/(abs(sim[i])+1e-30) for i in idxs)
    sigs = unique(round.([sigma0[i] for i in idxs]; sigdigits=6))
    nllsum = sum(nll_term(i) for i in idxs); global total += nllsum
    println(rpad(ob,20), rpad(length(idxs),5), rpad(round(rel;sigdigits=3),14), rpad(string(sigs),16), round(nllsum;digits=4))
end
println("TOTAL exa NLL (manual) = $(round(total;digits=4))   [should match exa obj 943.999;  PEtab nllh = 956.518]")
println("expected sigma: IRsum->IR_obs_std=0.04719,  others->std=0.24832")
