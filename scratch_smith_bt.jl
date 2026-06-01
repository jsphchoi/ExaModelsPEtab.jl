using ExaModels, PEtab
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE
const ROOT = @__DIR__
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl","variables.jl","collocation.jl","continuity.jl","objective.jl","userfuncs.jl")
    include(joinpath(ROOT, "src", f))
end
yaml = joinpath(ROOT,"examples","Benchmark-Models","Smith_BMCSystBiol2013","Smith_BMCSystBiol2013.yaml")
PEmodel = PEtab.PEtabModel(yaml); PEprob = PEtab.PEtabODEProblem(PEmodel)
c = ExaModels.ExaCore(; concrete = Val(true))
c, PEinfo = _create_variables(c, PEmodel, PEprob, 2)
try
    global c = _create_collocation(c, PEmodel, PEprob, PEinfo); println("STAGE collocation OK")
    global c = _create_continuity(c, PEmodel, PEprob, PEinfo); println("STAGE continuity OK")
    global c, y0, s0 = _create_objective(c, PEmodel, PEprob, PEinfo); println("STAGE objective OK")
catch e
    println("STAGE FAILED (after last OK above)")
    showerror(stdout, e); println()
    for fr in stacktrace(catch_backtrace())[1:min(end,20)]; println("  ", fr); end
end
