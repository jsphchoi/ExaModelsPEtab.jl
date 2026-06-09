# test_event_guard.jl — verify the SBML-<event> guard fires for true-event models (Liu/Smith) and
# passes for fixed-time-gate (Fujita) and no-event (Boehm) models.
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
_yaml(m) = joinpath(MODELDIR, m, first(filter(f->endswith(lowercase(f),".yaml"), readdir(joinpath(MODELDIR,m)))))
isguard(e) = occursin("does not support SBML <event>", sprint(showerror, e))

# (A) wiring + logic: full build path must ERROR at the guard for true-event models
println("--- expect GUARD to FIRE (true SBML <event> models) ---")
for m in ["Liu_IFACPapersOnLine2025", "Smith_BMCSystBiol2013"]
    PEmodel = PEtab.PEtabModel(_yaml(m)); PEprob = PEtab.PEtabODEProblem(PEmodel)
    try
        _build_petab_examodel(PEmodel, PEprob, nothing, 2)
        println("  $m : NO error  <<< UNEXPECTED (guard did not fire)")
    catch e
        println("  $m : ", isguard(e) ? "GUARD FIRED ✓" : "OTHER error <<< : $(sprint(showerror,e)[1:90])")
    end
end

# (B) logic: guard must PASS for fixed-time-gate and no-event models
println("--- expect guard to PASS (fixed-time gate / no events) ---")
for m in ["Fujita_SciSignal2010", "Boehm_JProteomeRes2014"]
    PEmodel = PEtab.PEtabModel(_yaml(m))
    try
        _assert_supported_events(PEmodel)
        println("  $m : guard pass ✓")
    catch e
        println("  $m : GUARD FIRED <<< UNEXPECTED : $(sprint(showerror,e)[1:90])")
    end
end
