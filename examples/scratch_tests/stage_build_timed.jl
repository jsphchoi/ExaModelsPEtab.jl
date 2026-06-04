# stage_build_timed.jl — TIMES each model-construction stage to locate compile
# blowups (PEtabModel → PEtabODEProblem → variables → collocation → continuity →
# objective → ExaModel). CPU backend (backend=nothing) so it does NOT touch the GPU.
#
# A warmup model is built first to pay the Julia/Symbolics JIT, so the per-stage
# numbers for the real targets are codegen-time, not first-call-compile time.
#
# Usage (from repo root):
#   julia --project=. examples/scratch_tests/stage_build_timed.jl [K] [model1] [model2 ...]
#   defaults: K=2, models = Perelson_Science1996 (fast baseline) + SalazarCavazos_MBoC2020

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

const WARMUP_MODEL = "Boehm_JProteomeRes2014"   # small, has expression observables (warms build_function JIT)

yaml_of(model) = begin
    d = joinpath(MODELDIR, model)
    joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
end

function build_timed(model, K; quiet=false)
    yaml = yaml_of(model)
    # print+flush AFTER EACH STAGE so the slow stage reveals itself live (output
    # stops at the culprit) instead of batching at the end.
    pr(n, t) = quiet || (println("  ", rpad(n, 24), round(t; digits=2), "s"); flush(stdout))
    quiet || (println("\n=== $model (K=$K) ==="); flush(stdout))
    t0 = time(); PEmodel = PEtab.PEtabModel(yaml);          pr("PEtabModel", time()-t0)
    t0 = time(); PEprob  = PEtab.PEtabODEProblem(PEmodel);  pr("PEtabODEProblem", time()-t0)
    c = ExaModels.ExaCore(; backend=nothing, concrete=Val(true))
    t0 = time(); (c, PEinfo)     = _create_variables(c, PEmodel, PEprob, K);  pr("_create_variables[mesh]", time()-t0)
    t0 = time(); c               = _create_collocation(c, PEmodel, PEprob, PEinfo); pr("_create_collocation", time()-t0)
    t0 = time(); c               = _create_continuity(c, PEmodel, PEprob, PEinfo);  pr("_create_continuity", time()-t0)
    t0 = time(); (c, y0, sigma0) = _create_objective(c, PEmodel, PEprob, PEinfo);   pr("_create_objective", time()-t0)
    t0 = time(); m               = ExaModels.ExaModel(c);                           pr("ExaModel(c)", time()-t0)
    quiet || (println("  -> nvar=$(m.meta.nvar) ncon=$(m.meta.ncon)"); flush(stdout))
    return 0.0
end

K      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
models = length(ARGS) >= 2 ? ARGS[2:end] : ["Perelson_Science1996", "SalazarCavazos_MBoC2020"]

println("warmup build on $WARMUP_MODEL (K=2) to pay JIT ...")
build_timed(WARMUP_MODEL, 2; quiet=true)
println("warmup done.\n")

for m in models
    try
        build_timed(m, K)
    catch e
        println("  !! $m FAILED: ", sprint(showerror, e))
    end
end
println("\nstage_build_timed complete.")
