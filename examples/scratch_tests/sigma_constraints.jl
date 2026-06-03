# sigma_constraints.jl — verifies sigma constraint feasibility at warm start
# across the five PEtab noise forms. Also checks whether the fallback warning
# ("does not follow an expected PEtab noise form") fires unexpectedly.
#
# Usage (from repo root):
#   julia --project=. examples/scratch_tests/sigma_constraints.jl [K]
# Default: K=2

using ExaModelsPEtab, PEtab, ExaModels, Logging
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

K = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2

const CASES = [
    ("Smith_BMCSystBiol2013",      "form 1: sigma = c (constant)"),
    ("Perelson_Science1996",       "form 2: sigma = theta (parameter)"),
    ("Armistead_CellDeathDis2024", "form 3: sigma = beta * y (proportional)"),
    ("Raia_CancerResearch2011",    "form 4: sigma = alpha + beta * y (affine)"),
    ("Liu_IFACPapersOnLine2025",   "form 5: sigma = sqrt(a^2 + (b*y)^2)"),
]

mutable struct WarnCatcher <: AbstractLogger
    inner::AbstractLogger
    hit::Bool
end
Logging.min_enabled_level(l::WarnCatcher) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::WarnCatcher, a...) = Logging.shouldlog(l.inner, a...)
Logging.catch_exceptions(l::WarnCatcher) = Logging.catch_exceptions(l.inner)
function Logging.handle_message(l::WarnCatcher, level, msg, mod, grp, id, file, line; kw...)
    level == Logging.Warn && occursin("does not follow an expected PEtab noise form", string(msg)) && (l.hit = true)
    Logging.handle_message(l.inner, level, msg, mod, grp, id, file, line; kw...)
end

println("=== sigma_constraints (K=$K) ===\n")
println(rpad("model", 30), rpad("warn?", 7), rpad("y+σ |viol|", 16), rpad("all |viol|", 16), "noise form")
println("-"^100)

for (model, desc) in CASES
    try
        d    = joinpath(MODELDIR, model)
        yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
        PEmodel = PEtab.PEtabModel(yaml)
        PEprob  = PEtab.PEtabODEProblem(PEmodel)

        c = ExaModels.ExaCore(; concrete=Val(true))
        c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
        c = _create_collocation(c, PEmodel, PEprob, PEinfo)
        c = _create_continuity(c, PEmodel, PEprob, PEinfo);  n2 = c.ncon
        catcher = WarnCatcher(global_logger(), false)
        c, y0, sigma0 = with_logger(catcher) do
            _create_objective(c, PEmodel, PEprob, PEinfo)
        end;  n3 = c.ncon
        m = ExaModels.ExaModel(c)
        ExaModels.set_start!(m, c.y, y0)
        ExaModels.set_start!(m, c.sigma, sigma0)

        x0 = m.meta.x0; cx = similar(x0, m.meta.ncon)
        ExaModels.cons!(m, x0, cx); cx = Array(cx)
        lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
        viol = max.(lcon .- cx, cx .- ucon, 0.0)
        obj_viol = n3 > n2 ? maximum(view(viol, n2+1:n3)) : 0.0

        println(rpad(model, 30), rpad(catcher.hit ? "WARN" : "ok", 7),
                rpad(string(round(obj_viol; sigdigits=4)), 16),
                rpad(string(round(maximum(viol); sigdigits=4)), 16), desc)
    catch e
        println(rpad(model, 30), rpad("ERR", 7), "  ", sprint(showerror, e)[1:min(end, 80)])
    end
end
