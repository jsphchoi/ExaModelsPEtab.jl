# Verifies the noise-model (sigma) refactor in objective.jl across the five PEtab forms.
# For each model it builds the ExaModel and checks:
#   (1) build succeeds,
#   (2) whether the "does not follow an expected PEtab noise form" fallback @warn fired
#       (forms 3-5 must take the reduce-obs->Y path => NO warning),
#   (3) max |violation| of the sigma constraints at the warm start (should be ~0:
#       sigma0 is computed by the same reduced formula the constraint enforces, and
#       y is warm-started to y0, so the residual must vanish to float precision).
#
# Usage: julia --project=. scratch_sigma_test.jl
using ExaModels
using PEtab
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE
using Logging

const SRC = joinpath(@__DIR__, "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","userfuncs.jl")
    include(joinpath(SRC, f))
end

# (model, form description)
const CASES = [
    ("Perelson_Science1996",        "form 2: sigma = theta (single param)        [Case B]"),
    ("Smith_BMCSystBiol2013",       "form 1: sigma = c (constant)                 [Case A]"),
    ("Armistead_CellDeathDis2024",  "form 3: sigma = beta * y (proportional)      [reduce]"),
    ("Raia_CancerResearch2011",     "form 4: sigma = alpha + beta * y (affine)    [reduce]"),
    ("Liu_IFACPapersOnLine2025",    "form 5: sigma = sqrt(a^2 + (b*y)^2)          [reduce]"),
]

# A logger that forwards everything but records whether our fallback warning fired.
mutable struct WarnCatcher <: AbstractLogger
    inner::AbstractLogger
    hit::Bool
end
Logging.min_enabled_level(l::WarnCatcher) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::WarnCatcher, args...) = Logging.shouldlog(l.inner, args...)
Logging.catch_exceptions(l::WarnCatcher) = Logging.catch_exceptions(l.inner)
function Logging.handle_message(l::WarnCatcher, level, message, _module, group, id, file, line; kwargs...)
    if level == Logging.Warn && occursin("does not follow an expected PEtab noise form", string(message))
        l.hit = true
    end
    Logging.handle_message(l.inner, level, message, _module, group, id, file, line; kwargs...)
end

function run_case(model::String, K::Int)
    yaml    = joinpath(@__DIR__, "examples", "Benchmark-Models", model, model * ".yaml")
    PEmodel = PEtab.PEtabModel(yaml)
    PEprob  = PEtab.PEtabODEProblem(PEmodel)

    catcher = WarnCatcher(global_logger(), false)
    c = ExaModels.ExaCore(; concrete = Val(true))
    c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
    c = _create_collocation(c, PEmodel, PEprob, PEinfo)
    c = _create_continuity(c, PEmodel, PEprob, PEinfo)
    n2 = c.ncon                                   # constraints before objective block
    c, y0, sigma0 = with_logger(catcher) do
        _create_objective(c, PEmodel, PEprob, PEinfo)
    end
    n3 = c.ncon

    m = ExaModels.ExaModel(c)
    ExaModels.set_start!(m, c.y, y0)
    ExaModels.set_start!(m, c.sigma, sigma0)

    x0 = m.meta.x0
    cx = similar(x0, m.meta.ncon); ExaModels.cons!(m, x0, cx); cx = Array(cx)
    lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
    viol = max.(lcon .- cx, cx .- ucon, 0.0)
    obj_viol = (n3 > n2) ? maximum(view(viol, n2+1:n3)) : 0.0   # y + sigma block
    all_viol = maximum(viol)
    (; warned = catcher.hit, obj_viol, all_viol, Nm = PEinfo.Nm, nvar = m.meta.nvar, ncon = m.meta.ncon)
end

cases = isempty(ARGS) ? CASES : filter(c -> any(a -> occursin(a, c[1]), ARGS), CASES)

println(rpad("model", 30), rpad("warn?", 7), rpad("y+sigma|viol|", 16), rpad("all|viol|", 16), "form")
println("-"^110)
for (model, desc) in cases
    try
        r = run_case(model, 2)
        flag = r.warned ? "WARN" : "ok"
        println(rpad(model, 30), rpad(flag, 7),
                rpad(string(round(r.obj_viol, sigdigits=4)), 16),
                rpad(string(round(r.all_viol, sigdigits=4)), 16), desc)
    catch e
        println(rpad(model, 30), rpad("ERR", 7), "  ", sprint(showerror, e)[1:min(end,80)])
    end
end
