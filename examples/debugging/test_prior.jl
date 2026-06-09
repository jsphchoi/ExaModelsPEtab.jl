# test_prior.jl — verify uniform & laplace prior implementations match PEtab.nllh.
# Strategy: copy Schwen (already has the objectivePrior columns + 6 verified parameterScaleNormal
# priors), ADD a new-type prior on a currently-prior-less estimated param, rebuild PEtab + exa, and
# check obj-consistency at nominal θ. If the total still matches PEtab to ~1e-7, the new prior type
# is correct (parameterScaleNormal already verified to 3e-9).
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
const BASE = joinpath(@__DIR__, "..", "Benchmark-Models", "Schwen_PONE2014")

# scenarios: (label, parameterId, parameterScale-of-that-param, objectivePriorType, objectivePriorParameters)
scenarios = [
    ("uniform / kin(log10)",      "kin",       "uniform", "1e-05;1000"),   # -log = log(b-a), constant
    ("laplace / fragments(lin)",  "fragments", "laplace", "0.5;0.2"),      # lin-scale (Raimundez/Lang case): |p-μ|/b+log2b
    ("laplace / kout(log10)",     "kout",      "laplace", "0.1;0.05"),     # log10-scale: |10^p-μ|/b+log2b
]

function edit_prior(dir, pid, ptype, pparams)
    pf = joinpath(dir, "parameters_Schwen_PONE2014.tsv")
    lines = readlines(pf); hdr = split(lines[1], '\t')
    ci_id = findfirst(==("parameterId"), hdr)
    ci_t  = findfirst(==("objectivePriorType"), hdr)
    ci_p  = findfirst(==("objectivePriorParameters"), hdr)
    for k in 2:length(lines)
        f = split(lines[k], '\t'); length(f) < length(hdr) && (f = vcat(f, fill("", length(hdr)-length(f))))
        if f[ci_id] == pid; f[ci_t] = ptype; f[ci_p] = pparams; lines[k] = join(f, '\t'); end
    end
    write(pf, join(lines, '\n') * '\n')
end

println(rpad("scenario",28), rpad("PEtab_nllh",16), rpad("exa_obj",16), "reldiff")
for (label, pid, ptype, pparams) in scenarios
    tmp = "/tmp/prior_test_" * replace(pid, r"[^a-z0-9]"i => "")
    rm(tmp; recursive=true, force=true); cp(BASE, tmp; force=true)
    edit_prior(tmp, pid, ptype, pparams)
    yaml = joinpath(tmp, "Schwen_PONE2014.yaml")
    try
        PEmodel = PEtab.PEtabModel(yaml); PEprob = PEtab.PEtabODEProblem(PEmodel)
        nllh = PEprob.nllh(PEtab.get_x(PEprob))
        m  = _build_petab_examodel(PEmodel, PEprob, nothing, 2)
        eo = ExaModels.obj(m, m.meta.x0)
        rd = abs(eo-nllh)/abs(nllh)
        println(rpad(label,28), rpad(round(nllh;digits=4),16), rpad(round(eo;digits=4),16),
                round(rd;sigdigits=3), rd < 1e-4 ? "  ✓" : "  <<< MISMATCH"); flush(stdout)
    catch e
        println(rpad(label,28), "FAILED: ", sprint(showerror,e)[1:min(120,end)]); flush(stdout)
    end
end
