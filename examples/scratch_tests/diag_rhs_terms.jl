using ExaModelsPEtab, PEtab, ExaModels
import Symbolics
import ModelingToolkitBase as MTK

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
get_yaml(m) = begin
    d = joinpath(MODELDIR, m)
    fs = filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))
    joinpath(d, first(fs))
end

# top-level additive term count of a symbolic expr
function nterms(ex)
    v = Symbolics.value(ex)
    if Symbolics.iscall(v) && Symbolics.operation(v) === (+)
        return length(Symbolics.arguments(v))
    end
    return 1
end

m = ARGS[1]
yaml = get_yaml(m)
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)

sys = PEprob.model_info.model.sys
f_exprs_raw = [eqn.rhs for eqn in MTK.equations(sys)]
dict_fixed_val = ExaModelsPEtab._resolve_fixed_vals(PEmodel, PEprob)
subst_rules = ExaModelsPEtab._assignment_substitutor(PEprob; bare = false)
f_exprs = [Symbolics.substitute(subst_rules(fr), dict_fixed_val) for fr in f_exprs_raw]

function report(f_exprs)
    println("Nz = $(length(f_exprs))")
    println("state | top_+_terms | expr_chars")
    worst = (0, 0, 0)
    for (v, fx) in enumerate(f_exprs)
        nt = nterms(fx)
        nchars = length(string(Symbolics.value(fx)))
        nchars > worst[3] && (worst = (v, nt, nchars))
        println("$v | $nt | $nchars")
    end
    println("\nWORST state $(worst[1]): $(worst[2]) top-level + terms, $(worst[3]) chars")
end
report(f_exprs)
