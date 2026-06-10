# check_cv_mapping.jl — cheap (no ExaModel build) verification of the x0SSpre cv-column fix.
# Confirms cv now carries a pre-eq column and dict_cidx_sscidx points the SS residual at it.
using ExaModelsPEtab, PEtab, ExaModels
import ModelingToolkitBase as MTK
using Symbolics
const SRC = joinpath(@__DIR__, "..", "..", "src")
for f in ("structs.jl","constants.jl","utils.jl"); include(joinpath(SRC, f)); end

model = length(ARGS) >= 1 ? ARGS[1] : "Zheng_PNAS2012"
d = joinpath(@__DIR__, "..", "Benchmark-Models", model)
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)

sim_cids   = _get_cids(PEmodel)
cv_ids     = _get_cv_cond_ids(PEmodel, PEprob)
cv_rows    = _get_cv_cond_rows(PEmodel, PEprob)
cv_cols    = _get_cv_colnames(PEmodel)
dict_ss    = _get_dict_cidx_sscidx(PEmodel, PEprob)
conds      = PEmodel.petab_tables[:conditions]

println("=== $model ===")
println("cv columns (Ncv): ", cv_cols)
println("sim conditions (cv cols 1:Nc): ", sim_cids)
println("ALL cv-cond ids (sim + extra pre-eq): ", cv_ids)
println("dict_cidx_sscidx (sim cidx -> cv column of its pre-eq cond): ", dict_ss)
println("\nper sim condition: pre-eq cv column & its values")
for cidx in eachindex(sim_cids)
    sscol = dict_ss[cidx]
    vals = [conds[cv_rows[sscol], col] for col in cv_cols]
    simvals = [conds[cv_rows[cidx], col] for col in cv_cols]
    println("  sim '$(sim_cids[cidx])' (col $cidx, cv=$simvals)  ->  pre-eq '$(cv_ids[sscol])' (col $sscol, cv=$vals)")
end
println("\nEXPECT: SS residual now uses the PRE-EQ column's cv (e.g. Zheng dilution=1), not the sim's (dilution=0).")
