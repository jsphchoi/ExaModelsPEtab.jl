# residual_check.jl — cheap end-to-end check of the x0SSpre cv fix (no ExaModel build).
# The pre-equilibrated steady state zss (PEtab's u0) must satisfy f(zss)=0 under the PRE-EQ
# inputs. Evaluate our RHS at zss with the PRE-EQ cv vs the (old, buggy) SIM cv:
#   ||f(zss; preeq_cv)|| ~ 0   (FIX: residual satisfied)
#   ||f(zss; sim_cv)||  >> 0   (OLD BUG: residual violated by the correct IC)
using ExaModelsPEtab, PEtab, ExaModels
import ModelingToolkitBase as MTK
import OrdinaryDiffEq as ODE
using Symbolics
using PEtab: get_x
const SRC = joinpath(@__DIR__, "..", "..", "src")
for f in ("structs.jl","constants.jl","utils.jl"); include(joinpath(SRC, f)); end

model = length(ARGS) >= 1 ? ARGS[1] : "Zheng_PNAS2012"
d = joinpath(@__DIR__, "..", "Benchmark-Models", model)
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)

Nz       = Int(PEprob.model_info.nstates)
pscale   = _get_pscale(PEprob)
θ        = Array(get_x(PEprob))
p_phys   = [_p_phys_val(θ, m, pscale) for m in 1:length(θ)]
cv_cols  = _get_cv_colnames(PEmodel)
conds    = PEmodel.petab_tables[:conditions]
cv_rows  = _get_cv_cond_rows(PEmodel, PEprob)
cv_ids   = _get_cv_cond_ids(PEmodel, PEprob)
sim_cids = _get_cids(PEmodel)
dict_ss  = _get_dict_cidx_sscidx(PEmodel, PEprob)

gate_syms = _get_gate_syms(PEprob)
fs = _get_rhs_funcs(PEmodel, PEprob, gate_syms)   # f[v](z..., p..., cv..., gates..., t)
Ngate = length(gate_syms)

# pre-equilibrated zss (correct IC) for each sim condition, from PEtab's get_odeproblem u0
si = PEprob.model_info.simulation_info
sim_ids = si.conditionids[:simulation]; preeq_ids = si.conditionids[:pre_equilibration]
cvvals(col) = [Float64(conds[cv_rows[col], c]) for c in cv_cols]

println("=== $model : SS residual at pre-equilibrated zss ===")
for cidx in eachindex(sim_cids)
    cid = Symbol(sim_cids[cidx]); pos = findfirst(==(cid), sim_ids)
    pos === nothing && continue
    oprob, _ = PEtab.get_odeproblem(get_x(PEprob), PEprob; condition = preeq_ids[pos] => sim_ids[pos])
    zss = Array(oprob.u0[1:Nz])
    sscol  = dict_ss[cidx]
    preeq_cv = cvvals(sscol); sim_cv = cvvals(cidx)
    resid(cvv) = [fs[v](zss..., p_phys..., cvv..., ntuple(_->0.0,Ngate)..., 0.0) for v in 1:Nz]
    rp = resid(preeq_cv); rs = resid(sim_cv)
    println("cond '$(sim_cids[cidx])': preeq_cv($(cv_ids[sscol]))=$preeq_cv  sim_cv=$sim_cv")
    println("   ||f(zss; PREEQ cv)|| = $(maximum(abs, rp))   (FIX -> ~0)")
    println("   ||f(zss; SIM   cv)|| = $(maximum(abs, rs))   (old bug -> >>0 if cv differs)")
end
