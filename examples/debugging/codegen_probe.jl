# codegen_probe.jl — diagnose the Zheng (x0SSpre) build slowness WITHOUT the slow build_function.
# Replicates _get_rhs_funcs up to the substituted RHS exprs, then reports per-equation:
#   - expression size (string length proxy for tree size) -> reveals a rule-inlining blowup
#   - LEAKING free variables (symbols not in all_syms) -> the "undefined symbolics variable" check
# Also times the PEtab build and the substitution step separately to localize the cost.
using ExaModelsPEtab, PEtab, ExaModels
import ModelingToolkitBase as MTK
using Symbolics
const SRC = joinpath(@__DIR__, "..", "..", "src")
for f in ("structs.jl","constants.jl","utils.jl"); include(joinpath(SRC, f)); end

model = length(ARGS) >= 1 ? ARGS[1] : "Zheng_PNAS2012"
d = joinpath(@__DIR__, "..", "Benchmark-Models", model)
yaml = joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
println("=== codegen_probe: $model ===")
PEmodel = PEtab.PEtabModel(yaml)
print("PEtab build: "); @time PEprob = PEtab.PEtabODEProblem(PEmodel)

gate_syms = _get_gate_syms(PEprob)
sys = PEprob.model_info.model.sys
f_exprs_raw = [eqn.rhs for eqn in MTK.equations(sys)]

print("_resolve_fixed_vals: "); @time dict_fixed_val = _resolve_fixed_vals(PEmodel, PEprob)
print("_assignment_substitutor: "); @time subst_rules = _assignment_substitutor(PEprob; bare = false)
print("substitute all RHS: "); @time f_exprs = [Symbolics.substitute(subst_rules(fr), dict_fixed_val) for fr in f_exprs_raw]

z_syms = _get_z_syms(PEprob); p_syms = _get_p_syms(PEprob); cv_syms = _get_cv_syms(PEmodel)
t_sym = Symbolics.Num(Symbolics.variable(:t))
all_syms = [z_syms; p_syms; cv_syms; gate_syms; [t_sym]]
bound = Set(string.(Symbolics.value.(all_syms)))

println("\nNz=$(length(z_syms)) Np=$(length(p_syms)) Ncv=$(length(cv_syms)) Ngate=$(length(gate_syms))")
println("rules (MTK.observed): $(length(MTK.observed(sys)))")
println("\nper-equation: size(strlen)  nvars  LEAKS")
leak_union = Set{String}()
for (i, fe) in enumerate(f_exprs)
    vars = Symbolics.get_variables(fe)
    leaks = sort(unique(string(v) for v in vars if !(string(v) in bound) && string(v) != "t"))
    for l in leaks; push!(leak_union, l); end
    sz = length(string(fe))
    flag = (sz > 5000 || !isempty(leaks)) ? "  <<<" : ""
    println("  eq $(lpad(i,2)): size=$(lpad(sz,8))  nvars=$(lpad(length(vars),4))  leaks=$(leaks)$flag")
end
println("\nALL LEAKING SYMBOLS (in any RHS, not in all_syms): ", sort(collect(leak_union)))
println("max expr size = ", maximum(length(string(fe)) for fe in f_exprs))
