# Diagnostic: time each build phase + verify gate values, for one model.
# Run from the events worktree: julia --project=examples examples/diag_gates.jl <ModelDir>
using ExaModelsPEtab, PEtab
const EP = ExaModelsPEtab

model = length(ARGS) >= 1 ? ARGS[1] : "Fujita_SciSignal2010"
yaml  = joinpath(@__DIR__, "..", "Benchmark-Models", model, model * ".yaml")
isfile(yaml) || (yaml = first(filter(isfile, [joinpath(@__DIR__, "..", "Benchmark-Models",model,f)
            for f in readdir(joinpath(@__DIR__, "..", "Benchmark-Models",model)) if endswith(f,".yaml")])))

println("== $model ==")
t = time(); PEmodel = PEtab.PEtabModel(yaml);        println("PEtabModel       $(round(time()-t;digits=1))s")
t = time(); PEprob  = PEtab.PEtabODEProblem(PEmodel); println("PEtabODEProblem  $(round(time()-t;digits=1))s")

K = 5
t = time()
zinit, Nz, N, K, Nc, t_meas, h, taus, L1 = EP._get_z_init(PEmodel, PEprob, K)
println("_get_z_init      $(round(time()-t;digits=1))s   (Nz=$Nz N=$N Nc=$Nc)")

gate_syms = EP._get_gate_syms(PEprob)
println("gate_syms (Ng=$(length(gate_syms))): ", gate_syms)

t = time()
gate_vals, gate_vals_ss = EP._get_gate_vals(PEmodel, PEprob, gate_syms, h, taus)
println("_get_gate_vals   $(round(time()-t;digits=1))s")

if length(gate_syms) >= 1
    # distinct gate patterns over (i,cidx)
    pats = Dict{Vector{Float64},Int}()
    for cidx in 1:Nc, i in 1:N
        v = gate_vals[:, i, cidx]; pats[v] = get(pats, v, 0) + 1
    end
    println("distinct gate patterns over (i,cidx): ", length(pats))
    for (v,n) in pats; println("   pattern ", v, "  count=", n); end
    println("gate_vals_ss (per condition):"); for cidx in 1:Nc; println("   cidx=$cidx -> ", gate_vals_ss[:,cidx]); end
end
println("== DONE ==")
