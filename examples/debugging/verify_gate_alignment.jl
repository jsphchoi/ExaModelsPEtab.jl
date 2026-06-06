# verify_gate_alignment.jl — for one event model, dump the exact gate (event-parameter) profile
# over the horizon and VERIFY that gate_vals[:,i,cidx] (the single value our collocation feeds to
# every node of interval i) equals PEtab's actual gate value at EACH collocation time t_ij.
#
#   julia --project=. examples/debugging/verify_gate_alignment.jl Giordano_Nature2020 [K]
#
# CPU only (PEtab + a stepped integrator) — does NOT build the ExaModel, so it won't touch the GPU.
using ExaModelsPEtab, PEtab
import Symbolics
import OrdinaryDiffEq as ODE
const EP = ExaModelsPEtab

model = get(ARGS, 1, "Giordano_Nature2020")
K     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
yaml  = joinpath(@__DIR__, "..", "Benchmark-Models", model, model * ".yaml")

println("== gate alignment check: $model  (K=$K) ==")
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)

_, Nz, N, _, Nc, t_meas, t_vec_mesh, h, taus, L1 = EP._get_z_init(PEmodel, PEprob, K)
gate_syms = EP._get_gate_syms(PEprob)
Ng = length(gate_syms)
gate_vals, gate_vals_ss = EP._get_gate_vals(PEmodel, PEprob, gate_syms, h, taus)
cids = EP._get_cids(PEmodel)

h_cum = cumsum(h) .- h
bnds  = cumsum(h)
println("Ng=$Ng  N=$N  Nc=$Nc  K=$K")
println("taus (collocation nodes, 0=left endpoint): ", round.(taus; digits=4))
println("gate_syms: ", gate_syms)
println("mesh nodes bnds (count=$(length(bnds))): first/last = $(round(bnds[1];digits=4)) .. $(round(bnds[end];digits=4))")

si        = PEprob.model_info.simulation_info
has_preeq = si.has_pre_equilibration
sim_ids   = si.conditionids[:simulation]
preeq_ids = si.conditionids[:pre_equilibration]
p_nominal = PEtab.get_x(PEprob)
solver    = PEprob.probinfo.solver.solver
abstol    = PEprob.probinfo.solver.abstol
reltol    = PEprob.probinfo.solver.reltol
gate_raw  = [Symbolics.value(g) for g in gate_syms]
read_gates(integ) = Float64[Float64(integ.ps[g]) for g in gate_raw]

onnode(t) = any(b -> isapprox(b, t; atol=1e-6, rtol=1e-6), bnds)
nodeidx(t) = findfirst(b -> isapprox(b, t; atol=1e-6, rtol=1e-6), bnds)

for (cidx, cid) in enumerate(Symbol.(cids))
    cond_arg = cid
    if has_preeq
        pos = findfirst(==(cid), sim_ids); pos === nothing && continue
        cond_arg = preeq_ids[pos] => sim_ids[pos]
    end
    oprob, cbs = PEtab.get_odeproblem(p_nominal, PEprob; condition = cond_arg)
    tend  = max(maximum(bnds), oprob.tspan[2])
    oprob = ODE.remake(oprob; tspan = (oprob.tspan[1], tend))

    # sample PEtab's gate at every collocation time t_ij (+ nodes so events fire)
    tijs   = [(i, k, h_cum[i] + taus[k+1]*h[i]) for i in 1:N for k in 1:K]
    tstops = sort(unique(vcat([t for (_,_,t) in tijs], collect(bnds))))
    integ  = ODE.init(oprob, solver; callback=cbs, tstops=tstops, abstol=abstol, reltol=reltol)
    g0     = read_gates(integ)
    samp   = Dict{Float64,Vector{Float64}}()
    prev   = copy(g0); toggles = Tuple{Float64,Vector{Float64},Vector{Float64}}[]
    for ts in tstops
        while integ.t < ts - 1e-12
            tprev = integ.t; ODE.step!(integ)
            (isfinite(integ.t) && integ.t > tprev) || break
        end
        g = read_gates(integ)
        if g != prev; push!(toggles, (integ.t, copy(prev), copy(g))); prev = copy(g); end
        samp[ts] = g
    end

    println("\n--- cidx=$cidx ($cid) ---  g0(t=0)=$g0   gate_vals_ss=$(gate_vals_ss[:,cidx])")
    println("  event toggles ($(length(toggles))):")
    for (t, a, b) in toggles
        ni = nodeidx(t)
        println("    t=$(round(t;digits=5))  $a -> $b   on_node=$(onnode(t)) (node #$(ni===nothing ? "—" : ni)) ",
                ni === nothing ? "" : " => splits intervals $(ni) | $(ni+1)")
    end

    # ALIGNMENT: does gate_vals[:,i,cidx] == PEtab gate at every collocation point t_ij of interval i?
    nmis = 0; firstmis = nothing
    for (i, k, t) in tijs
        ours   = gate_vals[:, i, cidx]
        theirs = samp[t]
        if !isapprox(ours, theirs; atol=1e-9)
            nmis += 1
            firstmis === nothing && (firstmis = (i, k, round(t;digits=5), ours, theirs))
        end
    end
    println("  ALIGNMENT: $(N*K - nmis)/$(N*K) collocation points match; mismatches=$nmis",
            firstmis === nothing ? "  ✅ gate_vals correct at every collocation time" :
            "  ❌ first mismatch i=$(firstmis[1]) k=$(firstmis[2]) t=$(firstmis[3]) ours=$(firstmis[4]) petab=$(firstmis[5])")
end
println("\n== DONE ==")
