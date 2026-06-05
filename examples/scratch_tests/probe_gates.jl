# Probe: how piecewise(time) gate parameters are represented & how to read their values over time.
# Run: julia --project=. examples/scratch_tests/probe_gates.jl
using ExaModelsPEtab
import PEtab
import ModelingToolkitBase as MTK
import Symbolics
import OrdinaryDiffEq as ODE

const EP = ExaModelsPEtab

yaml = joinpath(@__DIR__, "..", "Benchmark-Models", "Oliveira_NatCommun2021", "Oliveira_NatCommun2021.yaml")
println("== loading PEtabModel/PEtabODEProblem ==")
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)

sys = PEprob.model_info.model.sys
println("\n== MTK parameters matching __parameter_ifelse ==")
gate_syms = [pp for pp in MTK.parameters(sys) if occursin("__parameter_ifelse", string(pp))]
for g in gate_syms; println("  ", g, "  ::", typeof(g)); end
println("  Ng = ", length(gate_syms))

println("\n== petab_events ==")
try
    pe = PEprob.model_info.model.petab_events
    println("  typeof = ", typeof(pe), "  length = ", length(pe))
catch e
    println("  ERR ", e)
end

println("\n== parametermap defaults for gates ==")
pm = Dict(PEprob.model_info.model.parametermap)
for g in gate_syms
    println("  ", g, " => ", get(pm, g, "MISSING"))
end

println("\n== get_odeproblem for the (only) condition, read gate at t=0 (post-init) ==")
cids = EP._get_cids(PEmodel)
println("  cids = ", cids)
pnom = PEtab.get_x(PEprob)
cond = Symbol(cids[1])
oprob, cbs = PEtab.get_odeproblem(pnom, PEprob; condition = cond)
println("  oprob.tspan = ", oprob.tspan, "  typeof(cbs) = ", typeof(cbs))

# try reading gate values off the problem via .ps indexing and via getp
for g in gate_syms
    v1 = try; oprob.ps[g]; catch e; "ps[] ERR $(e)"; end
    println("  oprob.ps[$g] = ", v1)
end
println("  has getp via ODE? ", isdefined(ODE, :getp))
println("  has SavingCallback via ODE? ", isdefined(ODE, :SavingCallback))
println("  has init via ODE? ", isdefined(ODE, :init), "  step! ? ", isdefined(ODE, :step!))

println("\n== step an integrator, read gate over time ==")
solver = PEprob.probinfo.solver.solver
tmax = maximum(filter(isfinite, Float64.(PEmodel.petab_tables[:measurements][!,:time])))
println("  max measured t = ", tmax)
op2 = ODE.remake(oprob; tspan = (oprob.tspan[1], tmax))
integ = ODE.init(op2, solver; callback = cbs, abstol = PEprob.probinfo.solver.abstol, reltol = PEprob.probinfo.solver.reltol)
println("  at t=", integ.t, " gates = ", [integ.ps[g] for g in gate_syms])
nsteps = 0
prev = [integ.ps[g] for g in gate_syms]
changes = Tuple{Float64,Vector{Float64}}[]
while integ.t < tmax && nsteps < 100000
    ODE.step!(integ)
    nsteps += 1
    cur = [integ.ps[g] for g in gate_syms]
    if cur != prev
        push!(changes, (integ.t, copy(cur)))
        prev = cur
    end
end
println("  nsteps = ", nsteps, "  final t = ", integ.t)
println("  gate CHANGES within window (t, vals): ", isempty(changes) ? "NONE (constant)" : changes)

println("\n== DONE ==")
