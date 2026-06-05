# Probe: what is the correct gate value for the pre-equilibration steady-state residual?
# For each Brannmark sim condition, compare standalone-preeq vs paired, and check ||f(u0)||.
using ExaModelsPEtab, PEtab, Symbolics
import OrdinaryDiffEq as ODE
import LinearAlgebra: norm
const EP = ExaModelsPEtab

yaml = joinpath(@__DIR__, "..", "Benchmark-Models", "Brannmark_JBC2010", "Brannmark_JBC2010.yaml")
PEmodel = PEtab.PEtabModel(yaml); PEprob = PEtab.PEtabODEProblem(PEmodel)

gate_syms = EP._get_gate_syms(PEprob)
graw = [Symbolics.value(g) for g in gate_syms]
println("gate_syms = ", gate_syms)

si        = PEprob.model_info.simulation_info
sim_ids   = si.conditionids[:simulation]
preeq_ids = si.conditionids[:pre_equilibration]
cids      = EP._get_cids(PEmodel)
pnom      = PEtab.get_x(PEprob)
solver    = PEprob.probinfo.solver.solver
readg(integ) = [Float64(integ.ps[g]) for g in graw]
fnorm(op) = (du = similar(op.u0); op.f(du, op.u0, op.p, op.tspan[1]); norm(du))

for (cidx, cid) in enumerate(cids)
    pos   = findfirst(==(Symbol(cid)), sim_ids)
    sscid = preeq_ids[pos]
    println("\ncid=$cid  pre-eq=$sscid")
    # standalone pre-eq condition
    try
        op, cb = PEtab.get_odeproblem(pnom, PEprob; condition = sscid)
        integ  = ODE.init(op, solver; callback = cb)
        println("  STANDALONE ok: gates@t0=", readg(integ), "  ||f(u0)||=", round(fnorm(op);sigdigits=3))
    catch e
        println("  STANDALONE threw: ", sprint(showerror, e)[1:min(end,120)])
    end
    # paired pre-eq => sim (what _get_zss_init uses); u0 is the pre-eq steady state
    op2, cb2 = PEtab.get_odeproblem(pnom, PEprob; condition = sscid => Symbol(cid))
    integ2   = ODE.init(op2, solver; callback = cb2)
    println("  PAIRED gates@t0=", readg(integ2), "  ||f(u0)||=", round(fnorm(op2);sigdigits=3),
            "  (u0 = pre-eq steady state)")
    cidx >= 2 && break   # two conditions is enough
end
println("\n== DONE ==")
