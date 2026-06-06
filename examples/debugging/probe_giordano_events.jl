# probe_giordano_events.jl — does the gate (event-parameter) actually toggle for Giordano pred1,
# and at what times? Steps the integrator with the CALLBACK'S OWN tstops (not the measurement
# mesh), so a fixed-time event fires whether or not it lands on a measurement node. If toggles
# appear here but _get_gate_vals (which uses measurement-node tstops) saw zero, the bug is that
# off-node events are stepped over and the gate freezes at its default.
using ExaModelsPEtab, PEtab
import Symbolics
import OrdinaryDiffEq as ODE
const EP = ExaModelsPEtab

model = get(ARGS, 1, "Giordano_Nature2020")
cidsym = Symbol(get(ARGS, 2, "pred1"))
yaml  = joinpath(@__DIR__, "..", "Benchmark-Models", model, model * ".yaml")
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)

gate_syms = EP._get_gate_syms(PEprob)
gate_raw  = [Symbolics.value(g) for g in gate_syms]
p_nom  = PEtab.get_x(PEprob)
solver = PEprob.probinfo.solver.solver
abstol = PEprob.probinfo.solver.abstol
reltol = PEprob.probinfo.solver.reltol

oprob, cbs = PEtab.get_odeproblem(p_nom, PEprob; condition = cidsym)
println("== probe events: $model / $cidsym ==  tspan=", oprob.tspan, "  Ng=", length(gate_syms))

# Inspect the callback set + try to surface any preset event times.
cc = hasproperty(cbs, :continuous_callbacks) ? cbs.continuous_callbacks : ()
dc = hasproperty(cbs, :discrete_callbacks)   ? cbs.discrete_callbacks   : ()
println("continuous callbacks: ", length(cc), "   discrete callbacks: ", length(dc))
for (i, d) in enumerate(dc)
    for f in propertynames(d)
        try
            v = getproperty(d, f)
            if v isa AbstractVector{<:Real} && !isempty(v)
                println("  disc cb #$i .$f preset times = ", round.(Float64.(v); digits=4))
            end
        catch; end
    end
end

readg(integ) = Float64[Float64(integ.ps[g]) for g in gate_raw]

# Step over [0,45] letting the integrator/callbacks choose stops (PresetTimeCallback registers its
# own tstops). Record every time the gate vector changes. (Wrapped in a function so the while-loop
# uses function scope, not the top-level soft scope.)
function scan_toggles(oprob, cbs, solver, abstol, reltol, gate_raw)
    readg(integ) = Float64[Float64(integ.ps[g]) for g in gate_raw]
    integ = ODE.init(oprob, solver; callback = cbs, abstol = abstol, reltol = reltol)
    g0 = readg(integ); cur = copy(g0)
    println("g0 (t=0): all==1? ", all(==(1.0), g0), "  sum=", sum(g0), " / ", length(g0))
    toggles = 0
    while integ.t < 45.0
        tp = integ.t; ODE.step!(integ)
        (isfinite(integ.t) && integ.t > tp) || break
        g = readg(integ)
        if g != cur
            ch = findall(g .!= cur); toggles += 1
            println("  TOGGLE @ t=", round(integ.t; digits=5), "  gates ", ch, "  -> ", g[ch])
            cur = g
        end
    end
    println("total toggles over [0,45] (callback-native tstops): ", toggles,
            "   final t=", round(integ.t; digits=3), "   final gate sum=", sum(cur))
end
scan_toggles(oprob, cbs, solver, abstol, reltol, gate_raw)
println("== DONE ==")
