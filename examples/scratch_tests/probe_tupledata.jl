# Does ExaModels accept an NTuple-valued data field in the iterator, and splatting it into a
# build_function-style RHS? Mirrors passing per-interval gate values via the tuple (like h[i]).
using ExaModels
import Symbolics

# A "RHS" built by build_function taking a runtime number of extra (gate) args.
@Symbolics.variables zz gg1 gg2
fexpr = zz*zz + 3.0*gg1 + 5.0*gg2
f = Symbolics.build_function(fexpr, zz, gg1, gg2; expression = Val{false})

Ng = 2
c = ExaModels.ExaCore()
c, x = ExaModels.add_var(c, 3)

# iterator carries (i, h_i::Float64, gtup::NTuple{Ng,Float64}) — gtup computed with real i
gate_vals = Float64[ i==3 ? 0.0 : 1.0  for g in 1:Ng, i in 1:3]   # Ng x 3
itr = [(i, Float64(i), ntuple(g -> gate_vals[g,i], Ng)) for i in 1:3]

println("test A: splat gtup into f")
try
    c1, con = ExaModels.add_con(c, -hi*f(x[i], gv...) for (i, hi, gv) in itr)
    m = ExaModels.ExaModel(c1); println("  OK ncon=", m.meta.ncon)
catch e
    println("  FAIL: ", sprint(showerror, e)[1:min(end,200)])
end

println("test B: index gtup[1], gtup[2] explicitly")
try
    c2 = ExaModels.ExaCore(); c2, x2 = ExaModels.add_var(c2, 3)
    c2, con2 = ExaModels.add_con(c2, x2[i] + gv[1] + gv[2] for (i, hi, gv) in itr)
    m2 = ExaModels.ExaModel(c2); println("  OK ncon=", m2.meta.ncon)
catch e
    println("  FAIL: ", sprint(showerror, e)[1:min(end,200)])
end
println("== DONE ==")
