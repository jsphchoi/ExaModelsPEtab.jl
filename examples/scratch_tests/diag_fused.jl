using ExaModelsPEtab, PEtab, ExaModels

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
get_yaml(m) = begin
    d = joinpath(MODELDIR, m)
    fs = filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))
    joinpath(d, first(fs))
end

# The GPU kernel receives con.f (a SIMDFunction) BY VALUE; its sizeof is exactly the
# per-kernel parameter footprint that overflows the sm_70 31.996 KiB limit. The footprint
# is dominated by the Node expression-tree type/leaves embedded in SIMDFunction.f.
function report(c)
    cons = reverse(collect(c.cons))  # c.cons is newest-first; reverse => build order
    println("idx | kind | nrows | o1 | o2 | comp1 | comp2 | sizeof | node")
    for (i, con) in enumerate(cons)
        f = con.f
        bytes = sizeof(f)
        nrows = length(con.itr)
        kind = con isa ExaModels.ConstraintAugmentation ? "AUG" : "CON"
        c1 = try length(f.comp1.inner) catch; -1 end
        c2 = try length(f.comp2.inner) catch; -1 end
        node = string(typeof(f.f)); node = node[1:min(end, 38)]
        flag = (c2 > 1000 || bytes > 20000) ? "   <== BIG" : ""
        println("$i | $kind | $nrows | $(f.o1step) | $(f.o2step) | $c1 | $c2 | $bytes | $node$flag")
    end
end

m = ARGS[1]
yaml = get_yaml(m)
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)
c = ExaModels.ExaCore(; concrete=Val(true))  # CPU backend

c, PEinfo = ExaModelsPEtab._create_variables(c, PEmodel, PEprob, 6)
c = ExaModelsPEtab._create_collocation(c, PEmodel, PEprob, PEinfo)
println("\n### after collocation ($(length(c.cons)) layers) ###")
report(c)
c = ExaModelsPEtab._create_continuity(c, PEmodel, PEprob, PEinfo)
c, y0, sigma0 = ExaModelsPEtab._create_objective(c, PEmodel, PEprob, PEinfo)
println("\n### FULL model ($(length(c.cons)) layers) ###")
report(c)
