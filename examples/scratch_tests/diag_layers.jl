using ExaModelsPEtab, PEtab, ExaModels

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
get_yaml(m) = (d = joinpath(MODELDIR, m); joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d)))))

# Print every NEW constraint layer since `from`, with the real GPU-kernel-param drivers:
# comp1/comp2 = gradient/Hessian sparsity-compressor lengths (these tuples dominate the
# SIMDFunction sizeof that overflows the sm_70 kernel parameter space). Dump the expression
# string for big layers so we can SEE what is bloated. Flushes immediately.
function report(c, from)
    cons = reverse(collect(c.cons))
    for i in (from+1):length(cons)
        con = cons[i]; f = con.f
        kind = con isa ExaModels.ConstraintAugmentation ? "AUG" : "CON"
        c1 = try length(f.comp1.inner) catch; -1 end
        c2 = try length(f.comp2.inner) catch; -1 end
        big = sizeof(f) > 20000 || c2 > 1000
        println("L$i $kind nrows=$(length(con.itr)) o1=$(f.o1step) o2=$(f.o2step) comp1=$c1 comp2=$c2 sizeof=$(sizeof(f))$(big ? "  <==BIG" : "")")
        if big
            s = try ExaModels._expr_string(f.f) catch e; "(expr err: $e)" end
            println("      expr: ", s[1:min(end, 240)])
        end
        flush(stdout)
    end
    return length(cons)
end

m = ARGS[1]; K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
PEmodel = PEtab.PEtabModel(get_yaml(m)); PEprob = PEtab.PEtabODEProblem(PEmodel)
c = ExaModels.ExaCore(; concrete=Val(true))
c, PEinfo = ExaModelsPEtab._create_variables(c, PEmodel, PEprob, K)
println("### collocation ###"); flush(stdout)
c = ExaModelsPEtab._create_collocation(c, PEmodel, PEprob, PEinfo); n1 = report(c, 0)
println("### continuity ###"); flush(stdout)
c = ExaModelsPEtab._create_continuity(c, PEmodel, PEprob, PEinfo); n2 = report(c, n1)
println("### objective ###"); flush(stdout)
c, y0, s0 = ExaModelsPEtab._create_objective(c, PEmodel, PEprob, PEinfo); report(c, n2)
println("DONE nvar=$(c.nvar) ncon=$(c.ncon)")
