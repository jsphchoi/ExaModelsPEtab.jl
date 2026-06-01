# report.jl — combined final report.
# Reads examples/results/<model>.exa.txt (ExaModels/MadNLP) and <model>.petab.txt
# (PEtab.jl/Optim-IPNewton) and prints, for every model:
#   1. ExaModels compile status + time
#   2. MadNLP solve status + time (+ termination status, objective)
#   3. PEtab.jl compile + solve time (+ objective) — only where 1 & 2 succeeded.

using Printf

const RESULTDIR = joinpath(@__DIR__, "results")

read_result(path) = begin
    d = Dict{String,String}()
    if isfile(path)
        for line in eachline(path)
            i = findfirst('=', line); i === nothing && continue
            d[line[1:i-1]] = line[i+1:end]
        end
    end
    d
end
exa(m)   = read_result(joinpath(RESULTDIR, "$(m).exa.txt"))
petab(m) = read_result(joinpath(RESULTDIR, "$(m).petab.txt"))
g(d, k)  = get(d, k, "")

const MODELS = sort([replace(f, ".exa.txt" => "")
                     for f in readdir(RESULTDIR) if endswith(f, ".exa.txt")])

solved(m) = g(exa(m), "solve_status") == "ok" &&
            occursin("SUCCEEDED", uppercase(g(exa(m), "term_status")))

println("="^132)
println("FINAL BENCHMARK REPORT   —   ExaModelsPEtab (MadNLP/CUDA, K=5, tol=1e-6, 10-min limits)  vs  PEtab.jl (Optim.IPNewton)")
println("="^132)
@printf("%-34s | %-10s %-8s | %-12s %-8s %-21s | %-8s %-8s %-8s | %s\n",
        "Model", "EXA-cmp", "t(s)", "MadNLP-slv", "t(s)", "term-status",
        "PEtcmp", "PEtslv", "t(s)", "objectives  exa | petab")
println("-"^132)

n_solved = n_restfail = n_ctimeout = n_cerror = n_stimeout = n_serror = 0
for m in MODELS
    e = exa(m); p = petab(m)
    cs, ct = g(e, "compile_status"), g(e, "compile_time")
    ss, st = g(e, "solve_status"),   g(e, "solve_time")
    term   = g(e, "term_status")
    eobj   = g(e, "objective")

    cs == "timeout" && (global n_ctimeout += 1)
    cs == "error"   && (global n_cerror += 1)
    if cs == "ok"
        ss == "timeout" && (global n_stimeout += 1)
        ss == "error"   && (global n_serror += 1)
    end
    occursin("SUCCEEDED", uppercase(term)) && (global n_solved += 1)
    occursin("RESTORATION", uppercase(term)) && (global n_restfail += 1)

    if solved(m)
        pc, ps2 = g(p, "compile_status"), g(p, "solve_status")
        @printf("%-34s | %-10s %-8s | %-12s %-8s %-21s | %-8s %-8s %-8s | %s | %s\n",
                m, cs, ct, ss, st, term,
                pc, ps2, g(p, "solve_time"), eobj, g(p, "objective"))
    else
        @printf("%-34s | %-10s %-8s | %-12s %-8s %-21s | %-8s %-8s %-8s | %s\n",
                m, cs, ct, ss, st, term, "-", "-", "-", eobj == "" ? "" : eobj)
    end
end
println("-"^132)
@printf("Totals: %d models | MadNLP optimal=%d  restoration_failed=%d  compile_timeout=%d  compile_error=%d  solve_timeout=%d  solve_error=%d\n",
        length(MODELS), n_solved, n_restfail, n_ctimeout, n_cerror, n_stimeout, n_serror)
println("="^132)

# Detailed comparison for the MadNLP-solved set (the PEtab.jl comparison subset)
ss_models = filter(solved, MODELS)
if !isempty(ss_models)
    println("\nPEtab.jl comparison (models MadNLP solved to optimality):")
    @printf("%-34s | %10s %10s | %10s %10s | %-22s %-22s\n",
            "Model", "exaCmp(s)", "petCmp(s)", "exaSlv(s)", "petSlv(s)", "exa objective", "petab objective")
    println("-"^124)
    for m in ss_models
        e = exa(m); p = petab(m)
        @printf("%-34s | %10s %10s | %10s %10s | %-22s %-22s\n",
                m, g(e,"compile_time"), g(p,"compile_time"),
                g(e,"solve_time"), g(p,"solve_time"),
                g(e,"objective"), g(p,"objective"))
    end
    println("-"^124)
    println("Note: Bruno_JExpBot2016 is the JIT-warmup model; its ExaModels compile/solve")
    println("      times are artificially low (cached double-build). Representative warm")
    println("      figures from isolated runs: compile ~4 s, solve ~20 s.")
end
