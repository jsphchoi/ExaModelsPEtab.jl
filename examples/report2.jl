# report2.jl — final report for the Ipopt-filtered benchmark.
#
# 1. PEtab/Ipopt compile status+time, solve status+time
# 2. ExaModelsPEtab compile status+time, solve status+time
# 3. relative objective gap to the PEtab solution
# plus: proportion of PEtab optima found and ExaModels optima found, overall and
#       restricted to models with NO event callbacks (unsupported by the collocation).

using Printf
const RD = joinpath(@__DIR__, "results2")

read_result(p) = begin
    d = Dict{String,String}()
    isfile(p) && for line in eachline(p); i=findfirst('=',line); i===nothing||(d[line[1:i-1]]=line[i+1:end]); end
    d
end
pet(m) = read_result(joinpath(RD, "$(m).petab.txt"))
exa(m) = read_result(joinpath(RD, "$(m).exa.txt"))
g(d,k) = get(d,k,"")
fnum(s) = (v=tryparse(Float64,s); v)

models = sort([replace(f,".petab.txt"=>"") for f in readdir(RD) if endswith(f,".petab.txt")])

petab_ran(m)   = g(pet(m),"solve_status")=="ok"
petab_opt(m)   = g(pet(m),"optimum_found")=="true"
exa_opt(m)     = occursin("SUCCEEDED", uppercase(g(exa(m),"term_status")))
no_events(m)   = g(pet(m),"has_events")=="false"
function relgap(m)
    po = fnum(g(pet(m),"objective")); eo = fnum(g(exa(m),"objective"))
    (po===nothing || eo===nothing || !isfinite(po) || !isfinite(eo) || po==0) && return nothing
    abs(eo-po)/abs(po)
end

println("="^150)
println("IPOPT-FILTERED BENCHMARK  —  PEtab.jl (Ipopt/LBFGS, 1h cap)  vs  ExaModelsPEtab (MadNLP/CUDA, K=5, 30-min compile, solve=2x petab or 1h)")
println("="^150)
@printf("%-34s|%-3s| %-9s %-7s %-9s %-7s %-7s | %-9s %-7s %-9s %-7s %-20s | %-12s %-12s %-9s\n",
  "Model","ev","PEcmp","t","PEslv","t","optfnd","EXcmp","t","EXslv","t","term","PEtab obj","Exa obj","relgap")
println("-"^150)
for m in models
    p=pet(m); e=exa(m); rg=relgap(m)
    @printf("%-34s|%-3s| %-9s %-7s %-9s %-7s %-7s | %-9s %-7s %-9s %-7s %-20s | %-12s %-12s %-9s\n",
      m, g(p,"has_events")=="true" ? "Y" : "n",
      g(p,"compile_status"), g(p,"compile_time"), g(p,"solve_status"), g(p,"solve_time"),
      petab_opt(m) ? "yes" : "no",
      g(e,"compile_status"), g(e,"compile_time"), g(e,"solve_status"), g(e,"solve_time"), g(e,"term_status"),
      g(p,"objective")=="" ? "" : string(round(something(fnum(g(p,"objective")),NaN);digits=4)),
      g(e,"objective")=="" ? "" : string(round(something(fnum(g(e,"objective")),NaN);digits=4)),
      rg===nothing ? "" : @sprintf("%.2e", rg))
end
println("-"^150)

qual    = filter(petab_ran, models)               # comparison universe: PEtab ran
qual_ne = filter(m->petab_ran(m)&&no_events(m), models)
frac(sel,set) = isempty(set) ? "n/a" : @sprintf("%d/%d (%.0f%%)", count(sel,set), length(set), 100*count(sel,set)/length(set))

println("\nSUMMARY  (denominator = models where PEtab/Ipopt ran)")
println("  models total                    : ", length(models))
println("  PEtab ran (qualified for Exa)   : ", length(qual))
println("  of which have NO event callbacks: ", length(qual_ne))
println()
println("  PEtab optimum found (all qual)  : ", frac(petab_opt, qual))
println("  Exa  optimum found  (all qual)  : ", frac(exa_opt,   qual))
println("  PEtab optimum found (no-events) : ", frac(petab_opt, qual_ne))
println("  Exa  optimum found  (no-events) : ", frac(exa_opt,   qual_ne))

gaps = filter(!isnothing, [relgap(m) for m in qual])
gaps_ne = filter(!isnothing, [relgap(m) for m in qual_ne])
showgaps(label, gs) = isempty(gs) ? println("  $label: (none)") :
    @printf("  %s: n=%d  median=%.2e  max=%.2e  (<1%%: %d)\n", label, length(gs),
            sort(gs)[cld(length(gs),2)], maximum(gs), count(<(0.01), gs))
println("\nRELATIVE OBJECTIVE GAP to PEtab (where both objectives available):")
showgaps("all qualified", gaps)
showgaps("no-events    ", gaps_ne)
println("="^150)
