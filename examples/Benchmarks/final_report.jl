# final_report.jl — reads Benchmarks/results/{Model}_results.txt and prints a
# formatted benchmark table comparing ExaModelsPEtab (MadNLP/GPU) vs PEtab.jl
# (Optim.IPNewton). Also writes the same report to Benchmarks/final_report.txt.
# Run from the repo root:
#   julia --project=. examples/Benchmarks/final_report.jl          # first-run times (default)
#   julia --project=. examples/Benchmarks/final_report.jl --sgm    # SGM (n=3) rerun times

using Printf

const RESULTDIR  = joinpath(@__DIR__, "results")
const REPORT_TXT = joinpath(@__DIR__, "final_report.txt")
const USE_SGM    = "--sgm" in ARGS

include(joinpath(@__DIR__, "list_benchmarks.jl"))  # BENCHMARK_MODELS / PETAB_SOLVED_MODELS / EXA_SUPPORTED_MODELS

# The report rows are the models PEtab actually solved — the meaningful head-to-head set.
# PEtab-side failures (Alkan/Chen/Froehlich/Lang/Raia/Smith/Weber) are excluded. The two
# ExaModels-unsupported models (Beer, Liu) stay in the rows so the solved-count accounting is honest.
const ALL_MODELS = PETAB_SOLVED_MODELS  # 28

# ─── helpers ──────────────────────────────────────────────────────────────────
function read_result(m)
    d = Dict{String,String}()
    p = joinpath(RESULTDIR, "$(m)_results.txt")
    isfile(p) || return d
    for line in eachline(p)
        i = findfirst('=', line); i === nothing && continue
        d[line[1:i-1]] = line[i+1:end]
    end
    d
end

g(d, k)   = get(d, k, "")
fparse(s) = tryparse(Float64, s)
short_name(m) = split(m, '_')[1]

function center_str(s, n)
    len = length(s); len >= n && return s[1:n]
    l = div(n - len, 2); " "^l * s * " "^(n - len - l)
end

function pct_exa_str(d)
    ct = fparse(g(d, "exa_compile_time")); pt = fparse(g(d, "exa_presolve_time"))
    (ct === nothing || pt === nothing || ct <= 0.0) && return "-"
    @sprintf("%4.0f%%", 100.0 * (ct - pt) / ct)
end

function madnlp_code(d)
    g(d, "exa_compile_status") in ("", "missing_yaml", "timeout", "error") && return "-"
    g(d, "exa_compile_status") != "ok"                                      && return "-"
    ss = g(d, "exa_solve_status")
    ss == "skipped" && return "-"
    ss == "error"   && return "E"
    ss == "timeout" && return "T"
    term = uppercase(g(d, "exa_term_status"))
    isempty(term)                          && return "-"
    occursin("SUCCEEDED",      term)       && return "0"
    occursin("WALLTIME",       term)       && return "1"
    occursin("RESTORATION",    term)       && return "2"
    occursin("INVALID_NUMBER", term)       && return "3"
    occursin("ITER",           term)       && return "4"
    return "5"
end

function petab_code(d)
    g(d, "petab_compile_status") in ("", "missing_yaml", "timeout", "error") && return "-"
    g(d, "petab_compile_status") != "ok"                                      && return "-"
    ss = g(d, "petab_solve_status")
    ss == "skipped" && return "-"
    ss == "error"   && return "E"
    ss == "timeout" && return "T"
    g(d, "petab_optimum_found") == "true"  && return "0"
    g(d, "petab_optimum_found") == "false" && return "1"
    return "-"
end

function rel_gap_str(d)
    eo = fparse(g(d, "exa_objective"))
    po = fparse(g(d, "petab_objective"))
    (eo === nothing || po === nothing ||
     !isfinite(eo) || !isfinite(po) || po == 0.0) && return "-"
    @sprintf("%+.2e", (eo - po) / abs(po) * 100.0)
end

fmt_cmp(d, pfx) = begin
    cs = g(d, pfx * "compile_status"); t = g(d, pfx * "compile_time")
    (cs == "ok" && !isempty(t)) ? t : "-"
end
fmt_slv(d, pfx) = begin
    # --sgm: show SGM solve time for exa (GPU-kernel-free, comparable to petab_solve_time)
    if USE_SGM && pfx == "exa_"
        s = g(d, "exa_sgm_status"); t = g(d, "exa_sgm_solve_time")
        return (s == "ok" && !isempty(t)) ? t : "-"
    end
    ss = g(d, pfx * "solve_status"); t = g(d, pfx * "solve_time")
    (ss == "ok" && !isempty(t)) ? t : "-"
end

# ─── column widths ────────────────────────────────────────────────────────────
const W_NAME  = 16
const W_CTIME = 10   # Compile(s)
const W_PCT   = 6    # EXA(%)
const W_STIME = 10   # Solve(s)
const W_STAT  = 6    # Status
const W_GAP   = 11   # rel.gap(%)

# inner widths of each major group (columns separated by single spaces)
const W_EXA_INNER   = W_CTIME + 1 + W_PCT + 1 + W_STIME + 1 + W_STAT   # 34
const W_PETAB_INNER = W_CTIME + 1 + W_STIME + 1 + W_STAT                # 27

# ─── build report string ──────────────────────────────────────────────────────
buf = IOBuffer()

timing_label = USE_SGM ? "ExaModels + MadNLPGPU  [solve=SGM n=3]" : "ExaModels + MadNLPGPU"
major_hdr = @sprintf("%-*s | %s | %s | %*s",
    W_NAME, "",
    center_str(timing_label,              W_EXA_INNER),
    center_str("PEtab + IPNewton",        W_PETAB_INNER),
    W_GAP, "")

sub_hdr = @sprintf("%-*s | %*s %*s %*s %*s | %*s %*s %*s | %*s",
    W_NAME,  "Model",
    W_CTIME, "Compile(s)", W_PCT, "EXA(%)", W_STIME, "Solve(s)", W_STAT, "Status",
    W_CTIME, "Compile(s)", W_STIME, "Solve(s)", W_STAT, "Status",
    W_GAP,   "rel.gap(%)")

bar = "="^length(sub_hdr)
sep = "-"^length(sub_hdr)

println(buf, bar)
println(buf, major_hdr)
println(buf, sub_hdr)
println(buf, sep)

for m in ALL_MODELS
    d = read_result(m)
    @printf(buf, "%-*s | %*s %*s %*s %*s | %*s %*s %*s | %*s\n",
        W_NAME, short_name(m),
        W_CTIME, fmt_cmp(d,"exa_"),   W_PCT, pct_exa_str(d),
        W_STIME, fmt_slv(d,"exa_"),   W_STAT, madnlp_code(d),
        W_CTIME, fmt_cmp(d,"petab_"), W_STIME, fmt_slv(d,"petab_"), W_STAT, petab_code(d),
        W_GAP, rel_gap_str(d))
end
println(buf, sep)

# ─── summary ──────────────────────────────────────────────────────────────────
exa_opt(d) = madnlp_code(d) == "0"
all_d           = [read_result(m) for m in ALL_MODELS]                 # 28 PEtab-solved (rows)
exa_supported_d = [read_result(m) for m in EXA_SUPPORTED_MODELS]       # 26 exa-representable

n_bench   = length(BENCHMARK_MODELS)         # 35
n_petab   = length(PETAB_SOLVED_MODELS)      # 28
n_exa_sup = length(EXA_SUPPORTED_MODELS)     # 26
n_exa_opt = count(exa_opt, exa_supported_d)  # exa SOLVE_SUCCEEDED among the supported set

println(buf, "\nSUMMARY")
@printf(buf, "  Benchmark models       : %2d\n",                          n_bench)
@printf(buf, "  PEtab found optimum    : %2d / %2d\n",                     n_petab, n_bench)
@printf(buf, "  ExaModels-supported    : %2d  (PEtab-solved minus Beer, Liu)\n", n_exa_sup)
@printf(buf, "  ExaModels solved (opt) : %2d / %2d  (of ExaModels-supported)\n", n_exa_opt, n_exa_sup)

get_gap(d) = begin
    eo = fparse(g(d, "exa_objective")); po = fparse(g(d, "petab_objective"))
    (eo === nothing || po === nothing || !isfinite(eo) || !isfinite(po) || po == 0.0) ? nothing :
    (eo - po) / abs(po) * 100.0
end
gaps = filter(!isnothing, get_gap.(exa_supported_d))
if !isempty(gaps)
    max_idx = argmax(abs.(gaps))
    @printf(buf, "\nRELATIVE OBJECTIVE GAP  (exa_obj - petab_obj) / |petab_obj| × 100%%  [negative = ExaModels lower]:\n")
    @printf(buf, "  n=%d  median=%+.2e%%  max_abs=%+.2e%%  (|gap|<1%%: %d)\n",
            length(gaps), sort(gaps)[cld(length(gaps), 2)],
            gaps[max_idx], count(x -> abs(x) < 1.0, gaps))
end

println(buf, bar)
println(buf, "")
println(buf, "STATUS KEY")
println(buf, "  MadNLP (Status): 0=SOLVE_SUCCEEDED  1=WALLTIME_EXCEEDED  2=RESTORATION_FAILED")
println(buf, "                   3=INVALID_NUMBER_JACOBIAN  4=MAX_ITER_EXCEEDED  5=other")
println(buf, "                   E=exception  T=process_timeout  -=compile_failed/not_run")
println(buf, "  PEtab  (Status): 0=converged  1=ran_not_converged  E=error  T=timeout  -=compile_failed/not_run")
println(buf, "  EXA(%)         : fraction of compile time spent on ExaModels build")
println(buf, "                   (remainder = PEtab setup + ODE presolve at nominal θ)")
println(buf, "                   '-' for results from earlier benchmarks (presolve not tracked then)")
println(buf, "  rel.gap(%)     : (ExaModels_obj - PEtab_obj) / |PEtab_obj| × 100%  (negative = ExaModels lower)")
USE_SGM && println(buf, "  Timing mode    : SGM geometric mean over $(g(Dict("exa_sgm_n"=>"3"), "exa_sgm_n")) reruns (exa_sgm_* keys)")

# ─── output ───────────────────────────────────────────────────────────────────
report = String(take!(buf))
print(report)
open(REPORT_TXT, "w") do io; print(io, report); end
println("Report saved to: $REPORT_TXT")
