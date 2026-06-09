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

# The report rows are the CANONICAL exa-supported set — every model ExaModelsPEtab's collocation
# transcription can represent (EXA_SUPPORTED_MODELS in list_benchmarks.jl, 25), in ALPHABETICAL
# order. Models without a (recent) result row simply show blank/failed status, so the report is a
# complete, stable scoreboard of the supported set rather than just the rerun-target subset.
const ALL_MODELS = sort(EXA_SUPPORTED_MODELS)  # 25 (canonical exa-supported, alphabetical)

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

# A SOLVE_SUCCEEDED whose objective is meaningfully WORSE than PEtab's (positive rel.gap above
# this %) is a converged-but-suboptimal local minimum (e.g. Bertozzi +194%, Laske +3.6%,
# Okuonghae +1.9%), NOT a clean optimum. It gets the "0S" marker and is EXCLUDED from the
# "ExaModels solved" count (only true status 0 counts). Negative gaps (ExaModels lower than
# PEtab) are the collocation relaxation "ghost", not suboptimality, so they stay "0".
const SUBOPT_GAP_PCT = 1.5

# rel.gap (%) = (exa_obj - petab_obj)/|petab_obj| * 100; positive = ExaModels objective WORSE
# (higher, for an NLL minimization) than PEtab. nothing when either objective is missing/non-finite.
function rel_gap_val(d)
    eo = fparse(g(d, "exa_objective")); po = fparse(g(d, "petab_objective"))
    (eo === nothing || po === nothing || !isfinite(eo) || !isfinite(po) || po == 0.0) && return nothing
    (eo - po) / abs(po) * 100.0
end

# The SGM rerun count actually used (read from the data; falls back to "?" if no SGM ran).
# Used in the header label so it reflects the real n instead of a hardcoded value.
function sgm_n_used(all_d)
    for d in all_d
        n = g(d, "exa_sgm_n"); !isempty(n) && return n
        p = g(d, "petab_sgm_n"); !isempty(p) && return p
    end
    return "?"
end

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
    occursin("ACCEPTABLE",     term)       && return "A"   # SOLVED_TO_ACCEPTABLE_LEVEL (ε-optimal; valid solve)
    if occursin("SUCCEEDED", term)                          # SOLVE_SUCCEEDED: split clean vs suboptimal by gap
        gp = rel_gap_val(d)
        return (gp !== nothing && gp >= SUBOPT_GAP_PCT) ? "0S" : "0"
    end
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
    gp = rel_gap_val(d)
    gp === nothing ? "-" : @sprintf("%+.2e", gp)
end

fmt_cmp(d, pfx) = begin
    cs = g(d, pfx * "compile_status"); t = g(d, pfx * "compile_time")
    (cs == "ok" && !isempty(t)) ? t : "-"
end
fmt_slv(d, pfx) = begin
    # --sgm: show the SGM geometric-mean solve time for BOTH backends (exa: GPU-kernel-free reruns;
    # petab: n-rerun calibrate mean) — an n-vs-n head-to-head, comparable across the two columns.
    if USE_SGM
        s = g(d, pfx * "sgm_status"); t = g(d, pfx * "sgm_solve_time")
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

all_d           = [read_result(m) for m in ALL_MODELS]              # rerun target rows
exa_supported_d = [read_result(m) for m in EXA_SUPPORTED_MODELS]    # 25 exa-representable (context only)
const SGM_N     = sgm_n_used(all_d)

# Labels are kept within the column inner widths (34 / 27) so center_str never truncates them
# mid-string (the old "[solve=SGM n=3]" label overflowed 34 chars → a dangling unclosed '[').
exa_label   = USE_SGM ? "ExaModels + MadNLPGPU (SGM n=$SGM_N)" : "ExaModels + MadNLPGPU"
petab_label = USE_SGM ? "PEtab + IPNewton (SGM n=$SGM_N)"      : "PEtab + IPNewton"
major_hdr = @sprintf("%-*s | %s | %s | %*s",
    W_NAME, "",
    center_str(exa_label,   W_EXA_INNER),
    center_str(petab_label, W_PETAB_INNER),
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
# Scoped to the canonical exa-supported set (ALL_MODELS = sort(EXA_SUPPORTED_MODELS)), not all 35.
exa_opt(d)    = madnlp_code(d) == "0"    # TRUE status 0 only — clean optimum matching PEtab
exa_subopt(d) = madnlp_code(d) == "0S"   # SUCCEEDED but suboptimal (positive gap >= SUBOPT_GAP_PCT)

n_target     = length(ALL_MODELS)            # 25 exa-supported (rows)
n_exa_opt    = count(exa_opt, all_d)         # clean status-0 solves among the targets
n_exa_subopt = count(exa_subopt, all_d)      # converged-but-suboptimal (0S)

println(buf, "\nSUMMARY  (exa-supported set)")
@printf(buf, "  Exa-supported models   : %2d  (of %d benchmark / %d PEtab-solved / %d exa-supported)\n",
        n_target, length(BENCHMARK_MODELS), length(PETAB_SOLVED_MODELS), length(EXA_SUPPORTED_MODELS))
@printf(buf, "  ExaModels solved       : %2d / %2d  (TRUE status 0 only — clean optimum matching PEtab)\n", n_exa_opt, n_target)
@printf(buf, "  Succeeded-suboptimal   : %2d       (0S — SOLVE_SUCCEEDED but obj >= +%.1f%% vs PEtab; excluded above)\n", n_exa_subopt, SUBOPT_GAP_PCT)

get_gap(d) = begin
    eo = fparse(g(d, "exa_objective")); po = fparse(g(d, "petab_objective"))
    (eo === nothing || po === nothing || !isfinite(eo) || !isfinite(po) || po == 0.0) ? nothing :
    (eo - po) / abs(po) * 100.0
end
gaps = filter(!isnothing, get_gap.(all_d))
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
println(buf, "  MadNLP (Status): 0=SOLVE_SUCCEEDED (clean optimum, matches PEtab)  0S=SUCCEEDED but suboptimal (obj >= +$(SUBOPT_GAP_PCT)% vs PEtab)")
println(buf, "                   A=SOLVED_TO_ACCEPTABLE_LEVEL (ε-optimal)  1=WALLTIME_EXCEEDED  2=RESTORATION_FAILED")
println(buf, "                   3=INVALID_NUMBER_JACOBIAN  4=MAX_ITER_EXCEEDED  5=other")
println(buf, "                   E=exception  T=process_timeout  -=compile_failed/not_run")
println(buf, "  PEtab  (Status): 0=converged  1=ran_not_converged  E=error  T=timeout  -=compile_failed/not_run")
println(buf, "  EXA(%)         : fraction of compile time spent on ExaModels build")
println(buf, "                   (remainder = PEtab setup + ODE presolve at nominal θ)")
println(buf, "                   '-' for results from earlier benchmarks (presolve not tracked then)")
println(buf, "  rel.gap(%)     : (ExaModels_obj - PEtab_obj) / |PEtab_obj| × 100%  (negative = ExaModels lower)")
USE_SGM && println(buf, "  Timing mode    : both Solve(s) columns = SGM geometric mean over n=$SGM_N reruns (exa_sgm_* / petab_sgm_* keys)")

# ─── output ───────────────────────────────────────────────────────────────────
report = String(take!(buf))
print(report)
open(REPORT_TXT, "w") do io; print(io, report); end
println("Report saved to: $REPORT_TXT")
