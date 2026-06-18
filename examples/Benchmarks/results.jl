# results.jl — reads Benchmarks/results/{Model}_results.txt and prints a formatted benchmark table
# comparing ExaModelsPEtab + MadNLP on GPU (exagpu_*) and CPU (exacpu_*) against PEtab.jl +
# Optim.IPNewton (petab_*). Also writes the report to Benchmarks/results.txt.
#   julia --project=examples examples/Benchmarks/results.jl          # warm (SGM) solve times — DEFAULT
#   julia --project=examples examples/Benchmarks/results.jl --cold   # cold first-run solve times instead

using Printf

const RESULTDIR  = joinpath(@__DIR__, "results")
const REPORT_TXT = joinpath(@__DIR__, "results.txt")
const USE_SGM    = !("--cold" in ARGS)

include(joinpath(@__DIR__, "options.jl"))

const ALL_MODELS = sort(EXA_SUPPORTED_MODELS)  # 20 (reported exa target set: continuous + PEtab-solved)
const SGM_N      = BENCH_SGM_N
const SGM_SHIFT  = BENCH_SGM_SHIFT

# A SOLVE_SUCCEEDED/ACCEPTABLE whose objective is worse than PEtab's by ≥ this % is converged-but-
# suboptimal (0S / 0AS) and is EXCLUDED from the "ExaModels solved" count.
const SUBOPT_GAP_PCT = 2.0

# ─── helpers ──────────────────────────────────────────────────────────────────
function read_result(m)
    d = Dict{String,String}(); p = joinpath(RESULTDIR, "$(m)_results.txt")
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

# GAP(%) = (petab_obj(exa_p*) - petab_obj) / |petab_obj| × 100 — ExaModels' optimal parameters scored
# under PEtab's OWN objective, vs PEtab's optimum (fair, same-objective). The benchmark stores
# petab_obj(exa_p*) as `<pfx>petab_obj`; until a run populates it we fall back to the ExaModels
# objective `<pfx>objective` so the column is not empty.
function gap_val(d, pfx)
    eo = fparse(g(d, pfx * "petab_obj")); eo === nothing && (eo = fparse(g(d, pfx * "objective")))
    po = fparse(g(d, "petab_objective"))
    (eo === nothing || po === nothing || !isfinite(eo) || !isfinite(po) || po == 0.0) && return nothing
    (eo - po) / abs(po) * 100.0
end
# +X.Xe+X  (one-decimal mantissa, exponent without a leading zero)
gap_str(d, pfx) = (gp = gap_val(d, pfx); gp === nothing ? "-" :
                   replace(@sprintf("%+.1e", gp), "e+0" => "e+", "e-0" => "e-"))

# ExaModels MadNLP status code for backend `pfx` (exagpu_ / exacpu_).
function madnlp_code(d, pfx)
    g(d, pfx * "compile_status") != "ok" && return "-"
    ss = g(d, pfx * "solve_status")
    ss == "skipped" && return "-"
    ss == "timeout" && return "T"   # compiled+ran but hard-killed at the deadline (overshot max_wall_time) ⇒ timeout
    ss == "error" && return "E"
    term = uppercase(g(d, pfx * "term_status"))
    isempty(term) && return "-"
    if occursin("SUCCEEDED", term) || occursin("ACCEPTABLE", term)
        base = occursin("ACCEPTABLE", term) ? "0A" : "0"
        gp   = gap_val(d, pfx)
        return (gp !== nothing && gp >= SUBOPT_GAP_PCT) ? base * "S" : base
    end
    occursin("WALLTIME",         term) && return "T"
    occursin("RESTORATION",      term) && return "R"
    occursin("SEARCH_DIRECTION", term) && return "D"
    return "5"
end

function petab_code(d)
    g(d, "petab_compile_status") != "ok" && return "-"
    ss = g(d, "petab_solve_status")
    ss in ("skipped", "timeout") && return "-"
    ss == "error" && return "E"
    if g(d, "petab_optimum_found") == "true"
        # Optim has no single status code — report WHICH convergence criterion fired. Precedence:
        # gradient (strongest; genuine first-order stationary) → F_RELTOL plateau → X_ABSTOL zero-step.
        # Anything but "0" means Optim declared success with |g| > g_tol (NOT certified stationary).
        g(d, "petab_gconverged") == "true" && return "0"   # gradient criterion: |g| ≤ g_tol
        g(d, "petab_fconverged") == "true" && return "0F"  # F_RELTOL: objective-change plateau
        g(d, "petab_xconverged") == "true" && return "0X"  # X_ABSTOL: step-size collapse / zero step
        return "0"  # unreachable: converged ⟺ (x||f||g), all three captured above; default if flag-read failed
    end
    if g(d, "petab_optimum_found") == "false"
        # distinguish a wall-time timeout (Optim hit time_limit, ≈ MadNLP "T") from generic
        # non-convergence: solve_time reaching the wall is a solid timeout signal.
        st = fparse(g(d, "petab_solve_time"))
        (st !== nothing && st >= 0.99 * BENCH_SOLVE_LIMIT) && return "T"
        return "1"
    end
    return "-"
end

pct_exa_str(d, pfx) = begin
    ct = fparse(g(d, pfx * "compile_time")); pt = fparse(g(d, pfx * "presolve_time"))
    (ct === nothing || pt === nothing || ct <= 0.0) ? "-" : @sprintf("%4.0f%%", 100.0 * (ct - pt) / ct)
end
fmt_cmp(d, pfx) = (g(d, pfx * "compile_status") == "ok" && !isempty(g(d, pfx * "compile_time"))) ? g(d, pfx * "compile_time") : "-"
fmt_slv(d, pfx) = begin
    if USE_SGM && g(d, pfx * "sgm_status") == "ok"
        raw = g(d, pfx * "solve_times")
        ts  = isempty(raw) ? Float64[] : Float64[x for x in (tryparse(Float64, s) for s in split(raw, ",")) if x !== nothing]
        if !isempty(ts)
            string(shifted_geomean(ts, SGM_SHIFT))   # report-time shift from the raw per-rerun times
        elseif !isempty(g(d, pfx * "sgm_solve_time"))
            g(d, pfx * "sgm_solve_time")             # legacy file (no raw times): pre-refactor stored aggregate
        else
            "-"
        end
    elseif !USE_SGM && g(d, pfx * "solve_status") == "ok" && !isempty(g(d, pfx * "solve_time"))
        g(d, pfx * "solve_time")
    elseif occursin("WALLTIME", uppercase(g(d, pfx * "term_status"))) && !isempty(g(d, pfx * "solve_time"))
        # timeout: no rerun times recorded, so show the elapsed solve time (≈ max walltime) instead of "-"
        g(d, pfx * "solve_time")
    elseif !isempty(g(d, pfx * "solve_time")) &&
           (st = fparse(g(d, pfx * "solve_time"))) !== nothing && st >= 0.99 * BENCH_SOLVE_LIMIT
        # timeout with no term_status / SGM (e.g. PEtab non-converged at the wall): show elapsed time,
        # matching the exa WALLTIME behavior above instead of "-"
        g(d, pfx * "solve_time")
    else
        "-"
    end
end

# ─── column widths ────────────────────────────────────────────────────────────
const W_NAME = 14   # fits "SalazarCavazos"
const W_CMPL, W_PCT, W_SOL, W_STAT, W_GAP = 7, 6, 7, 4, 8
const W_EXA_INNER   = W_CMPL + 1 + W_PCT + 1 + W_SOL + 2 + W_STAT + 1 + W_GAP   # GPU & CPU groups (incl GAP; +1 space before STAT)
const W_PETAB_INNER = W_CMPL + 1 + W_SOL + 2 + W_STAT                            # PEtab (no GAP; +1 space before STAT)

# ─── build report ───────────────────────────────────────────────────────────────
buf   = IOBuffer()
all_d = [read_result(m) for m in ALL_MODELS]

major_hdr = @sprintf("%-*s | %s | %s | %s",
    W_NAME, "",
    center_str("ExaModels + MadNLP (GPU)", W_EXA_INNER),
    center_str("ExaModels + MadNLP (CPU)", W_EXA_INNER),
    center_str("PEtab + IPNewton",         W_PETAB_INNER))
sub_hdr = @sprintf("%-*s | %*s %*s %*s  %*s %*s | %*s %*s %*s  %*s %*s | %*s %*s  %*s",
    W_NAME, "Model",
    W_CMPL,"CMPL(s)", W_PCT,"EXA(%)", W_SOL,"SOL(s)", W_STAT,"STAT", W_GAP,"GAP(%)",
    W_CMPL,"CMPL(s)", W_PCT,"EXA(%)", W_SOL,"SOL(s)", W_STAT,"STAT", W_GAP,"GAP(%)",
    W_CMPL,"CMPL(s)", W_SOL,"SOL(s)", W_STAT,"STAT")
bar = "="^length(sub_hdr); sep = "-"^length(sub_hdr)

println(buf, bar); println(buf, major_hdr); println(buf, sub_hdr); println(buf, sep)
for m in ALL_MODELS
    d = read_result(m)
    @printf(buf, "%-*s | %*s %*s %*s  %*s %*s | %*s %*s %*s  %*s %*s | %*s %*s  %*s\n",
        W_NAME, short_name(m),
        W_CMPL,fmt_cmp(d,"exagpu_"), W_PCT,pct_exa_str(d,"exagpu_"), W_SOL,fmt_slv(d,"exagpu_"), W_STAT,madnlp_code(d,"exagpu_"), W_GAP,gap_str(d,"exagpu_"),
        W_CMPL,fmt_cmp(d,"exacpu_"), W_PCT,pct_exa_str(d,"exacpu_"), W_SOL,fmt_slv(d,"exacpu_"), W_STAT,madnlp_code(d,"exacpu_"), W_GAP,gap_str(d,"exacpu_"),
        W_CMPL,fmt_cmp(d,"petab_"),  W_SOL,fmt_slv(d,"petab_"),      W_STAT,petab_code(d))
end
println(buf, sep)

# ─── summary (GPU is the primary ExaModels backend) ─────────────────────────────
exa_opt(d)    = madnlp_code(d,"exagpu_") in ("0", "0A")
exa_subopt(d) = madnlp_code(d,"exagpu_") in ("0S", "0AS")

println(buf, "\nSUMMARY (ExaModelsPEtab target set: continuous + PEtab-solved)")
@printf(buf, "  Target models          : %2d       (of %d; %d 'Possible Discontinuities', %d PEtab.jl failed compile)\n",
        length(EXA_SUPPORTED_MODELS), length(BENCHMARK_MODELS), length(_POSSIBLE_DISCONTINUITIES), length(_PETAB_UNSOLVED))
@printf(buf, "  ExaModels solved (GPU) : %2d / %2d  (status 0 + 0A, full/acceptable optimum)\n", count(exa_opt, all_d), length(ALL_MODELS))
@printf(buf, "  Solved-but-suboptimal  : %2d       (0S / 0AS, converged but obj ≥+%.1f%% vs PEtab; excluded above)\n", count(exa_subopt, all_d), SUBOPT_GAP_PCT)

println(buf, "")
println(buf, "  CMPL(s) := Model compilation time")
println(buf, "  EXA(%)  := Fraction of model compile time spent on actual ExaModels build (PEtab setup + mesh generation)")
println(buf, "  SOL(s)  := Solver solve time, shifted geometric mean (by δ = $(SGM_SHIFT)s) over n=$SGM_N reruns")
println(buf, "  STAT    := Solver status")
println(buf, "  GAP(%)  := (petab.nllh(exa_p*) - petab_obj) / |petab_obj| × 100%  (negative => ExaModels lower)")

# ─── status key (only codes present in the table) ───────────────────────────────
madnlp_desc = Dict("0"=>"SOLVE_SUCCEEDED", "0A"=>"SOLVED_TO_ACCEPTABLE_LEVEL",
    "0S"=>"SOLVE_SUCCEEDED, suboptimal (≥+$(SUBOPT_GAP_PCT)% vs PEtab)",
    "0AS"=>"SOLVED_TO_ACCEPTABLE_LEVEL, suboptimal (≥+$(SUBOPT_GAP_PCT)% vs PEtab)",
    "T"=>"WALLTIME_EXCEEDED (timeout)", "R"=>"RESTORATION_FAILED", "D"=>"SEARCH_DIRECTION_BECOMES_TOO_SMALL",
    "5"=>"other", "E"=>"Error", "-"=>"compile_failed/not_run")
const MADNLP_ORDER = ["0","0A","0S","0AS","T","R","D","5","E","-"]
petab_desc = Dict(
    "0"  => "Converged by G_ABSTOL, ‖g‖∞ ≤ g_tol",
    "0F" => "Converged by F_RELTOL, |Δf| ≤ f_reltol·|f| (with ‖g‖ > g_tol)",
    "0X" => "Converged by X_ABSTOL, ‖Δx‖∞ ≤ x_abstol (with ‖g‖ > g_tol)",
    "T"  => "Walltime exceeded (timeout)",
    "1"  => "Not converged",
    "E"  => "Error", "-" => "compile_failed/not_run")
const PETAB_ORDER = ["0","0F","0X","T","1","E","-"]

madnlp_present = Set(madnlp_code(d,p) for d in all_d for p in ("exagpu_","exacpu_"))
petab_present  = Set(petab_code(d) for d in all_d)
keyline(code, desc) = "   " * lpad(code, 3) * " : " * desc

println(buf, "\n  MadNLP (Status)")
for code in MADNLP_ORDER; code in madnlp_present && println(buf, keyline(code, madnlp_desc[code])); end
println(buf, "\n  PEtab  (Status)")
for code in PETAB_ORDER;  code in petab_present  && println(buf, keyline(code, petab_desc[code]));  end
println(buf, "\n", bar)

# ─── output ───────────────────────────────────────────────────────────────────
report = String(take!(buf))
print(report)
open(REPORT_TXT, "w") do io; print(io, report); end
println("\nReport saved to: $REPORT_TXT")
