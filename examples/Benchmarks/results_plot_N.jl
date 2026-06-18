# results_plot_N.jl — solver speedup vs PEtab.jl over the exa target set, x = N (collocation timesteps).
#   x = N (number of mesh intervals), log10 — data-derived, shared by exa GPU/CPU and PEtab (no NLP needed).
#   y = SGM10-time speedup vs PEtab.jl     — petab_sgm / solver_sgm (log10). PEtab ⇒ 1 (the 10^0 line).
# Same three series and styling as results_plot_nvar.jl; only the x-metric differs.
#
# N (= PEinfo.N) is not stored by the benchmark, so on the first run each model is loaded, build phase-1
# is run, and the PEInfo integer counts (PEinfo_N/K/Nz/Nc/Ncv/Nm/Np) are CACHED into {Model}_results.txt;
# later runs read the cache and skip the heavy load. Steady-state models have no mesh (PEinfo_N=0) and
# are excluded from this plot (no meaningful timestep count on a log-x axis).
#
# Usage: julia --project=examples examples/Benchmarks/results_plot_N.jl [results_dir]

using Plots
using Plots.PlotMeasures   # mm units for plot margins
gr()

const HERE      = @__DIR__
include(joinpath(HERE, "options.jl"))   # EXA_SUPPORTED_MODELS, shifted_geomean, BENCH_SGM_SHIFT, BENCH_K
const RESULTDIR = length(ARGS) >= 1 ? ARGS[1] : joinpath(HERE, "results")
const MODELDIR  = joinpath(HERE, "..", "Benchmark-Models")
const K         = BENCH_K

function read_result(m)
    d = Dict{String,String}(); p = joinpath(RESULTDIR, "$(m)_results.txt")
    isfile(p) || return d
    for line in eachline(p)
        i = findfirst('=', line); i === nothing && continue
        d[line[1:i-1]] = line[i+1:end]
    end
    d
end

# read-modify-write that preserves all existing keys (other backends / PEtab / prior cache)
function write_result(path, updates)
    existing = Dict{String,String}()
    if isfile(path)
        for line in eachline(path)
            i = findfirst('=', line); i === nothing && continue
            existing[line[1:i-1]] = line[i+1:end]
        end
    end
    merged = merge(existing, Dict(string(k) => string(v) for (k, v) in updates))
    open(path, "w") do io
        for k in sort(collect(keys(merged))); println(io, "$k=", merged[k]); end
    end
end

# locate a model's PEtab .yaml under Benchmark-Models/<model>/
get_yaml(m) = begin
    d = joinpath(MODELDIR, m); isdir(d) || return nothing
    fs = filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))
    isempty(fs) ? nothing : joinpath(d, first(fs))
end

# PEInfo integer count fields cached per model (K-independent except K itself)
const _PEINFO_KEYS = ("Np", "Nz", "Nc", "Ncv", "Nm", "N", "K")

# build phase-1 only (no codegen) to obtain PEinfo, then cache its integer counts into the result file
function cache_peinfo!(m)
    yaml = get_yaml(m); yaml === nothing && return
    PEmodel = PEtab.PEtabModel(yaml)
    PEprob  = PEtab.PEtabODEProblem(PEmodel)
    c = ExaModels.ExaCore(; concrete = Val(true))   # CPU core; the counts are backend-independent
    PEinfo = ExaModelsPEtab._is_steady_state(PEmodel) ?
        ExaModelsPEtab._create_variables_ss(c, PEmodel, PEprob)[2] :
        ExaModelsPEtab._create_variables(c, PEmodel, PEprob, K)[2]
    write_result(joinpath(RESULTDIR, "$(m)_results.txt"),
                 Dict("PEinfo_$f" => getfield(PEinfo, Symbol(f)) for f in _PEINFO_KEYS))
end

# one-time cache fill: only load the heavy build stack if some model is missing its PEInfo counts
missing_models = [m for m in EXA_SUPPORTED_MODELS if !haskey(read_result(m), "PEinfo_N")]
if !isempty(missing_models)
    @info "caching PEInfo counts for $(length(missing_models)) model(s) (one-time)..."
    @eval using ExaModelsPEtab, PEtab, ExaModels
    for m in missing_models
        try; Base.invokelatest(cache_peinfo!, m); catch e; @warn "PEInfo cache failed for $m" exception=e; end
    end
end

# SGM10 [s] for a backend prefix (same logic as results_plot_nvar.jl)
function sgm10(d, pfx)
    get(d, pfx * "sgm_status", "") == "ok" || return nothing
    raw = get(d, pfx * "solve_times", "")
    if !isempty(raw)
        ts = Float64[x for x in (tryparse(Float64, s) for s in split(raw, ",")) if x !== nothing]
        !isempty(ts) && return shifted_geomean(ts, BENCH_SGM_SHIFT)
    end
    tryparse(Float64, get(d, pfx * "sgm_solve_time", ""))
end

# N for the x-axis: cached PEinfo_N. Steady-state models have no mesh (PEinfo_N=0) and no
# meaningful N, so they are EXCLUDED from this plot entirely (return nothing ⇒ main loop skips).
function get_N(d)
    v = tryparse(Int, get(d, "PEinfo_N", ""))
    (v === nothing || v == 0) && return nothing
    return v
end

const SUBOPT_GAP_PCT = 2.0
function gap_val(d, pfx)
    eo = tryparse(Float64, get(d, pfx * "petab_obj", ""))
    eo === nothing && (eo = tryparse(Float64, get(d, pfx * "objective", "")))
    po = tryparse(Float64, get(d, "petab_objective", ""))
    (eo === nothing || po === nothing || !isfinite(eo) || !isfinite(po) || po == 0.0) && return nothing
    (eo - po) / abs(po) * 100.0
end
function madnlp_code(d, pfx)
    get(d, pfx * "compile_status", "") == "ok" || return "-"
    ss = get(d, pfx * "solve_status", "")
    ss == "skipped" && return "-"
    ss == "timeout" && return "T"
    ss == "error"   && return "E"
    term = uppercase(get(d, pfx * "term_status", ""))
    isempty(term) && return "-"
    if occursin("SUCCEEDED", term) || occursin("ACCEPTABLE", term)
        base = occursin("ACCEPTABLE", term) ? "0A" : "0"
        gp   = gap_val(d, pfx)
        return (gp !== nothing && gp >= SUBOPT_GAP_PCT) ? base * "S" : base
    end
    return "5"
end

# Julia logo colors
const J_PURPLE = RGB(0.584, 0.345, 0.698)
const J_GREEN  = RGB(0.220, 0.596, 0.149)
const J_RED    = RGB(0.796, 0.235, 0.200)

gpu_x = Float64[]; gpu_y = Float64[]
cpu_x = Float64[]; cpu_y = Float64[]
pet_x  = Float64[]; pet_y  = Float64[]   # PEtab baseline where ExaModels also solved (0/0A)
petx_x = Float64[]; petx_y = Float64[]   # PEtab solved but ExaModels did NOT (the 9/20) ⇒ X'd boxes
for m in EXA_SUPPORTED_MODELS
    d  = read_result(m)
    nv = get_N(d);               nv === nothing && continue       # no cached N ⇒ no x position
    pt = sgm10(d, "petab_"); (pt === nothing || pt <= 0) && continue   # PEtab baseline must have solved
    if madnlp_code(d, "exagpu_") in ("0", "0A")                   # ExaModels solved ⇒ full GPU/CPU/baseline set
        g  = sgm10(d, "exagpu_");    (g !== nothing && g > 0) && (push!(gpu_x, nv); push!(gpu_y, pt / g))
        c  = sgm10(d, "exacpu_");    (c !== nothing && c > 0) && (push!(cpu_x, nv); push!(cpu_y, pt / c))
        push!(pet_x, nv); push!(pet_y, 1.0)
    else                                                          # PEtab solved but ExaModels did not (9/20)
        push!(petx_x, nv); push!(petx_y, 1.0)
    end
end

# ticks at every order of 10 over the data range (ignore non-positive values: log axis can't show N=0)
prange(v) = (vp = filter(>(0), v); isempty(vp) ? (0:0) : (floor(Int, log10(minimum(vp))):ceil(Int, log10(maximum(vp)))))
xt = [10.0^k for k in prange(vcat(gpu_x, cpu_x, pet_x, petx_x))]
yt = [10.0^k for k in prange(vcat(gpu_y, cpu_y, pet_y, petx_y))]

# common x-range (small log-padding) so the axis + trend lines span the full plot, edge to edge
_xa = filter(>(0), vcat(gpu_x, cpu_x, pet_x, petx_x))
_xlo, _xhi = isempty(_xa) ? (1.0, 10.0) : extrema(_xa)
_pf  = (_xhi / _xlo) ^ 0.02            # ~2% log-padding each side
XLIM = (_xlo / _pf, _xhi * _pf)

# ── series 1: ExaModels + MadNLP (GPU). Global cosmetics live on this scatter() call. ──
plt = scatter(gpu_x, gpu_y;
              label  = "ExaModels + MadNLP (GPU)",  # legend text
              marker = :circle,                     # marker shape
              ms     = 8,                            # marker size
              mc     = J_PURPLE,                     # marker fill color
              msc    = :black,                       # marker outline color
              msw    = 1.5,                          # outline width
              malpha = 1.0,                          # marker opacity (0–1)
              xscale = :log10,                       # x log scale
              yscale = :log10,                       # y log scale
              xticks = xt,                           # x ticks (powers of 10)
              yticks = yt,                           # y ticks (powers of 10)
              xlims  = XLIM,                          # x-limits (trend lines span this)
              xlabel = "Number of timesteps (N)",    # x axis label
              ylabel = "Speedup",                    # y axis label
              guidefontsize  = 15,                   # axis-label font size
              tickfontsize   = 11,                   # tick-number font size
              legendfontsize = 11,                   # legend font size
              legend     = :bottomright,                # legend position
              size       = (820, 460),               # figure size (px)
              grid       = true,                     # gridlines on/off
              gridalpha  = 0.2,                      # gridline opacity
              framestyle = :box,                     # axes frame style
              left_margin   = 3mm,                   # room for y-label (raise if clipped)
              bottom_margin = 3mm,                   # room for x-label (raise if clipped)
              top_margin    = 1mm,                   # room above plot
              right_margin  = 0mm)                   # room at right
# ── series 2 & 3: only per-series overrides (shape, size, colors); cosmetics inherit from above ──
scatter!(plt, cpu_x, cpu_y;
         label = "ExaModels + MadNLP (CPU)", marker = :utriangle,  # green triangles
         ms = 7, mc = J_GREEN, msc = :black, msw = 1.0, malpha = 1.0)
scatter!(plt, pet_x, pet_y;
         label = "PEtab.jl + IPNewton (CPU)", marker = :square,    # red squares (baseline)
         ms = 6, mc = J_RED, msc = :black, msw = 1.0, malpha = 1.0)
# PEtab solved but ExaModels did NOT reach an optimum (the 9/20): red box stamped with a black X.
scatter!(plt, petx_x, petx_y;
         label = "", marker = :square,                             # red box (unlabeled; X overlay labels it)
         ms = 6, mc = J_RED, msc = :black, msw = 1.0, malpha = 1.0)
scatter!(plt, petx_x, petx_y;
         label = "ExaModels failed or suboptimal", marker = :xcross,  # black X inside the box
         ms = 5, mc = :black, msc = :black, msw = 2.0, malpha = 1.0)

# least-squares trend line per solver series, fit in log–log space (power-law); thin dashed, no legend
function regline!(plt, x, y, color, xspan)
    # keep only positive pairs: log10 needs x,y > 0. Steady-state models sit at N=0 and would
    # otherwise inject log10(0) = -Inf into the fit (NaN slope ⇒ no line drawn).
    xp = Float64[]; yp = Float64[]
    for (xi, yi) in zip(x, y); (xi > 0 && yi > 0) && (push!(xp, xi); push!(yp, yi)); end
    length(xp) < 2 && return
    lx = log10.(xp); ly = log10.(yp); n = length(lx)
    mx = sum(lx) / n; my = sum(ly) / n; sxx = sum((lx .- mx) .^ 2)
    sxx == 0 && return
    b = sum((lx .- mx) .* (ly .- my)) / sxx; a = my - b * mx     # ly = a + b·lx  (fit on the data)
    xs = collect(xspan)                                          # but draw across the full plot width
    plot!(plt, xs, 10.0 .^ (a .+ b .* log10.(xs)); ls = :dash, lw = 1, lc = color, label = "")
end
regline!(plt, gpu_x, gpu_y, J_PURPLE, XLIM)   # GPU trend (purple)
regline!(plt, cpu_x, cpu_y, J_GREEN, XLIM)    # CPU trend (green)

hline!(plt, [1.0]; ls = :dash, lc = :gray, lw = 1, label = "")     # dashed PEtab baseline at y=1

out = joinpath(HERE, "results_plot_N.png")
savefig(plt, out)
println("saved: $out")
println("points plotted — GPU: $(length(gpu_x)), CPU: $(length(cpu_x)), PEtab: $(length(pet_x)), PEtab-only (X'd): $(length(petx_x))")
plt   # return the plot so `include("results_plot_N.jl")` displays it in the REPL / IDE plot pane
