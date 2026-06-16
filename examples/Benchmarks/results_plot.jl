# create_plot.jl — single scatter figure of solver speedup vs PEtab.jl over the exa target set.
#   x = nvar (NLP size), log10            — exa GPU and CPU share a model's nvar; PEtab points are
#                                            placed at that same model's nvar (PEtab has no NLP).
#   y = SGM10-time speedup vs PEtab.jl    — petab_sgm / solver_sgm  (log10). PEtab ⇒ 1 (the 10^0 line).
# Three series (legend order): MadNLP (GPU), MadNLP (CPU), IPNewton (CPU).
#
# Usage: julia --project=examples examples/Benchmarks/create_plot.jl [results_dir]
#   results_dir defaults to examples/Benchmarks/results.

using Plots
using Plots.PlotMeasures   # mm units for plot margins
gr()

const HERE      = @__DIR__
include(joinpath(HERE, "options.jl"))   # EXA_SUPPORTED_MODELS, shifted_geomean, BENCH_SGM_SHIFT, BENCH_K
const RESULTDIR = length(ARGS) >= 1 ? ARGS[1] : joinpath(HERE, "results")

function read_result(m)
    d = Dict{String,String}(); p = joinpath(RESULTDIR, "$(m)_results.txt")
    isfile(p) || return d
    for line in eachline(p)
        i = findfirst('=', line); i === nothing && continue
        d[line[1:i-1]] = line[i+1:end]
    end
    d
end

# SGM10 [s] for a backend prefix: report-time shift from raw solve_times when present, else the
# stored aggregate (legacy files). Returns nothing if there is no warm timing.
function sgm10(d, pfx)
    get(d, pfx * "sgm_status", "") == "ok" || return nothing
    raw = get(d, pfx * "solve_times", "")
    if !isempty(raw)
        ts = Float64[x for x in (tryparse(Float64, s) for s in split(raw, ",")) if x !== nothing]
        !isempty(ts) && return shifted_geomean(ts, BENCH_SGM_SHIFT)
    end
    tryparse(Float64, get(d, pfx * "sgm_solve_time", ""))
end

function get_nvar(d)
    for k in ("exagpu_nvar", "exacpu_nvar")
        v = tryparse(Int, get(d, k, "")); v !== nothing && return v
    end
    return nothing   # model not built yet (cleared / mid-rerun) ⇒ no x position
end

# Julia logo colors
const J_PURPLE = RGB(0.584, 0.345, 0.698)
const J_GREEN  = RGB(0.220, 0.596, 0.149)
const J_RED    = RGB(0.796, 0.235, 0.200)

gpu_x = Float64[]; gpu_y = Float64[]
cpu_x = Float64[]; cpu_y = Float64[]
pet_x = Float64[]; pet_y = Float64[]
for m in EXA_SUPPORTED_MODELS
    d  = read_result(m)
    nv = get_nvar(d);            nv === nothing && continue       # never built ⇒ no x position
    pt = sgm10(d, "petab_");     (pt === nothing || pt <= 0) && continue   # need the PEtab baseline
    g  = sgm10(d, "exagpu_");    (g !== nothing && g > 0) && (push!(gpu_x, nv); push!(gpu_y, pt / g))
    c  = sgm10(d, "exacpu_");    (c !== nothing && c > 0) && (push!(cpu_x, nv); push!(cpu_y, pt / c))
    push!(pet_x, nv); push!(pet_y, 1.0)
end

# ticks at every order of 10 over the data range
prange(v) = isempty(v) ? (0:0) : (floor(Int, log10(minimum(v))):ceil(Int, log10(maximum(v))))
xt = [10.0^k for k in prange(vcat(gpu_x, cpu_x, pet_x))]
yt = [10.0^k for k in prange(vcat(gpu_y, cpu_y, pet_y))]

# common x-range (small log-padding) so the axis + trend lines span the full plot, edge to edge
_xa = vcat(gpu_x, cpu_x, pet_x)
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
              xlabel = "Number of variables",        # x axis label
              ylabel = "Speedup",                    # y axis label
              guidefontsize  = 14,                   # axis-label font size
              tickfontsize   = 11,                   # tick-number font size
              legendfontsize = 11,                   # legend font size
              legend     = :topright,                # legend position
              size       = (820, 460),               # figure size (px)
              grid       = true,                     # gridlines on/off
              gridalpha  = 0.2,                      # gridline opacity
              framestyle = :box,                     # axes frame style
              left_margin   = 3mm,                   # room for y-label (raise if clipped)
              bottom_margin = 3mm,                   # room for x-label (raise if clipped)
              top_margin    = 0mm,                   # room above plot
              right_margin  = 0mm)                   # room at right
# ── series 2 & 3: only per-series overrides (shape, size, colors); cosmetics inherit from above ──
scatter!(plt, cpu_x, cpu_y;
         label = "ExaModels + MadNLP (CPU)", marker = :utriangle,  # green triangles
         ms = 7, mc = J_GREEN, msc = :black, msw = 1.0, malpha = 1.0)
scatter!(plt, pet_x, pet_y;
         label = "PEtab.jl + IPNewton (CPU)", marker = :square,    # red squares (baseline)
         ms = 6, mc = J_RED, msc = :black, msw = 1.0, malpha = 1.0)

# least-squares trend line per solver series, fit in log–log space (power-law); thin dashed, no legend
function regline!(plt, x, y, color, xspan)
    length(x) < 2 && return
    lx = log10.(x); ly = log10.(y); n = length(lx)
    mx = sum(lx) / n; my = sum(ly) / n; sxx = sum((lx .- mx) .^ 2)
    sxx == 0 && return
    b = sum((lx .- mx) .* (ly .- my)) / sxx; a = my - b * mx     # ly = a + b·lx  (fit on the data)
    xs = collect(xspan)                                          # but draw across the full plot width
    plot!(plt, xs, 10.0 .^ (a .+ b .* log10.(xs)); ls = :dash, lw = 1, lc = color, label = "")
end
regline!(plt, gpu_x, gpu_y, J_PURPLE, XLIM)   # GPU trend (purple)
regline!(plt, cpu_x, cpu_y, J_GREEN, XLIM)    # CPU trend (green)

hline!(plt, [1.0]; ls = :dash, lc = :gray, lw = 1, label = "")     # dashed PEtab baseline at y=1

out = joinpath(HERE, "results_plot.png")
savefig(plt, out)
println("saved: $out")
println("points plotted — GPU: $(length(gpu_x)), CPU: $(length(cpu_x)), PEtab: $(length(pet_x))")
plt   # return the plot so `include("results_plot.jl")` displays it in the REPL / IDE plot pane
