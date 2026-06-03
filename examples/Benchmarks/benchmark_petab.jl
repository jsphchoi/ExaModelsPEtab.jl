# benchmark_petab.jl — PEtab.jl + Optim.IPNewton benchmark
#
# Compiles and solves each PEtab model with Optim.IPNewton (PEtab.jl's recommended
# optimizer). Results written to Benchmarks/results/{Model}_results.txt using prefixed
# keys (petab_*) so ExaModels results in the same file are preserved.
#
# Intended to be launched once per model (many in parallel from a shell loop):
#   for m in <model_list>; do
#     julia --project=. -t 1 examples/Benchmarks/benchmark_petab.jl "$m" &
#   done
# Or run serially:
#   julia --project=. -t 1 examples/Benchmarks/benchmark_petab.jl Bachmann_MSB2011
#
# To run all 35 models in parallel (up to PAR concurrent workers):
#   PAR=12
#   for m in $(julia --project=. -e 'include("examples/Benchmarks/benchmark_petab.jl"); print_models()'); do
#     while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 2; done
#     julia --project=. -t 1 examples/Benchmarks/benchmark_petab.jl "$m" &
#   done; wait

using PEtab, Optim

# ─── CONFIGURABLE SETTINGS ────────────────────────────────────────────────────
const TOL           = 1e-6            # Optim g_tol (matches MadNLP tol)
const SOLVE_LIMIT   = 86400.0         # Optim time_limit [s] (24 hr, matches MadNLP)
const COMPILE_LIMIT = 1800.0          # hard compile deadline [s] (30 min)
const MAX_ITER      = 100_000_000     # large so wall time is always the bottleneck
const WARMUP_MODEL  = "Bruno_JExpBot2016"
# ──────────────────────────────────────────────────────────────────────────────

const MODELDIR  = joinpath(@__DIR__, "..", "Benchmark-Models")
const RESULTDIR = joinpath(@__DIR__, "results")

const ALL_MODELS = [
    "Alkan_SciSignal2018", "Armistead_CellDeathDis2024", "Bachmann_MSB2011",
    "Beer_MolBioSystems2014", "Bertozzi_PNAS2020", "Blasi_CellSystems2016",
    "Boehm_JProteomeRes2014", "Borghans_BiophysChem1997", "Brannmark_JBC2010",
    "Bruno_JExpBot2016", "Chen_MSB2009", "Crauste_CellSystems2017",
    "Elowitz_Nature2000", "Fiedler_BMCSystBiol2016", "Froehlich_CellSystems2018",
    "Fujita_SciSignal2010", "Giordano_Nature2020", "Isensee_JCB2018",
    "Lang_PLOSComputBiol2024", "Laske_PLOSComputBiol2019", "Liu_IFACPapersOnLine2025",
    "Lucarelli_CellSystems2018", "Okuonghae_ChaosSolitonsFractals2020",
    "Oliveira_NatCommun2021", "Perelson_Science1996", "Rahman_MBS2016",
    "Raia_CancerResearch2011", "Raimundez_PCB2020", "SalazarCavazos_MBoC2020",
    "Schwen_PONE2014", "Smith_BMCSystBiol2013", "Sneyd_PNAS2002",
    "Weber_BMC2015", "Zhao_QuantBiol2020", "Zheng_PNAS2012",
]

print_models() = foreach(m -> print(m, " "), ALL_MODELS)

get_yaml(m) = begin
    d = joinpath(MODELDIR, m); isdir(d) || return nothing
    fs = filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))
    isempty(fs) ? nothing : joinpath(d, first(fs))
end

result_path(m) = joinpath(RESULTDIR, "$(m)_results.txt")

function read_result(path)
    d = Dict{String,String}()
    isfile(path) || return d
    for line in eachline(path)
        i = findfirst('=', line); i === nothing && continue
        d[line[1:i-1]] = line[i+1:end]
    end
    return d
end

function write_result(path, updates)
    existing = read_result(path)
    merged = merge(existing, Dict(string(k) => replace(string(v), '\n' => ' ', '\r' => ' ')
                                  for (k, v) in updates))
    open(path, "w") do io
        for k in sort(collect(keys(merged))); println(io, "$k=", merged[k]); end
        flush(io)
    end
end

function with_hard_deadline(f, seconds::Real)
    pid = getpid()
    w = run(`bash -c "sleep $(seconds); kill -9 $(pid)"`; wait=false)
    try; return f(); finally; try; kill(w); catch; end; end
end

function has_events(PEprob)
    cbs = PEprob.model_info.simulation_info.callbacks
    for (_, cb) in cbs
        cc = hasproperty(cb, :continuous_callbacks) ? getproperty(cb, :continuous_callbacks) : ()
        dc = hasproperty(cb, :discrete_callbacks)   ? getproperty(cb, :discrete_callbacks)   : ()
        (!isempty(cc) || !isempty(dc)) && return true
    end
    return false
end

optim_opts() = Optim.Options(
    iterations     = MAX_ITER,
    time_limit     = SOLVE_LIMIT,
    g_tol          = TOL,
    f_reltol       = 1e-8,
    allow_f_increases = true,
    successive_f_tol  = 3,
    show_trace     = false,
    x_abstol       = 0.0,
)

function run_worker(m)
    mkpath(RESULTDIR)
    rp = result_path(m)
    yaml = get_yaml(m)

    if yaml === nothing
        write_result(rp, Dict("petab_compile_status" => "missing_yaml",
                               "petab_solve_status"   => "skipped"))
        return
    end

    # ── COMPILE ──────────────────────────────────────────────────────────────
    write_result(rp, Dict(
        "petab_compile_status" => "compiling", "petab_compile_time" => "",
        "petab_solve_status"   => "skipped",   "petab_solve_time"   => "",
        "petab_objective"      => "",           "petab_iter"        => "",
        "petab_optimum_found"  => "",           "petab_has_events"  => "",
        "petab_error"          => "",
    ))
    @info "[$m] PEtab compiling (limit=$(COMPILE_LIMIT)s)..."

    # JIT warmup: pay the generic ForwardDiff/Optim JIT cost on a small model
    try
        wy = get_yaml(WARMUP_MODEL)
        if wy !== nothing
            wp = PEtabODEProblem(PEtabModel(wy))
            calibrate(wp, get_x(wp), Optim.IPNewton();
                      options=Optim.Options(iterations=3))
        end
    catch; end

    PEprob = nothing
    try
        t0 = time()
        PEprob = with_hard_deadline(COMPILE_LIMIT) do
            p  = PEtabODEProblem(PEtabModel(yaml))
            x0 = get_x(p); np = length(x0)
            p.nllh(x0); p.grad!(zeros(np), x0); p.hess!(zeros(np, np), x0)
            p
        end
        write_result(rp, Dict(
            "petab_compile_status" => "ok",
            "petab_compile_time"   => round(time() - t0; digits=2),
            "petab_has_events"     => has_events(PEprob),
        ))
    catch e
        write_result(rp, Dict(
            "petab_compile_status" => "error",
            "petab_error"          => sprint(showerror, e),
        ))
        return
    end

    # ── SOLVE ─────────────────────────────────────────────────────────────────
    @info "[$m] PEtab solving (Optim.IPNewton, time_limit=$(SOLVE_LIMIT)s)..."
    write_result(rp, Dict("petab_solve_status" => "solving"))
    try
        t0 = time()
        res = with_hard_deadline(SOLVE_LIMIT + 600.0) do
            calibrate(PEprob, get_x(PEprob), Optim.IPNewton(); options=optim_opts())
        end
        write_result(rp, Dict(
            "petab_solve_status"  => (res.converged === :Optimisation_failed || !isfinite(res.fmin)) ?
                                     "error" : "ok",
            "petab_solve_time"    => round(time() - t0; digits=2),
            "petab_objective"     => res.fmin,
            "petab_iter"          => res.niterations,
            "petab_optimum_found" => string(res.converged === true),
        ))
    catch e
        write_result(rp, Dict(
            "petab_solve_status" => "error",
            "petab_error"        => sprint(showerror, e),
        ))
    end
end

isempty(ARGS) && error("usage: benchmark_petab.jl <model_name>")
run_worker(ARGS[1])
