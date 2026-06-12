# run_petab.jl — PEtab.jl + Optim.IPNewton benchmark
#
# Compiles and solves each PEtab model with Optim.IPNewton (PEtab.jl's recommended
# optimizer). Results written to Benchmarks/results/{Model}_results.txt using prefixed
# keys (petab_*) so ExaModels results in the same file are preserved.
#
# Intended to be launched once per model (many in parallel from a shell loop):
#   for m in <model_list>; do
#     julia --project=examples -t 1 examples/Benchmarks/run_petab.jl "$m" &
#   done
# Or run serially:
#   julia --project=examples -t 1 examples/Benchmarks/run_petab.jl Bachmann_MSB2011
#
# To run all 35 models in parallel (up to PAR concurrent workers):
#   PAR=12
#   for m in $(julia --project=examples -e 'include("examples/Benchmarks/run_petab.jl"); print_models()'); do
#     while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 2; done
#     julia --project=examples -t 1 examples/Benchmarks/run_petab.jl "$m" &
#   done; wait

using PEtab, Optim

# ─── CONFIGURABLE SETTINGS ────────────────────────────────────────────────────
# ALL solver/benchmark settings live in options.jl (single source of truth, shared with
# run_examodels.jl). Change them THERE, not here — below we only alias the BENCH_* constants.
const MODELDIR  = joinpath(@__DIR__, "..", "Benchmark-Models")
const RESULTDIR = joinpath(@__DIR__, "results")

include(joinpath(@__DIR__, "options.jl"))  # model lists + BENCH_* config (TOL, SGM_N, limits, ...)

const TOL           = BENCH_TOL                  # Optim g_tol (== MadNLP tol)
const SOLVE_LIMIT   = BENCH_SOLVE_LIMIT          # Optim time_limit [s] (== MadNLP max_wall_time)
const COMPILE_LIMIT = BENCH_PETAB_COMPILE_LIMIT  # PEtab build wall cap [s]
const MAX_ITER      = BENCH_MAX_ITER
const N_SGM_RERUNS  = BENCH_SGM_N                 # shared SGM rerun count (== exa's)
const WARMUP_MODEL  = BENCH_WARMUP_MODEL          # benchmark Bruno separately via run_bruno.jl (Crauste-warmed)
# ──────────────────────────────────────────────────────────────────────────────

# PEtab attempts EVERY benchmark model (optimum-found is only known after solving), minus the
# shared JIT warmup Bruno (benchmarked via run_bruno.jl). Only used by print_models() /
# the .sh driver; run_worker() benchmarks whatever model name is passed as ARGS[1].
const ALL_MODELS = filter(!=(WARMUP_MODEL), BENCHMARK_MODELS)  # 34

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
    f_reltol       = BENCH_PETAB_F_RELTOL,
    allow_f_increases = true,
    successive_f_tol  = BENCH_PETAB_SUCCESSIVE_FTOL,
    show_trace     = false,
    x_abstol       = BENCH_PETAB_X_ABSTOL,
)

# ─── PEtab SGM solve reruns ─────────────────────────────────────────────────────
# Mirrors the exa SGM pass: re-solve the SAME compiled PEtabODEProblem from the SAME nominal
# start N_SGM_RERUNS times and store the geometric mean solve time under petab_sgm_* — so the
# PEtab column is an n=10 head-to-head against exa_sgm_solve_time. Only meaningful after a
# converged first solve (petab_optimum_found=true); skipped otherwise.
function run_petab_sgm(rp, PEprob)
    write_result(rp, Dict("petab_sgm_status" => "running", "petab_sgm_n" => N_SGM_RERUNS))
    solve_times = Float64[]
    for i in 1:N_SGM_RERUNS
        @info "[petab SGM] solve $i/$N_SGM_RERUNS ..."
        try
            t0 = time()
            with_hard_deadline(SOLVE_LIMIT + 600.0) do
                calibrate(PEprob, get_x(PEprob), Optim.IPNewton(); options=optim_opts())
            end
            push!(solve_times, round(time() - t0; digits=2))
        catch e
            @error "[petab SGM] solve $i failed" exception=(e, catch_backtrace())
            write_result(rp, Dict("petab_sgm_status" => "error",
                                  "petab_sgm_error"  => sprint(showerror, e)))
            return
        end
    end
    sgm_solve = round(exp(sum(log, solve_times) / length(solve_times)); digits=2)
    write_result(rp, Dict("petab_sgm_status"     => "ok",
                          "petab_sgm_n"          => N_SGM_RERUNS,
                          "petab_sgm_solve_time" => sgm_solve))
    @info "[petab SGM] done: solve=$sgm_solve s (n=$N_SGM_RERUNS)"
end

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

    # ── SGM SOLVE RERUNS (timing-only; needs a converged first solve) ─────────────
    if N_SGM_RERUNS > 0
        d = read_result(rp)
        if get(d, "petab_optimum_found", "") == "true" &&
           get(d, "petab_sgm_status", "") ∉ ("ok", "error")
            run_petab_sgm(rp, PEprob)
        elseif get(d, "petab_optimum_found", "") != "true"
            write_result(rp, Dict("petab_sgm_status" => "skipped",
                "petab_sgm_error" => "first solve optimum_found=$(get(d,"petab_optimum_found",""))≠true; SGM skipped"))
        end
    end
end

isempty(ARGS) && error("usage: run_petab.jl <model_name>")
run_worker(ARGS[1])
