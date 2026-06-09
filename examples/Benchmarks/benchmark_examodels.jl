# benchmark_examodels.jl — ExaModelsPEtab + MadNLP GPU benchmark
#
# Builds petab_examodel (with CUDABackend) and solves with MadNLP on GPU for the
# ExaModelsPEtab-supported models (EXA_SUPPORTED_MODELS in list_benchmarks.jl, minus the
# warmup/separately-benchmarked ones). Results are written to Benchmarks/results/{Model}_results.txt
# using prefixed keys (exa_*) so petab results in the same file are preserved.
#
# Compilation timing is split into two phases:
#   Phase 1 (presolve)   — PEtab setup + ODE solve at nominal θ to init the mesh
#   Phase 2 (ExaModels)  — collocation/continuity/objective setup + ExaModel(c) build
# Both timings are stored; final_report.jl computes the %ExaModels column from them.
#
# After a successful first compile+solve, N_SGM_RERUNS additional reruns are performed and
# the geometric mean (SGM, n=10 by default) of the solve times is stored under exa_sgm_*
# keys for timing reproducibility analysis. SGM is skipped unless the first solve is
# SOLVE_SUCCEEDED (timing a diverged / walltime-capped solve is meaningless).
#
# The run is resumable: models with a terminal result are skipped. Wrap with
# benchmark_examodels.sh to restart after a watchdog SIGKILL on compile timeout.
#
# Two-GPU usage (strided partition, one instance per GPU):
#   julia --project=. -t 1 examples/Benchmarks/benchmark_examodels.jl <gpu_id> <num_instances> <instance_idx>
# Single-GPU usage:
#   julia --project=. -t 1 examples/Benchmarks/benchmark_examodels.jl 0 1 0

using ExaModelsPEtab, PEtab, CUDA, MadNLPGPU, CUDSS, ExaModels

# ─── CONFIGURABLE SETTINGS ────────────────────────────────────────────────────
# ALL solver/benchmark settings live in list_benchmarks.jl (single source of truth, shared with
# benchmark_petab.jl). Change them THERE, not here — below we only alias the BENCH_* constants to
# the local names this script uses.
const MODELDIR  = joinpath(@__DIR__, "..", "Benchmark-Models")
const RESULTDIR = joinpath(@__DIR__, "results")

include(joinpath(@__DIR__, "list_benchmarks.jl"))  # model lists + BENCH_* config (K, TOL, SGM_N, limits, ...)

const K             = BENCH_K
const TOL           = BENCH_TOL
const COMPILE_LIMIT = BENCH_COMPILE_LIMIT
const SOLVE_LIMIT   = BENCH_SOLVE_LIMIT
const MAX_ITER      = BENCH_MAX_ITER
const N_SGM_RERUNS  = BENCH_SGM_N
const ACCEPT_TOL    = BENCH_ACCEPT_TOL
const ACCEPT_ITER   = BENCH_ACCEPT_ITER
const WARMUP_MODEL  = BENCH_WARMUP_MODEL
# ──────────────────────────────────────────────────────────────────────────────

# The timed in-loop set = EXA_RERUN_INLOOP in list_benchmarks.jl, ordered clean-success →
# least-reliable so the high-confidence results land first. It excludes ONLY Bruno (the shared
# JIT warmup), benchmarked separately via benchmark_bruno.jl. Crauste is timed here normally
# (warmed on Bruno, like every other in-loop model).
# Override with the BENCH_SUBSET env var (comma-separated, order-preserving) for an ad-hoc set.
const ALL_MODELS = haskey(ENV, "BENCH_SUBSET") ?
    String.(split(ENV["BENCH_SUBSET"], ',')) : EXA_RERUN_INLOOP

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

# Read-modify-write: preserves keys from other benchmarks (e.g. petab_*) in the same file.
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

function exa_finished(m)
    d = read_result(result_path(m))
    cs  = get(d, "exa_compile_status", "")
    ss  = get(d, "exa_solve_status",   "")
    sgm = get(d, "exa_sgm_status",     "")
    # Terminal compile failures + manual skips — nothing left to do
    cs in ("timeout", "error", "missing_yaml", "skipped") && return true
    cs != "ok" && return false
    # First solve failed/timed out — no SGM, done
    ss in ("timeout", "error") && return true
    ss != "ok" && return false
    # First solve succeeded — done only when SGM is also terminal ("skipped" = diverged solve, no SGM)
    return sgm in ("ok", "error", "skipped")
end

# ─── build one ExaModel (compile phases 1+2) ──────────────────────────────────
function build_model(yaml, t_origin)
    PEmodel = PEtab.PEtabModel(yaml)
    PEprob  = PEtab.PEtabODEProblem(PEmodel)
    c = ExaModels.ExaCore(; backend=CUDA.CUDABackend(), concrete=Val(true))

    # Steady-state models (all measurements at time=inf) take the no-collocation-mesh path:
    # variables (zss in place of z) → steady-state residual + conservation → objective.
    if ExaModelsPEtab._is_steady_state(PEmodel)
        c, PEinfo = ExaModelsPEtab._create_variables_ss(c, PEmodel, PEprob)
        t_phase1 = time() - t_origin
        c = ExaModelsPEtab._create_ss_constraints(c, PEmodel, PEprob, PEinfo)
        c, y0, sigma0 = ExaModelsPEtab._create_objective_ss(c, PEmodel, PEprob, PEinfo)
        mdl = ExaModels.ExaModel(c)
        ExaModels.set_start!(mdl, c.y, y0)
        ExaModels.set_start!(mdl, c.sigma, sigma0)
        CUDA.synchronize()
        return mdl, mdl.meta.nvar, mdl.meta.ncon, t_phase1
    end

    c, PEinfo = ExaModelsPEtab._create_variables(c, PEmodel, PEprob, K)
    t_phase1 = time() - t_origin
    c = ExaModelsPEtab._create_collocation(c, PEmodel, PEprob, PEinfo)
    c = ExaModelsPEtab._create_continuity(c, PEmodel, PEprob, PEinfo)
    c, y0, sigma0 = ExaModelsPEtab._create_objective(c, PEmodel, PEprob, PEinfo)
    mdl = ExaModels.ExaModel(c)
    ExaModels.set_start!(mdl, c.y, y0)
    ExaModels.set_start!(mdl, c.sigma, sigma0)
    CUDA.synchronize()
    return mdl, mdl.meta.nvar, mdl.meta.ncon, t_phase1
end

# ─── SGM solve reruns ─────────────────────────────────────────────────────────
# Runs madnlp N_SGM_RERUNS times on an already-built model (GPU kernels cached
# after the first solve), measuring pure MadNLP iteration time. No recompile —
# SGM compile times are not meaningful (Symbolics/GPU already cached) and
# exa_compile_time (first run, cold) is the representative compilation metric.
# madnlp does not mutate model.meta.x0, so all reruns start from the same point.
function run_sgm_reruns(m, rp, model)
    write_result(rp, Dict("exa_sgm_status" => "running", "exa_sgm_n" => N_SGM_RERUNS))
    solve_times = Float64[]
    for i in 1:N_SGM_RERUNS
        @info "[$m] SGM solve $i/$N_SGM_RERUNS ..."
        try
            t0 = time()
            with_hard_deadline(SOLVE_LIMIT + 3600.0) do
                madnlp(model; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
                       max_wall_time=SOLVE_LIMIT, linear_solver=MadNLPGPU.CUDSSSolver)
            end
            push!(solve_times, round(time() - t0; digits=2))
        catch e
            @error "[$m] SGM solve $i failed" exception=(e, catch_backtrace())
            write_result(rp, Dict("exa_sgm_status" => "error",
                                  "exa_sgm_error"  => sprint(showerror, e)))
            return
        end
    end
    sgm_solve = round(exp(sum(log, solve_times) / length(solve_times)); digits=2)
    write_result(rp, Dict(
        "exa_sgm_status"     => "ok",
        "exa_sgm_n"          => N_SGM_RERUNS,
        "exa_sgm_solve_time" => sgm_solve,
    ))
    @info "[$m] SGM done: solve=$sgm_solve s (n=$N_SGM_RERUNS)"
end

# ─── main per-model benchmark ─────────────────────────────────────────────────
function bench_one(m)
    rp   = result_path(m)
    yaml = get_yaml(m)

    d = read_result(rp)
    first_run_done = get(d, "exa_compile_status", "") == "ok" &&
                     get(d, "exa_solve_status",   "") == "ok"

    model = nothing

    if !first_run_done
        if yaml === nothing
            write_result(rp, Dict("exa_compile_status" => "missing_yaml",
                                   "exa_solve_status"   => "skipped"))
            return
        end

        # ── COMPILE ──────────────────────────────────────────────────────────
        write_result(rp, Dict(
            "exa_compile_status" => "compiling", "exa_compile_time" => "",
            "exa_presolve_time"  => "",          "exa_solve_status"  => "skipped",
            "exa_solve_time"     => "",          "exa_term_status"   => "",
            "exa_objective"      => "",          "exa_iter"          => "",
            "exa_nvar"           => "",          "exa_ncon"          => "",
            "exa_error"          => "",
        ))
        @info "[$m] compiling (K=$K, compile_limit=$(COMPILE_LIMIT)s)..."
        try
            t0 = time()
            mdl, nvar, ncon, t_phase1 = with_hard_deadline(COMPILE_LIMIT) do
                build_model(yaml, t0)
            end
            model = mdl
            write_result(rp, Dict(
                "exa_compile_status" => "ok",
                "exa_compile_time"   => round(time() - t0; digits=2),
                "exa_presolve_time"  => round(t_phase1;    digits=2),
                "exa_K"              => K,      # collocation points/interval (mesh fidelity)
                "exa_nvar"           => nvar,   # NLP problem size (decision variables)
                "exa_ncon"           => ncon,
            ))
        catch e
            write_result(rp, Dict(
                "exa_compile_status" => "error",
                "exa_compile_time"   => "",
                "exa_error"          => sprint(showerror, e),
            ))
            @error "[$m] compile failed" exception=(e, catch_backtrace())
            return
        end

        # ── SOLVE (first run — includes GPU kernel compilation) ───────────────
        @info "[$m] solving with MadNLP (max_wall_time=$(SOLVE_LIMIT)s)..."
        write_result(rp, Dict("exa_solve_status" => "solving"))
        try
            t0 = time()
            res = with_hard_deadline(SOLVE_LIMIT + 3600.0) do
                madnlp(model; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
                       max_wall_time=SOLVE_LIMIT, linear_solver=MadNLPGPU.CUDSSSolver)
            end
            write_result(rp, Dict(
                "exa_solve_status" => "ok",
                "exa_solve_time"   => round(time() - t0; digits=2),
                "exa_term_status"  => string(res.status),
                "exa_objective"    => res.objective,
                "exa_iter"         => res.iter,
            ))
        catch e
            write_result(rp, Dict(
                "exa_solve_status" => "error",
                "exa_error"        => sprint(showerror, e),
            ))
            @error "[$m] solve failed" exception=(e, catch_backtrace())
            model = nothing; GC.gc(); CUDA.reclaim()
            return
        end
    end

    # ── SGM SOLVE RERUNS ──────────────────────────────────────────────────────
    # Only when first compile+solve both succeeded and SGM not yet done.
    d2 = read_result(rp)
    if get(d2, "exa_compile_status", "") == "ok" &&
       get(d2, "exa_solve_status",   "") == "ok" &&
       get(d2, "exa_sgm_status",     "") ∉ ("ok", "error", "skipped")

        # SGM reruns are timing-only — skip them unless the first solve actually CONVERGED
        # (SOLVE_SUCCEEDED). SGM on a diverged / walltime-capped solve is wasted GPU time.
        if !(uppercase(get(d2, "exa_term_status", "")) in ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL"))
            write_result(rp, Dict("exa_sgm_status" => "skipped",
                "exa_sgm_error" => "first solve term=$(get(d2, "exa_term_status", ""))∉{SUCCEEDED,ACCEPTABLE}; SGM skipped"))
            @info "[$m] SGM skipped (solve term=$(get(d2, "exa_term_status", "")), not SOLVE_SUCCEEDED)"
        else
            # Resuming: model not in memory — rebuild silently (time not recorded)
            if model === nothing
                yaml === nothing && return
                @info "[$m] rebuilding for SGM (resume)..."
                try
                    t0 = time()
                    model, _, _, _ = with_hard_deadline(COMPILE_LIMIT) do
                        build_model(yaml, t0)
                    end
                    # prime GPU kernels with one solve before timing
                    madnlp(model; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
                           max_wall_time=SOLVE_LIMIT, linear_solver=MadNLPGPU.CUDSSSolver)
                catch e
                    @error "[$m] SGM rebuild failed" exception=(e, catch_backtrace())
                    write_result(rp, Dict("exa_sgm_status" => "error",
                                          "exa_sgm_error"  => sprint(showerror, e)))
                    model = nothing; GC.gc(); CUDA.reclaim()
                    return
                end
            end

            run_sgm_reruns(m, rp, model)
        end
    end

    model = nothing; GC.gc(); CUDA.reclaim()
    @info "[$m] done"
end

function warmup()
    yaml = get_yaml(WARMUP_MODEL); yaml === nothing && return
    @info "warmup: JIT build+solve on $WARMUP_MODEL ..."
    try
        t0 = time()
        mdl, _, _, _ = build_model(yaml, t0)
        CUDA.synchronize()
        madnlp(mdl; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER, max_wall_time=250.0,
               linear_solver=MadNLPGPU.CUDSSSolver)
        mdl = nothing; GC.gc(); CUDA.reclaim()
        @info "warmup done"
    catch e; @warn "warmup failed" exception=(e, catch_backtrace()); end
end

function main()
    gpu_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
    ninst  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    idx    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0
    CUDA.device!(gpu_id)
    @info "instance $idx/$ninst on GPU $gpu_id ($(CUDA.name(CUDA.device())))"
    mkpath(RESULTDIR)

    # Convert in-progress sentinels from a prior killed run to timeout/interrupted
    for m in ALL_MODELS
        d = read_result(result_path(m))
        if get(d, "exa_compile_status", "") == "compiling"
            write_result(result_path(m), Dict("exa_compile_status" => "timeout",
                                               "exa_compile_time"   => string(COMPILE_LIMIT)))
        elseif get(d, "exa_compile_status", "") == "ok" &&
               get(d, "exa_solve_status", "") == "solving"
            write_result(result_path(m), Dict("exa_solve_status" => "timeout"))
        elseif get(d, "exa_sgm_status", "") == "running"
            write_result(result_path(m), Dict("exa_sgm_status" => "interrupted"))
        end
    end

    mine = [m for (i, m) in enumerate(ALL_MODELS) if (i - 1) % ninst == idx]
    todo = filter(!exa_finished, mine)
    @info "instance $idx: $(length(mine)) models assigned, $(length(todo)) remaining"
    isempty(todo) && (@info "nothing to do"; return)

    warmup()
    for (i, m) in enumerate(todo)
        println("\n" * "="^60); println("[$idx][$i/$(length(todo))] $m"); println("="^60)
        bench_one(m)
    end
    @info "instance $idx complete"
end

main()
