# benchmark_bruno.jl — dedicated Bruno benchmark, warmed up on Crauste.
#
# Bruno_JExpBot2016 is the SHARED JIT-warmup model for both benchmark_examodels.jl and
# benchmark_petab.jl, so it is excluded from both of their timed runs: a model that is
# benchmarked by the same script that warms up on it gets a pre-warmed, invalid compile
# time (Bruno's petab_compile_time came out 4.96 s vs ~200-300 s for its cold peers).
#
# This script benchmarks the TARGET with BOTH backends in a single process, warming up on a
# DIFFERENT model (never the target), so the target's exa_* and petab_* numbers are valid and
# comparable. The measurement logic is copied verbatim from the two parent scripts to keep parity.
#
# CANONICAL USE: Bruno only. Bruno is the sole model that cannot be timed by its own warming
# scripts, so it is benchmarked here, warmed on Crauste:
#   julia --project=. -t 1 examples/Benchmarks/benchmark_bruno.jl Bruno_JExpBot2016   [gpu_id]
# Crauste itself is now timed normally in the main loops (benchmark_examodels.jl + _petab.jl,
# warmed on Bruno) and no longer needs this script. The TARGET arg still accepts Crauste (warmed
# on Bruno) for ad-hoc parity checks, but the canonical Crauste numbers come from the main loops.
# (no target arg ⇒ defaults to Bruno, warmed on Crauste.)

using ExaModelsPEtab, PEtab, CUDA, MadNLPGPU, CUDSS, ExaModels, Optim

# ─── CONFIGURABLE SETTINGS (must match the two parent scripts) ──────────────────
const TARGET        = (length(ARGS) >= 1 && !occursin(r"^\d+$", ARGS[1])) ? ARGS[1] : "Bruno_JExpBot2016"  # model under test (arg 1)
const WARMUP_MODEL  = TARGET == "Crauste_CellSystems2017" ? "Bruno_JExpBot2016" : "Crauste_CellSystems2017"  # warmup ≠ target ⇒ valid timing
const K             = parse(Int, get(ENV, "EXA_K", "4"))       # collocation points per mesh interval (env-overridable; matches benchmark_examodels.jl)
const TOL           = 1e-6             # solver tolerance (both)
const COMPILE_LIMIT = 14400.0          # exa compile deadline [s] (4 hr)
const PETAB_COMPILE_LIMIT = 1800.0     # petab compile deadline [s] (30 min)
const SOLVE_LIMIT   = 7200.0          # max_wall_time / time_limit [s] (2 hr; matches the parent scripts)
const MAX_ITER      = 100_000_000
const N_SGM_RERUNS  = parse(Int, get(ENV, "EXA_SGM_N", "5"))   # exa rerun count for geometric mean timing (env-overridable; canonical n=5)
const ACCEPT_TOL    = parse(Float64, get(ENV, "EXA_ACCEPT_TOL", "1e-4"))   # ε-optimal acceptable termination (see benchmark_examodels.jl)
const ACCEPT_ITER   = parse(Int,     get(ENV, "EXA_ACCEPT_ITER", "10"))
const PETAB_SGM_N   = parse(Int, get(ENV, "PETAB_SGM_N", "5")) # petab rerun count for geometric mean timing (env-overridable)
# ────────────────────────────────────────────────────────────────────────────────

const MODELDIR  = joinpath(@__DIR__, "..", "Benchmark-Models")
const RESULTDIR = joinpath(@__DIR__, "results")

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

# Read-modify-write: preserves keys from the other backend in the same file.
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

# ─── ExaModels build (copied from benchmark_examodels.jl) ───────────────────────
function build_model(yaml, t_origin)
    PEmodel = PEtab.PEtabModel(yaml)
    PEprob  = PEtab.PEtabODEProblem(PEmodel)
    c = ExaModels.ExaCore(; backend=CUDA.CUDABackend(), concrete=Val(true))
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

# ─── SGM solve reruns (copied from benchmark_examodels.jl) ──────────────────────
function run_sgm_reruns(m, rp, model)
    write_result(rp, Dict("exa_sgm_status" => "running", "exa_sgm_n" => N_SGM_RERUNS))
    solve_times = Float64[]
    for i in 1:N_SGM_RERUNS
        @info "[$m] SGM solve $i/$N_SGM_RERUNS ..."
        try
            t0 = time()
            madnlp(model; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
                   max_wall_time=SOLVE_LIMIT, linear_solver=MadNLPGPU.CUDSSSolver)
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

# ─── ExaModels benchmark for the target (compile + first solve + SGM) ───────────
function bench_exa(m)
    rp   = result_path(m)
    yaml = get_yaml(m)
    yaml === nothing && (write_result(rp, Dict("exa_compile_status" => "missing_yaml",
                                               "exa_solve_status"   => "skipped")); return)
    model = nothing

    # ── COMPILE ──
    write_result(rp, Dict(
        "exa_compile_status" => "compiling", "exa_compile_time" => "",
        "exa_presolve_time"  => "",          "exa_solve_status"  => "skipped",
        "exa_solve_time"     => "",          "exa_term_status"   => "",
        "exa_objective"      => "",          "exa_iter"          => "",
        "exa_nvar"           => "",          "exa_ncon"          => "",
        "exa_error"          => "",
    ))
    @info "[$m] EXA compiling (K=$K)..."
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
        write_result(rp, Dict("exa_compile_status" => "error",
                              "exa_error" => sprint(showerror, e)))
        @error "[$m] EXA compile failed" exception=(e, catch_backtrace())
        return
    end

    # ── FIRST SOLVE (includes GPU kernel compilation) ──
    @info "[$m] EXA solving with MadNLP..."
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
        write_result(rp, Dict("exa_solve_status" => "error",
                              "exa_error" => sprint(showerror, e)))
        @error "[$m] EXA solve failed" exception=(e, catch_backtrace())
        model = nothing; GC.gc(); CUDA.reclaim()
        return
    end

    # ── SGM SOLVE RERUNS (kernels cached) ──
    run_sgm_reruns(m, rp, model)
    model = nothing; GC.gc(); CUDA.reclaim()
    @info "[$m] EXA done"
end

# ─── PEtab benchmark for the target (copied from benchmark_petab.jl run_worker) ─
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

function has_events(PEprob)
    cbs = PEprob.model_info.simulation_info.callbacks
    for (_, cb) in cbs
        cc = hasproperty(cb, :continuous_callbacks) ? getproperty(cb, :continuous_callbacks) : ()
        dc = hasproperty(cb, :discrete_callbacks)   ? getproperty(cb, :discrete_callbacks)   : ()
        (!isempty(cc) || !isempty(dc)) && return true
    end
    return false
end

function bench_petab(m)
    rp   = result_path(m)
    yaml = get_yaml(m)
    yaml === nothing && (write_result(rp, Dict("petab_compile_status" => "missing_yaml",
                                               "petab_solve_status"   => "skipped")); return)

    # ── COMPILE ──
    write_result(rp, Dict(
        "petab_compile_status" => "compiling", "petab_compile_time" => "",
        "petab_solve_status"   => "skipped",   "petab_solve_time"   => "",
        "petab_objective"      => "",           "petab_iter"        => "",
        "petab_optimum_found"  => "",           "petab_has_events"  => "",
        "petab_error"          => "",
    ))
    @info "[$m] PEtab compiling..."

    # NOTE: no in-worker warmup here — the process-level warmup on Crauste (NOT the target)
    # already paid the generic ForwardDiff/Optim JIT, so this compile timing is valid.
    PEprob = nothing
    try
        t0 = time()
        PEprob = with_hard_deadline(PETAB_COMPILE_LIMIT) do
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
        write_result(rp, Dict("petab_compile_status" => "error",
                              "petab_error" => sprint(showerror, e)))
        return
    end

    # ── SOLVE ──
    @info "[$m] PEtab solving (Optim.IPNewton)..."
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
        write_result(rp, Dict("petab_solve_status" => "error",
                              "petab_error" => sprint(showerror, e)))
    end

    # ── PEtab SGM SOLVE RERUNS (n=10 head-to-head; needs a converged first solve) ──
    if PETAB_SGM_N > 0 && get(read_result(rp), "petab_optimum_found", "") == "true"
        write_result(rp, Dict("petab_sgm_status" => "running", "petab_sgm_n" => PETAB_SGM_N))
        ptimes = Float64[]
        ok = true
        for i in 1:PETAB_SGM_N
            @info "[$m] PEtab SGM solve $i/$PETAB_SGM_N ..."
            try
                t0 = time()
                with_hard_deadline(SOLVE_LIMIT + 600.0) do
                    calibrate(PEprob, get_x(PEprob), Optim.IPNewton(); options=optim_opts())
                end
                push!(ptimes, round(time() - t0; digits=2))
            catch e
                @error "[$m] PEtab SGM solve $i failed" exception=(e, catch_backtrace())
                write_result(rp, Dict("petab_sgm_status" => "error",
                                      "petab_sgm_error"  => sprint(showerror, e)))
                ok = false; break
            end
        end
        if ok
            psgm = round(exp(sum(log, ptimes) / length(ptimes)); digits=2)
            write_result(rp, Dict("petab_sgm_status" => "ok", "petab_sgm_n" => PETAB_SGM_N,
                                  "petab_sgm_solve_time" => psgm))
            @info "[$m] PEtab SGM done: solve=$psgm s (n=$PETAB_SGM_N)"
        end
    end
end

# ─── warmups on the WARMUP_MODEL (NOT the target), one per backend ───────────────
function warmup_exa()
    yaml = get_yaml(WARMUP_MODEL); yaml === nothing && return
    @info "EXA warmup: JIT build+solve on $WARMUP_MODEL ..."
    try
        t0 = time()
        mdl, _, _, _ = build_model(yaml, t0)
        CUDA.synchronize()
        madnlp(mdl; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER, max_wall_time=250.0,
               linear_solver=MadNLPGPU.CUDSSSolver)
        mdl = nothing; GC.gc(); CUDA.reclaim()
        @info "EXA warmup done"
    catch e; @warn "EXA warmup failed" exception=(e, catch_backtrace()); end
end

function warmup_petab()
    yaml = get_yaml(WARMUP_MODEL); yaml === nothing && return
    @info "PEtab warmup: JIT calibrate on $WARMUP_MODEL ..."
    try
        wp = PEtabODEProblem(PEtabModel(yaml))
        calibrate(wp, get_x(wp), Optim.IPNewton(); options=Optim.Options(iterations=3))
        @info "PEtab warmup done"
    catch e; @warn "PEtab warmup failed" exception=(e, catch_backtrace()); end
end

function main()
    # gpu_id = the first purely-numeric arg (target name is the non-numeric arg 1); default 0
    gpu_idx = findfirst(a -> occursin(r"^\d+$", a), ARGS)
    gpu_id  = gpu_idx === nothing ? 0 : parse(Int, ARGS[gpu_idx])
    CUDA.device!(gpu_id)
    mkpath(RESULTDIR)
    @info "benchmark_bruno: target=$TARGET warmup=$WARMUP_MODEL on GPU $gpu_id ($(CUDA.name(CUDA.device())))"

    warmup_exa()
    bench_exa(TARGET)

    warmup_petab()
    bench_petab(TARGET)

    @info "benchmark_bruno complete"
end

main()
