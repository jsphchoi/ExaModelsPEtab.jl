# benchmark_crauste.jl — one-off ExaModelsPEtab + MadNLP GPU benchmark for Crauste.
#
# Crauste is the JIT warmup model in benchmark_examodels.jl (excluded from ALL_MODELS),
# so it never gets a real exa_* timing. This driver gives Crauste a proper benchmark by
# warming the generic JIT on Bruno first, then running the identical compile→solve→SGM
# sequence used for every other model, writing the same exa_* keys to its result file.
#
# Usage (run on an uncontended GPU; the canonical 19-model run uses GPU 1):
#   julia --project=. -t 1 examples/Benchmarks/benchmark_crauste.jl <gpu_id>

using ExaModelsPEtab, PEtab, CUDA, MadNLPGPU, CUDSS, ExaModels

# ─── CONFIGURABLE SETTINGS (identical to benchmark_examodels.jl) ───────────────
const K             = 6               # collocation points per mesh interval
const TOL           = 1e-6            # MadNLP solver tolerance
const COMPILE_LIMIT = 14400.0         # hard compile deadline [s] (4 hr)
const SOLVE_LIMIT   = 86400.0         # MadNLP max_wall_time [s] (24 hr)
const MAX_ITER      = 100_000_000     # large so wall time is always the bottleneck
const N_SGM_RERUNS  = 3               # rerun count for geometric mean timing
const WARMUP_MODEL  = "Bruno_JExpBot2016"        # pre-warms generic JIT only
const TARGET_MODEL  = "Crauste_CellSystems2017"  # the model we actually benchmark
# ──────────────────────────────────────────────────────────────────────────────

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

# ─── build one ExaModel (compile phases 1+2) ──────────────────────────────────
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

# ─── SGM solve reruns ─────────────────────────────────────────────────────────
function run_sgm_reruns(m, rp, model)
    write_result(rp, Dict("exa_sgm_status" => "running", "exa_sgm_n" => N_SGM_RERUNS))
    solve_times = Float64[]
    for i in 1:N_SGM_RERUNS
        @info "[$m] SGM solve $i/$N_SGM_RERUNS ..."
        try
            t0 = time()
            with_hard_deadline(SOLVE_LIMIT + 3600.0) do
                madnlp(model; tol=TOL, max_iter=MAX_ITER,
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

# ─── per-model benchmark (compile + first solve + SGM) ─────────────────────────
function bench_one(m)
    rp   = result_path(m)
    yaml = get_yaml(m)
    yaml === nothing && (@error "[$m] missing yaml"; return)

    model = nothing

    # ── COMPILE ───────────────────────────────────────────────────────────────
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
            "exa_nvar"           => nvar,
            "exa_ncon"           => ncon,
        ))
    catch e
        write_result(rp, Dict(
            "exa_compile_status" => "error", "exa_compile_time" => "",
            "exa_error"          => sprint(showerror, e),
        ))
        @error "[$m] compile failed" exception=(e, catch_backtrace())
        return
    end

    # ── SOLVE (first run — includes GPU kernel compilation) ───────────────────
    @info "[$m] solving with MadNLP (max_wall_time=$(SOLVE_LIMIT)s)..."
    write_result(rp, Dict("exa_solve_status" => "solving"))
    try
        t0 = time()
        res = with_hard_deadline(SOLVE_LIMIT + 3600.0) do
            madnlp(model; tol=TOL, max_iter=MAX_ITER,
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
                              "exa_error"        => sprint(showerror, e)))
        @error "[$m] solve failed" exception=(e, catch_backtrace())
        model = nothing; GC.gc(); CUDA.reclaim()
        return
    end

    # ── SGM SOLVE RERUNS ──────────────────────────────────────────────────────
    run_sgm_reruns(m, rp, model)

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
        madnlp(mdl; tol=TOL, max_iter=MAX_ITER, max_wall_time=250.0,
               linear_solver=MadNLPGPU.CUDSSSolver)
        mdl = nothing; GC.gc(); CUDA.reclaim()
        @info "warmup done"
    catch e; @warn "warmup failed" exception=(e, catch_backtrace()); end
end

function main()
    gpu_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
    CUDA.device!(gpu_id)
    @info "Crauste benchmark on GPU $gpu_id ($(CUDA.name(CUDA.device()))); warmup=$WARMUP_MODEL"
    mkpath(RESULTDIR)
    warmup()
    println("\n" * "="^60); println("[Crauste benchmark] $TARGET_MODEL"); println("="^60)
    bench_one(TARGET_MODEL)
    @info "Crauste benchmark complete"
end

main()
