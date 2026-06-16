# run_examodels.jl — ExaModelsPEtab + MadNLP benchmark (GPU or CPU)
#
# Builds petab_examodel and solves it with MadNLP using the LiftedKKT (condensed-space) regime —
# the same SparseCondensedKKTSystem + RelaxEquality config on both backends; only the linear solver
# differs (GPU: CUDSS, CPU: MadNLP's default for the condensed system). Results are written to
# Benchmarks/results/{Model}_results.txt under a backend-specific prefix so the two backends and the
# PEtab results coexist in one file:
#   BENCH_BACKEND=gpu (default) -> exagpu_*
#   BENCH_BACKEND=cpu           -> exacpu_*
#
# For each solve we also store <prefix>petab_obj := PEtab's own objective evaluated at the ExaModels
# optimal parameters (res.solution[1:Np], which live on PEtab's estimation scale) — the fair,
# same-objective value behind results.jl's GAP(%) column.
#
# Compilation timing splits into Phase 1 (PEtab setup + ODE presolve) and Phase 2 (ExaModels build);
# both are stored, and after a converged first solve N_SGM_RERUNS warm reruns give the SGM solve time.
# The run is resumable (terminal results are skipped); wrap with run_examodels.sh for watchdog restarts.
#
# Usage (GPU, strided one instance per GPU):
#   julia --project=examples -t 1 examples/Benchmarks/run_examodels.jl <gpu_id> <num_instances> <instance_idx>
# CPU (gpu_id arg ignored; run several instances to parallelize across cores):
#   BENCH_BACKEND=cpu julia --project=examples -t 1 examples/Benchmarks/run_examodels.jl 0 <num_instances> <instance_idx>

using ExaModelsPEtab, PEtab, CUDA, MadNLP, MadNLPGPU, CUDSS, ExaModels

# ─── CONFIGURABLE SETTINGS (single source of truth = options.jl) ────────────────
const MODELDIR  = joinpath(@__DIR__, "..", "Benchmark-Models")
const RESULTDIR = joinpath(@__DIR__, "results")
include(joinpath(@__DIR__, "options.jl"))

const K             = BENCH_K
const TOL           = BENCH_TOL
const COMPILE_LIMIT = BENCH_COMPILE_LIMIT
const SOLVE_LIMIT   = BENCH_SOLVE_LIMIT
const MAX_ITER      = BENCH_MAX_ITER
const N_SGM_RERUNS  = BENCH_SGM_N
const SGM_SHIFT     = BENCH_SGM_SHIFT
const ACCEPT_TOL    = BENCH_ACCEPT_TOL
const ACCEPT_ITER   = BENCH_ACCEPT_ITER
const WARMUP_MODEL  = BENCH_WARMUP_MODEL

# Backend: "gpu" (CUDA + CUDSS, default) or "cpu" (no CUDA). PFX prefixes every result key so the
# GPU and CPU runs write into the same {Model}_results.txt without clobbering each other.
const BACKEND = lowercase(get(ENV, "BENCH_BACKEND", "gpu"))
const IS_GPU  = BACKEND != "cpu"
const PFX     = IS_GPU ? "exagpu_" : "exacpu_"

# LiftedKKT (condensed-space) MadNLP regime — IDENTICAL on both backends (GPU just adds CUDSS).
const KKT_OPTS = (kkt_system = MadNLP.SparseCondensedKKTSystem,
                  equality_treatment = MadNLP.RelaxEquality,
                  fixed_variable_treatment = MadNLP.RelaxBound)
solve_madnlp(model) = IS_GPU ?
    madnlp(model; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
           max_wall_time=SOLVE_LIMIT, linear_solver=MadNLPGPU.CUDSSSolver, KKT_OPTS...) :
    madnlp(model; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
           max_wall_time=SOLVE_LIMIT, KKT_OPTS...)
gpu_reclaim() = IS_GPU && (GC.gc(); CUDA.reclaim())
# ──────────────────────────────────────────────────────────────────────────────

# Timed in-loop set = EXA_RERUN_INLOOP (options.jl); override via BENCH_SUBSET (comma-separated).
const ALL_MODELS = haskey(ENV, "BENCH_SUBSET") ?
    String.(split(ENV["BENCH_SUBSET"], ',')) : EXA_RERUN_INLOOP

get_yaml(m) = begin
    d = joinpath(MODELDIR, m); isdir(d) || return nothing
    fs = filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))
    isempty(fs) ? nothing : joinpath(d, first(fs))
end
result_path(m) = joinpath(RESULTDIR, "$(m)_results.txt")

function read_result(path)
    d = Dict{String,String}(); isfile(path) || return d
    for line in eachline(path)
        i = findfirst('=', line); i === nothing && continue
        d[line[1:i-1]] = line[i+1:end]
    end
    return d
end

# Read-modify-write: preserves keys from the other backend / PEtab in the same file.
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

# Resumption: is this backend done with model m? (reads PFX-prefixed keys)
function exa_finished(m)
    d = read_result(result_path(m))
    cs  = get(d, PFX * "compile_status", "")
    ss  = get(d, PFX * "solve_status",   "")
    sgm = get(d, PFX * "sgm_status",     "")
    cs in ("timeout", "error", "missing_yaml", "skipped") && return true
    cs != "ok" && return false
    ss in ("timeout", "error") && return true
    ss != "ok" && return false
    return sgm in ("ok", "error", "skipped")
end

# PEtab's own objective at the ExaModels optimum: res.solution[1:Np] are the estimated parameters on
# PEtab's estimation scale (warm-started from get_x), so PEprob.nllh of them is directly comparable
# to petab_objective (the value calibrate minimized). Returns NaN if the eval fails.
function petab_obj_at_exa(PEprob, res)
    Np = PEprob.nparameters_estimate
    xstar = Array(res.solution)[1:Np]
    try; return PEprob.nllh(xstar); catch; return NaN; end
end

# inf-norm (max) constraint violation max(lcon - c(x), c(x) - ucon, 0) at point x; 0 if unconstrained.
function max_constr_viol(model, x)
    model.meta.ncon == 0 && return 0.0
    c = similar(x, model.meta.ncon); ExaModels.cons!(model, x, c)
    c = Array(c); lc = Array(model.meta.lcon); uc = Array(model.meta.ucon)
    maximum(max.(lc .- c, c .- uc, 0.0))
end

# ─── build one ExaModel (compile phases 1+2); returns the PEtabODEProblem too ──
function build_model(yaml, t_origin)
    PEmodel = PEtab.PEtabModel(yaml)
    PEprob  = PEtab.PEtabODEProblem(PEmodel)
    backend = IS_GPU ? CUDA.CUDABackend() : nothing
    c = ExaModels.ExaCore(; backend, concrete=Val(true))

    if ExaModelsPEtab._is_steady_state(PEmodel)
        c, PEinfo = ExaModelsPEtab._create_variables_ss(c, PEmodel, PEprob)
        t_phase1 = time() - t_origin
        c = ExaModelsPEtab._create_constraints_ss(c, PEmodel, PEprob, PEinfo)
        c, y0, sigma0 = ExaModelsPEtab._create_objective_ss(c, PEmodel, PEprob, PEinfo)
    else
        c, PEinfo = ExaModelsPEtab._create_variables(c, PEmodel, PEprob, K)
        t_phase1 = time() - t_origin
        c = ExaModelsPEtab._create_collocation(c, PEmodel, PEprob, PEinfo)
        c = ExaModelsPEtab._create_continuity(c, PEmodel, PEprob, PEinfo)
        c, y0, sigma0 = ExaModelsPEtab._create_objective(c, PEmodel, PEprob, PEinfo)
    end
    mdl = ExaModels.ExaModel(c)
    ExaModels.set_start!(mdl, c.y, y0)
    ExaModels.set_start!(mdl, c.sigma, sigma0)
    IS_GPU && CUDA.synchronize()
    return mdl, PEprob, mdl.meta.nvar, mdl.meta.ncon, t_phase1
end

# ─── SGM warm reruns (timing only) ──────────────────────────────────────────────
function run_sgm_reruns(m, rp, model)
    write_result(rp, Dict(PFX*"sgm_status" => "running", PFX*"sgm_n" => N_SGM_RERUNS))
    solve_times = Float64[]
    for i in 1:N_SGM_RERUNS
        @info "[$m] SGM solve $i/$N_SGM_RERUNS ..."
        try
            t0 = time()
            with_hard_deadline(SOLVE_LIMIT + 3600.0) do; solve_madnlp(model); end
            push!(solve_times, round(time() - t0; digits=2))
        catch e
            @error "[$m] SGM solve $i failed" exception=(e, catch_backtrace())
            write_result(rp, Dict(PFX*"sgm_status" => "error", PFX*"sgm_error" => sprint(showerror, e)))
            return
        end
    end
    sgm_solve = shifted_geomean(solve_times, SGM_SHIFT)
    write_result(rp, Dict(PFX*"sgm_status" => "ok", PFX*"sgm_n" => N_SGM_RERUNS,
                          PFX*"solve_times" => join(solve_times, ","), PFX*"sgm_solve_time" => sgm_solve))
    @info "[$m] SGM done: solve=$sgm_solve s (n=$N_SGM_RERUNS)"
end

# ─── main per-model benchmark ─────────────────────────────────────────────────
function bench_one(m)
    rp   = result_path(m)
    yaml = get_yaml(m)
    d = read_result(rp)
    first_run_done = get(d, PFX*"compile_status", "") == "ok" && get(d, PFX*"solve_status", "") == "ok"
    model = nothing

    if !first_run_done
        if yaml === nothing
            write_result(rp, Dict(PFX*"compile_status" => "missing_yaml", PFX*"solve_status" => "skipped"))
            return
        end

        # ── COMPILE ──────────────────────────────────────────────────────────
        write_result(rp, Dict(
            PFX*"compile_status" => "compiling", PFX*"compile_time" => "", PFX*"presolve_time" => "",
            PFX*"solve_status"   => "skipped",   PFX*"solve_time"   => "", PFX*"term_status"   => "",
            PFX*"objective"      => "",          PFX*"petab_obj"    => "", PFX*"iter" => "",
            PFX*"nvar"           => "",          PFX*"ncon"         => "", PFX*"error" => "",
            PFX*"constr_viol"    => "",
        ))
        @info "[$m] compiling on $BACKEND (K=$K, compile_limit=$(COMPILE_LIMIT)s)..."
        local PEprob
        try
            t0 = time()
            mdl, PEprob, nvar, ncon, t_phase1 = with_hard_deadline(COMPILE_LIMIT) do
                build_model(yaml, t0)
            end
            model = mdl
            write_result(rp, Dict(
                PFX*"compile_status" => "ok",
                PFX*"compile_time"   => round(time() - t0; digits=2),
                PFX*"presolve_time"  => round(t_phase1;    digits=2),
                PFX*"K" => K, PFX*"nvar" => nvar, PFX*"ncon" => ncon,
            ))
        catch e
            write_result(rp, Dict(PFX*"compile_status" => "error", PFX*"compile_time" => "", PFX*"error" => sprint(showerror, e)))
            @error "[$m] compile failed" exception=(e, catch_backtrace())
            return
        end

        # ── SOLVE (first run; GPU includes one-time kernel JIT) ───────────────
        @info "[$m] solving with MadNLP/$BACKEND (max_wall_time=$(SOLVE_LIMIT)s)..."
        write_result(rp, Dict(PFX*"solve_status" => "solving"))
        try
            t0 = time()
            res = with_hard_deadline(SOLVE_LIMIT + 3600.0) do; solve_madnlp(model); end
            write_result(rp, Dict(
                PFX*"solve_status" => "ok",
                PFX*"solve_time"   => round(time() - t0; digits=2),
                PFX*"term_status"  => string(res.status),
                PFX*"objective"    => res.objective,
                PFX*"petab_obj"    => petab_obj_at_exa(PEprob, res),
                PFX*"iter"         => res.iter,
                PFX*"constr_viol"  => max_constr_viol(model, res.solution),
            ))
        catch e
            write_result(rp, Dict(PFX*"solve_status" => "error", PFX*"error" => sprint(showerror, e)))
            @error "[$m] solve failed" exception=(e, catch_backtrace())
            model = nothing; gpu_reclaim()
            return
        end
    end

    # ── SGM SOLVE RERUNS (only after a converged first solve) ──────────────────
    d2 = read_result(rp)
    if get(d2, PFX*"compile_status", "") == "ok" && get(d2, PFX*"solve_status", "") == "ok" &&
       get(d2, PFX*"sgm_status", "") ∉ ("ok", "error", "skipped")
        if !(uppercase(get(d2, PFX*"term_status", "")) in ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL"))
            write_result(rp, Dict(PFX*"sgm_status" => "skipped",
                PFX*"sgm_error" => "first solve term=$(get(d2, PFX*"term_status", ""))∉{SUCCEEDED,ACCEPTABLE}; SGM skipped"))
            @info "[$m] SGM skipped (solve term=$(get(d2, PFX*"term_status", "")))"
        else
            if model === nothing  # resuming: rebuild silently (time not recorded), prime once
                yaml === nothing && return
                @info "[$m] rebuilding for SGM (resume)..."
                try
                    t0 = time()
                    model, _, _, _, _ = with_hard_deadline(COMPILE_LIMIT) do; build_model(yaml, t0); end
                    solve_madnlp(model)
                catch e
                    @error "[$m] SGM rebuild failed" exception=(e, catch_backtrace())
                    write_result(rp, Dict(PFX*"sgm_status" => "error", PFX*"sgm_error" => sprint(showerror, e)))
                    model = nothing; gpu_reclaim(); return
                end
            end
            run_sgm_reruns(m, rp, model)
        end
    end

    model = nothing; gpu_reclaim()
    @info "[$m] done"
end

function warmup()
    yaml = get_yaml(WARMUP_MODEL); yaml === nothing && return
    @info "warmup: JIT build+solve on $WARMUP_MODEL ($BACKEND) ..."
    try
        t0 = time()
        mdl, _, _, _, _ = build_model(yaml, t0)
        IS_GPU && CUDA.synchronize()
        madnlp(mdl; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
               max_wall_time=250.0, KKT_OPTS..., (IS_GPU ? (; linear_solver=MadNLPGPU.CUDSSSolver) : (;))...)
        mdl = nothing; gpu_reclaim()
        @info "warmup done"
    catch e; @warn "warmup failed" exception=(e, catch_backtrace()); end
end

function main()
    gpu_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
    ninst  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    idx    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0
    if IS_GPU
        CUDA.device!(gpu_id)
        @info "instance $idx/$ninst on GPU $gpu_id ($(CUDA.name(CUDA.device())))  prefix=$PFX"
    else
        @info "instance $idx/$ninst on CPU  prefix=$PFX"
    end
    mkpath(RESULTDIR)

    # Convert in-progress sentinels from a prior killed run to terminal states (this backend only)
    for m in ALL_MODELS
        d = read_result(result_path(m))
        if get(d, PFX*"compile_status", "") == "compiling"
            write_result(result_path(m), Dict(PFX*"compile_status" => "timeout", PFX*"compile_time" => string(COMPILE_LIMIT)))
        elseif get(d, PFX*"compile_status", "") == "ok" && get(d, PFX*"solve_status", "") == "solving"
            write_result(result_path(m), Dict(PFX*"solve_status" => "timeout"))
        elseif get(d, PFX*"sgm_status", "") == "running"
            write_result(result_path(m), Dict(PFX*"sgm_status" => "interrupted"))
        end
    end

    mine = [m for (i, m) in enumerate(ALL_MODELS) if (i - 1) % ninst == idx]
    todo = filter(!exa_finished, mine)
    @info "instance $idx: $(length(mine)) models assigned, $(length(todo)) remaining"
    isempty(todo) && (@info "nothing to do"; return)

    warmup()
    for (i, m) in enumerate(todo)
        println("\n" * "="^60); println("[$idx][$i/$(length(todo))] $m ($BACKEND)"); println("="^60)
        bench_one(m)
    end
    @info "instance $idx complete"
end

main()
