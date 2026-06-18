# run_bruno.jl — dedicated Bruno benchmark, warmed up on Crauste.
#
# Bruno_JExpBot2016 is the SHARED JIT-warmup model for run_examodels.jl and run_petab.jl, so it is
# excluded from both of their timed runs (a model benchmarked by the script that warms on it gets an
# invalid pre-warmed compile time). This script benchmarks the TARGET, warming on a DIFFERENT model.
#
# Backend (BENCH_BACKEND, like run_examodels.jl): "gpu" (default) -> exagpu_*, "cpu" -> exacpu_*.
# The PEtab side (petab_*) is backend-independent, so it runs only on the GPU pass.
#   julia --project=examples -t 1 examples/Benchmarks/run_bruno.jl Bruno_JExpBot2016 [gpu_id]
#   BENCH_BACKEND=cpu julia --project=examples -t 1 examples/Benchmarks/run_bruno.jl Bruno_JExpBot2016

using ExaModelsPEtab, PEtab, CUDA, MadNLP, MadNLPGPU, CUDSS, ExaModels, Optim

# ─── CONFIGURABLE SETTINGS (single source of truth = options.jl) ──────────────────
const MODELDIR  = joinpath(@__DIR__, "..", "Benchmark-Models")
const RESULTDIR = joinpath(@__DIR__, "results")
include(joinpath(@__DIR__, "options.jl"))

const TARGET       = (length(ARGS) >= 1 && !occursin(r"^\d+$", ARGS[1])) ? ARGS[1] : "Bruno_JExpBot2016"
const WARMUP_MODEL = TARGET == "Crauste_CellSystems2017" ? "Bruno_JExpBot2016" : "Crauste_CellSystems2017"

const K             = BENCH_K
const TOL           = BENCH_TOL
const COMPILE_LIMIT = BENCH_COMPILE_LIMIT
const PETAB_COMPILE_LIMIT = BENCH_PETAB_COMPILE_LIMIT
const SOLVE_LIMIT   = BENCH_SOLVE_LIMIT
const MAX_ITER      = BENCH_MAX_ITER
const N_SGM_RERUNS  = BENCH_SGM_N
const SGM_SHIFT     = BENCH_SGM_SHIFT
const ACCEPT_TOL    = BENCH_ACCEPT_TOL
const ACCEPT_ITER   = BENCH_ACCEPT_ITER
const PETAB_SGM_N   = BENCH_SGM_N

const BACKEND = lowercase(get(ENV, "BENCH_BACKEND", "gpu"))
const IS_GPU  = BACKEND != "cpu"
const PFX     = IS_GPU ? "exagpu_" : "exacpu_"
const KKT_OPTS = (kkt_system = MadNLP.SparseCondensedKKTSystem,
                  equality_treatment = MadNLP.RelaxEquality,
                  fixed_variable_treatment = MadNLP.RelaxBound)
solve_madnlp(model) = IS_GPU ?
    madnlp(model; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
           max_wall_time=SOLVE_LIMIT, linear_solver=MadNLPGPU.CUDSSSolver, KKT_OPTS...) :
    madnlp(model; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
           max_wall_time=SOLVE_LIMIT, KKT_OPTS...)
gpu_reclaim() = IS_GPU && (GC.gc(); CUDA.reclaim())
# ────────────────────────────────────────────────────────────────────────────────

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

# ─── ExaModels build (backend-parametrized; returns PEtabODEProblem) ──────────────
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

function run_sgm_reruns(m, rp, model)
    write_result(rp, Dict(PFX*"sgm_status" => "running", PFX*"sgm_n" => N_SGM_RERUNS))
    solve_times = Float64[]
    for i in 1:N_SGM_RERUNS
        @info "[$m] SGM solve $i/$N_SGM_RERUNS ..."
        try
            t0 = time(); solve_madnlp(model); push!(solve_times, round(time() - t0; digits=2))
        catch e
            @error "[$m] SGM solve $i failed" exception=(e, catch_backtrace())
            write_result(rp, Dict(PFX*"sgm_status" => "error", PFX*"sgm_error" => sprint(showerror, e))); return
        end
    end
    sgm_solve = shifted_geomean(solve_times, SGM_SHIFT)
    write_result(rp, Dict(PFX*"sgm_status" => "ok", PFX*"sgm_n" => N_SGM_RERUNS,
                          PFX*"solve_times" => join(solve_times, ","), PFX*"sgm_solve_time" => sgm_solve))
    @info "[$m] SGM done: solve=$sgm_solve s (n=$N_SGM_RERUNS)"
end

# ─── ExaModels benchmark for the target (compile + first solve + SGM) ───────────
function bench_exa(m)
    rp   = result_path(m)
    yaml = get_yaml(m)
    yaml === nothing && (write_result(rp, Dict(PFX*"compile_status" => "missing_yaml", PFX*"solve_status" => "skipped")); return)
    model = nothing

    write_result(rp, Dict(
        PFX*"compile_status" => "compiling", PFX*"compile_time" => "", PFX*"presolve_time" => "",
        PFX*"solve_status"   => "skipped",   PFX*"solve_time"   => "", PFX*"term_status"   => "",
        PFX*"objective"      => "",          PFX*"petab_obj"    => "", PFX*"iter" => "",
        PFX*"nvar"           => "",          PFX*"ncon"         => "", PFX*"error" => "",
        PFX*"constr_viol"    => "",
    ))
    @info "[$m] EXA compiling on $BACKEND (K=$K)..."
    local PEprob
    try
        t0 = time()
        mdl, PEprob, nvar, ncon, t_phase1 = with_hard_deadline(COMPILE_LIMIT) do; build_model(yaml, t0); end
        model = mdl
        write_result(rp, Dict(
            PFX*"compile_status" => "ok", PFX*"compile_time" => round(time() - t0; digits=2),
            PFX*"presolve_time"  => round(t_phase1; digits=2), PFX*"K" => K, PFX*"nvar" => nvar, PFX*"ncon" => ncon,
        ))
    catch e
        write_result(rp, Dict(PFX*"compile_status" => "error", PFX*"error" => sprint(showerror, e)))
        @error "[$m] EXA compile failed" exception=(e, catch_backtrace()); return
    end

    @info "[$m] EXA solving with MadNLP/$BACKEND..."
    write_result(rp, Dict(PFX*"solve_status" => "solving"))
    try
        t0 = time()
        res = with_hard_deadline(SOLVE_LIMIT + 3600.0) do; solve_madnlp(model); end
        write_result(rp, Dict(
            PFX*"solve_status" => "ok", PFX*"solve_time" => round(time() - t0; digits=2),
            PFX*"term_status"  => string(res.status), PFX*"objective" => res.objective,
            PFX*"petab_obj"    => petab_obj_at_exa(PEprob, res), PFX*"iter" => res.iter,
            PFX*"constr_viol"  => max_constr_viol(model, res.solution),
        ))
    catch e
        write_result(rp, Dict(PFX*"solve_status" => "error", PFX*"error" => sprint(showerror, e)))
        @error "[$m] EXA solve failed" exception=(e, catch_backtrace()); model = nothing; gpu_reclaim(); return
    end

    if uppercase(get(read_result(rp), PFX*"term_status", "")) in ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL")
        run_sgm_reruns(m, rp, model)
    else
        write_result(rp, Dict(PFX*"sgm_status" => "skipped"))
    end
    model = nothing; gpu_reclaim()
    @info "[$m] EXA done"
end

# ─── PEtab benchmark for the target (backend-independent; runs on the GPU pass) ──
optim_opts() = Optim.Options(iterations=MAX_ITER, time_limit=SOLVE_LIMIT, g_tol=TOL,
    f_reltol=BENCH_PETAB_F_RELTOL, allow_f_increases=true, successive_f_tol=BENCH_PETAB_SUCCESSIVE_FTOL,
    show_trace=false, x_abstol=BENCH_PETAB_X_ABSTOL)

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
    yaml === nothing && (write_result(rp, Dict("petab_compile_status" => "missing_yaml", "petab_solve_status" => "skipped")); return)

    write_result(rp, Dict(
        "petab_compile_status" => "compiling", "petab_compile_time" => "",
        "petab_solve_status"   => "skipped",   "petab_solve_time"   => "",
        "petab_objective"      => "",           "petab_iter"        => "",
        "petab_optimum_found"  => "",           "petab_has_events"  => "", "petab_error" => "",
    ))
    @info "[$m] PEtab compiling..."
    PEprob = nothing
    try
        t0 = time()
        PEprob = with_hard_deadline(PETAB_COMPILE_LIMIT) do
            p  = PEtabODEProblem(PEtabModel(yaml)); x0 = get_x(p); np = length(x0)
            p.nllh(x0); p.grad!(zeros(np), x0); p.hess!(zeros(np, np), x0); p
        end
        write_result(rp, Dict("petab_compile_status" => "ok", "petab_compile_time" => round(time() - t0; digits=2),
                              "petab_has_events" => has_events(PEprob)))
    catch e
        write_result(rp, Dict("petab_compile_status" => "error", "petab_error" => sprint(showerror, e))); return
    end

    @info "[$m] PEtab solving (Optim.IPNewton)..."
    write_result(rp, Dict("petab_solve_status" => "solving"))
    try
        t0 = time()
        res = with_hard_deadline(SOLVE_LIMIT + 600.0) do; calibrate(PEprob, get_x(PEprob), Optim.IPNewton(); options=optim_opts()); end
        gconv = ""; gres = ""; fconv = ""; xconv = ""
        try; o = res.original
            gconv = string(Optim.g_converged(o)); gres = string(Optim.g_residual(o))
            fconv = string(Optim.f_converged(o)); xconv = string(Optim.x_converged(o)); catch; end
        write_result(rp, Dict(
            "petab_solve_status"  => (res.converged === :Optimisation_failed || !isfinite(res.fmin)) ? "error" : "ok",
            "petab_solve_time"    => round(time() - t0; digits=2), "petab_objective" => res.fmin,
            "petab_iter"          => res.niterations, "petab_optimum_found" => string(res.converged === true),
            "petab_gconverged"    => gconv, "petab_gresidual" => gres,
            "petab_fconverged"    => fconv, "petab_xconverged" => xconv,
        ))
    catch e
        write_result(rp, Dict("petab_solve_status" => "error", "petab_error" => sprint(showerror, e)))
    end

    if PETAB_SGM_N > 0 && get(read_result(rp), "petab_optimum_found", "") == "true"
        write_result(rp, Dict("petab_sgm_status" => "running", "petab_sgm_n" => PETAB_SGM_N))
        ptimes = Float64[]; ok = true
        for i in 1:PETAB_SGM_N
            @info "[$m] PEtab SGM solve $i/$PETAB_SGM_N ..."
            try
                t0 = time()
                with_hard_deadline(SOLVE_LIMIT + 600.0) do; calibrate(PEprob, get_x(PEprob), Optim.IPNewton(); options=optim_opts()); end
                push!(ptimes, round(time() - t0; digits=2))
            catch e
                @error "[$m] PEtab SGM solve $i failed" exception=(e, catch_backtrace())
                write_result(rp, Dict("petab_sgm_status" => "error", "petab_sgm_error" => sprint(showerror, e))); ok = false; break
            end
        end
        if ok
            psgm = shifted_geomean(ptimes, SGM_SHIFT)
            write_result(rp, Dict("petab_sgm_status" => "ok", "petab_sgm_n" => PETAB_SGM_N,
                                  "petab_solve_times" => join(ptimes, ","), "petab_sgm_solve_time" => psgm))
            @info "[$m] PEtab SGM done: solve=$psgm s (n=$PETAB_SGM_N)"
        end
    end
end

function warmup_exa()
    yaml = get_yaml(WARMUP_MODEL); yaml === nothing && return
    @info "EXA warmup: JIT build+solve on $WARMUP_MODEL ($BACKEND) ..."
    try
        t0 = time(); mdl, _, _, _, _ = build_model(yaml, t0); IS_GPU && CUDA.synchronize()
        madnlp(mdl; tol=TOL, acceptable_tol=ACCEPT_TOL, acceptable_iter=ACCEPT_ITER, max_iter=MAX_ITER,
               max_wall_time=250.0, KKT_OPTS..., (IS_GPU ? (; linear_solver=MadNLPGPU.CUDSSSolver) : (;))...)
        mdl = nothing; gpu_reclaim(); @info "EXA warmup done"
    catch e; @warn "EXA warmup failed" exception=(e, catch_backtrace()); end
end
function warmup_petab()
    yaml = get_yaml(WARMUP_MODEL); yaml === nothing && return
    @info "PEtab warmup: JIT calibrate on $WARMUP_MODEL ..."
    try
        wp = PEtabODEProblem(PEtabModel(yaml)); calibrate(wp, get_x(wp), Optim.IPNewton(); options=Optim.Options(iterations=3))
        @info "PEtab warmup done"
    catch e; @warn "PEtab warmup failed" exception=(e, catch_backtrace()); end
end

function main()
    gpu_idx = findfirst(a -> occursin(r"^\d+$", a), ARGS)
    gpu_id  = gpu_idx === nothing ? 0 : parse(Int, ARGS[gpu_idx])
    IS_GPU && CUDA.device!(gpu_id)
    mkpath(RESULTDIR)
    @info "run_bruno: target=$TARGET warmup=$WARMUP_MODEL backend=$BACKEND prefix=$PFX"

    # BENCH_PETAB_ONLY=1 re-runs only the petab_ (PEtab) side, leaving exa results untouched.
    petab_only = get(ENV, "BENCH_PETAB_ONLY", "0") == "1"

    if !petab_only
        warmup_exa()
        bench_exa(TARGET)
    end

    if IS_GPU || petab_only   # PEtab is backend-independent — run it only on the GPU pass (or petab-only)
        warmup_petab()
        bench_petab(TARGET)
    end
    @info "run_bruno complete"
end

main()
