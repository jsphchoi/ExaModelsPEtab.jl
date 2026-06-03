# bench2_petab.jl  —  PHASE 1 worker (one model per process; run many in parallel).
#
# Solves a PEtab problem with Optim.IPNewton() on the default PEtabODEProblem (which
# auto-selects the size-appropriate Hessian: exact ForwardDiff for small models,
# Gauss-Newton for medium) — PEtab.jl's recommended optimizer for the small/medium models
# that dominate this collection. Up to a 1-hour wall limit. Records compile/solve
# status+time, objective, Optim convergence flag, iters, and whether the model has event
# callbacks (discontinuities the ExaModels collocation cannot represent).
#
# "Pass" (qualifies for the ExaModels phase) = the optimizer RAN without erroring.
#
# Usage:  julia --project=. -t 1 examples/bench2_petab.jl <model_name>

using PEtab, Optim

const PETAB_MAXIT  = 100_000            # large so wall-time is the bottleneck
const PETAB_LIMIT  = parse(Float64, get(ENV, "PETAB_LIMIT", "3600.0"))    # 1 h solve wall limit
const COMPILE_LIMIT = parse(Float64, get(ENV, "PETAB_COMPILE_LIMIT", "1800.0"))  # 30 min build limit

# Optim options: PEtab's default tolerances (g_tol=1e-6, f_reltol=1e-8, allow_f_increases,
# successive_f_tol=3) plus a large iteration budget and the wall-time cap.
optim_opts(timelimit) = Optim.Options(iterations = PETAB_MAXIT, time_limit = timelimit,
    g_tol = 1.0e-6, f_reltol = 1.0e-8, allow_f_increases = true, successive_f_tol = 3,
    show_trace = false, x_abstol = 0.0)
const WARMUP_MODEL = "Bruno_JExpBot2016"

const MODELDIR  = joinpath(@__DIR__, "Benchmark-Models")
const RESULTDIR = joinpath(@__DIR__, "results2")

get_yaml(m) = begin
    d = joinpath(MODELDIR, m); isdir(d) || return nothing
    fs = filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))
    isempty(fs) ? nothing : joinpath(d, first(fs))
end
petab_path(m) = joinpath(RESULTDIR, "$(m).petab.txt")

function write_result(path, d)
    open(path, "w") do io
        for k in sort(collect(keys(d)))
            println(io, "$k=", replace(string(d[k]), '\n'=>' ', '\r'=>' '))
        end
        flush(io)
    end
end

# external-process watchdog (no Julia thread → no interference); pre-write pessimistic status
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

function run_worker(m)
    mkpath(RESULTDIR)
    rp = petab_path(m)
    R = Dict{String,Any}("model"=>m, "compile_status"=>"compiling", "compile_time"=>"",
        "solve_status"=>"skipped", "solve_time"=>"", "objective"=>"", "opt_status"=>"",
        "iters"=>"", "optimum_found"=>"", "has_events"=>"", "error"=>"")
    write_result(rp, R)

    yaml = get_yaml(m)
    if yaml === nothing
        R["compile_status"]="missing_yaml"; write_result(rp, R); return
    end

    # warm up the PEtab + Optim JIT on a small model so timings reflect real work
    try
        wy = get_yaml(WARMUP_MODEL)
        if wy !== nothing
            wprob = PEtabODEProblem(PEtabModel(wy))
            calibrate(wprob, get_x(wprob), Optim.IPNewton();
                      options = Optim.Options(iterations = 3))
        end
    catch; end

    # COMPILE (includes triggering the model-specific derivative JIT — notably the
    # ForwardDiff Hessian — so the solve phase is JIT-free and Optim's time_limit governs it)
    PEprob = nothing
    try
        t0 = time()
        PEprob = with_hard_deadline(COMPILE_LIMIT) do
            p = PEtabODEProblem(PEtabModel(yaml))
            x0 = get_x(p); np = length(x0)
            p.nllh(x0); p.grad!(zeros(np), x0); p.hess!(zeros(np, np), x0)
            p
        end
        R["compile_time"]   = round(time()-t0; digits=2)
        R["compile_status"] = "ok"
        R["has_events"]     = has_events(PEprob)
    catch e
        R["compile_status"]="error"; R["error"]=sprint(showerror, e)
        write_result(rp, R); return
    end
    write_result(rp, R)

    # SOLVE (Optim.IPNewton)
    R["solve_status"]="solving"; write_result(rp, R)
    try
        t0 = time()
        # generous backstop: Optim only checks the wall limit between iterations, so a slow
        # final (Hessian) iteration can overrun time_limit; don't kill it prematurely.
        res = with_hard_deadline(PETAB_LIMIT + 600.0) do
            calibrate(PEprob, get_x(PEprob), Optim.IPNewton(); options = optim_opts(PETAB_LIMIT))
        end
        R["solve_time"]    = round(time()-t0; digits=2)
        R["objective"]     = res.fmin
        R["opt_status"]    = string(res.converged)
        R["iters"]         = res.niterations
        # Optim.converged returns a Bool; an optimum is "found" when it is true
        R["optimum_found"] = res.converged === true
        # "ran" (qualifies) iff the optimizer did not throw and produced a finite objective
        R["solve_status"]  = (res.converged === :Optmisation_failed || !isfinite(res.fmin)) ?
                             "error" : "ok"
    catch e
        R["solve_status"]="error"; R["error"]=sprint(showerror, e)
    end
    write_result(rp, R)
    return
end

isempty(ARGS) && error("usage: bench2_petab.jl <model_name>")
run_worker(ARGS[1])
