# bench2_petab.jl  —  PHASE 1 worker (one model per process; run many in parallel).
#
# Solves a PEtab problem with Ipopt (LBFGS Hessian approx), up to a 1-hour wall limit,
# and records compile/solve status+time, objective, Ipopt status, iters, and whether the
# model has event callbacks (discontinuities the ExaModels collocation cannot represent).
#
# "Pass" (qualifies for the ExaModels phase) = Ipopt RAN without erroring.
#
# Usage:  julia --project=. -t 1 examples/bench2_petab.jl <model_name>

using PEtab, Ipopt

const TOL          = 1e-8
const PETAB_MAXIT  = 1_000_000          # large so wall-time is the bottleneck
const PETAB_LIMIT  = parse(Float64, get(ENV, "PETAB_LIMIT", "3600.0"))    # 1 h solve wall limit
const COMPILE_LIMIT = parse(Float64, get(ENV, "PETAB_COMPILE_LIMIT", "1800.0"))  # 30 min build limit
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
        "solve_status"=>"skipped", "solve_time"=>"", "objective"=>"", "ipopt_status"=>"",
        "iters"=>"", "optimum_found"=>"", "has_events"=>"", "error"=>"")
    write_result(rp, R)

    yaml = get_yaml(m)
    if yaml === nothing
        R["compile_status"]="missing_yaml"; write_result(rp, R); return
    end

    # warm up the PEtab + Ipopt JIT on a small model so timings reflect real work
    try
        wy = get_yaml(WARMUP_MODEL)
        if wy !== nothing
            wprob = PEtabODEProblem(PEtabModel(wy))
            calibrate(wprob, get_x(wprob), IpoptOptimizer(true);
                      options = IpoptOptions(max_iter = 3))
        end
    catch; end

    # COMPILE
    PEprob = nothing
    try
        t0 = time()
        PEprob = with_hard_deadline(COMPILE_LIMIT) do
            PEtabODEProblem(PEtabModel(yaml))
        end
        R["compile_time"]   = round(time()-t0; digits=2)
        R["compile_status"] = "ok"
        R["has_events"]     = has_events(PEprob)
    catch e
        R["compile_status"]="error"; R["error"]=sprint(showerror, e)
        write_result(rp, R); return
    end
    write_result(rp, R)

    # SOLVE (Ipopt LBFGS)
    R["solve_status"]="solving"; write_result(rp, R)
    try
        opts = IpoptOptions(max_iter = PETAB_MAXIT, tol = TOL, max_wall_time = PETAB_LIMIT)
        t0 = time()
        # generous backstop: Ipopt only checks the wall limit between iterations, so a
        # slow final iteration can overrun max_wall_time; don't kill it prematurely.
        res = with_hard_deadline(PETAB_LIMIT + 600.0) do
            calibrate(PEprob, get_x(PEprob), IpoptOptimizer(true); options = opts)
        end
        R["solve_time"]    = round(time()-t0; digits=2)
        R["objective"]     = res.fmin
        R["ipopt_status"]  = string(res.converged)
        R["iters"]         = res.niterations
        # Ipopt status 0 = Solve_Succeeded, 1 = Solved_To_Acceptable_Level
        R["optimum_found"] = res.converged in (0, 1)
        # "ran" (qualifies) iff Ipopt did not throw and produced a finite objective
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
