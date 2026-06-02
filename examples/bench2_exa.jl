# bench2_exa.jl  —  PHASE 2 worker (one long-lived, warmed, resumable, GPU-pinned process).
#
# For every model that the PEtab phase RAN (results2/<m>.petab.txt solve_status=ok), builds
# the ExaModelsPEtab collocation model (K=5, 30-min compile limit) and solves it with MadNLP
# on the GPU. The MadNLP wall limit is set from the PEtab result:
#     PEtab found an optimum  -> 2 x (PEtab solve time)
#     PEtab only ran (no opt) -> 1 hour
#
# Two instances are run (one per GPU) over a strided partition of the model list.
# Usage:  julia --project=. -t 1 examples/bench2_exa.jl <gpu_id> <num_instances> <instance_idx>

using Printf
using ExaModelsPEtab, CUDA, MadNLPGPU, CUDSS

const K             = 5
const TOL           = 1e-6
const COMPILE_LIMIT = 1800.0          # 30 min
const FALLBACK_SOLVE = 3600.0         # 1 hour when PEtab did not converge
const MAX_ITER      = 100_000_000
const WARMUP_MODEL  = "Bruno_JExpBot2016"

const MODELDIR  = joinpath(@__DIR__, "Benchmark-Models")
const RESULTDIR = joinpath(@__DIR__, "results2")

get_yaml(m) = begin
    d = joinpath(MODELDIR, m); isdir(d) || return nothing
    fs = filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))
    isempty(fs) ? nothing : joinpath(d, first(fs))
end
exa_path(m)   = joinpath(RESULTDIR, "$(m).exa.txt")
petab_path(m) = joinpath(RESULTDIR, "$(m).petab.txt")

function write_result(path, d)
    open(path, "w") do io
        for k in sort(collect(keys(d))); println(io, "$k=", replace(string(d[k]), '\n'=>' ', '\r'=>' ')); end
        flush(io)
    end
end
function read_result(path)
    d = Dict{String,String}(); isfile(path) || return d
    for line in eachline(path); i=findfirst('=',line); i===nothing && continue; d[line[1:i-1]]=line[i+1:end]; end
    return d
end

# external-process watchdog (no Julia thread → no cuDSS interference)
function with_hard_deadline(f, seconds::Real)
    pid = getpid()
    w = run(`bash -c "sleep $(seconds); kill -9 $(pid)"`; wait=false)
    try; return f(); finally; try; kill(w); catch; end; end
end

petab_ran(m)    = get(read_result(petab_path(m)), "solve_status", "") == "ok"
function exa_finished(m)
    d = read_result(exa_path(m)); cs = get(d,"compile_status",""); ss = get(d,"solve_status","")
    return cs in ("timeout","error","missing_yaml") || (cs=="ok" && ss in ("ok","timeout","error"))
end

# MadNLP wall limit from the PEtab result
function madnlp_limit(m)
    d = read_result(petab_path(m))
    opt = get(d,"optimum_found","") == "true"
    t   = tryparse(Float64, get(d,"solve_time","")); t === nothing && (t = FALLBACK_SOLVE/2)
    return opt ? 2*t : FALLBACK_SOLVE
end

function bench_one(m)
    rp = exa_path(m)
    R = Dict{String,Any}("model"=>m, "compile_status"=>"compiling", "compile_time"=>"",
        "solve_status"=>"skipped", "solve_time"=>"", "term_status"=>"", "objective"=>"",
        "iter"=>"", "nvar"=>"", "ncon"=>"", "solve_limit"=>"", "error"=>"")
    write_result(rp, R)
    yaml = get_yaml(m)
    if yaml === nothing; R["compile_status"]="missing_yaml"; write_result(rp,R); return; end

    # COMPILE
    model = nothing
    @info "[$m] compiling (K=$K, limit=$(COMPILE_LIMIT)s)..."
    try
        t0 = time()
        model = with_hard_deadline(COMPILE_LIMIT) do
            x = petab_examodel(yaml; backend=CUDA.CUDABackend(), K=K); CUDA.synchronize(); x
        end
        R["compile_time"]=round(time()-t0;digits=2); R["compile_status"]="ok"
        R["nvar"]=model.meta.nvar; R["ncon"]=model.meta.ncon
    catch e
        R["compile_status"]="error"; R["compile_time"]=""; R["error"]=sprint(showerror,e)
        write_result(rp,R); @error "[$m] compile failed" exception=(e,catch_backtrace()); return
    end
    write_result(rp, R)

    # SOLVE
    lim = madnlp_limit(m); R["solve_limit"]=round(lim;digits=1)
    @info "[$m] solving with MadNLP (limit=$(round(lim;digits=1))s)..."
    R["solve_status"]="solving"; write_result(rp, R)
    try
        t0 = time()
        res = with_hard_deadline(lim + 90.0) do
            madnlp(model; tol=TOL, max_iter=MAX_ITER, max_wall_time=lim, linear_solver=MadNLPGPU.CUDSSSolver)
        end
        R["solve_time"]=round(time()-t0;digits=2); R["term_status"]=string(res.status)
        R["objective"]=res.objective; R["iter"]=res.iter; R["solve_status"]="ok"
    catch e
        R["solve_status"]="error"; R["solve_time"]=""; R["error"]=sprint(showerror,e)
        @error "[$m] solve failed" exception=(e,catch_backtrace())
    end
    write_result(rp, R)
    @info "[$m] done compile=$(R["compile_status"])($(R["compile_time"])s) solve=$(R["solve_status"])($(R["solve_time"])s) term=$(R["term_status"])"
    model=nothing; GC.gc(); CUDA.reclaim()
    return
end

function warmup()
    yaml = get_yaml(WARMUP_MODEL); yaml === nothing && return
    @info "warmup: JIT build+solve on $WARMUP_MODEL ..."
    try
        m = petab_examodel(yaml; backend=CUDA.CUDABackend(), K=K); CUDA.synchronize()
        madnlp(m; tol=TOL, max_iter=MAX_ITER, max_wall_time=250.0, linear_solver=MadNLPGPU.CUDSSSolver)
        m=nothing; GC.gc(); CUDA.reclaim(); @info "warmup done"
    catch e; @warn "warmup failed" exception=(e,catch_backtrace()); end
end

function main()
    gpu_id   = parse(Int, ARGS[1]); ninst = parse(Int, ARGS[2]); idx = parse(Int, ARGS[3])
    CUDA.device!(gpu_id)
    @info "instance $idx/$ninst on GPU $gpu_id ($(CUDA.name(CUDA.device())))"
    mkpath(RESULTDIR)

    # resume: convert leftover in-progress sentinels from a killed run to timeout
    for f in readdir(RESULTDIR)
        endswith(f, ".exa.txt") || continue
        nm = replace(f, ".exa.txt"=>""); d = read_result(exa_path(nm))
        if get(d,"compile_status","")=="compiling"
            d["compile_status"]="timeout"; d["compile_time"]=string(COMPILE_LIMIT); write_result(exa_path(nm), d)
        elseif get(d,"compile_status","")=="ok" && get(d,"solve_status","")=="solving"
            d["solve_status"]="timeout"; write_result(exa_path(nm), d)
        end
    end

    # passing models (PEtab ran), strided partition for this instance, sorted for determinism
    passing = sort([replace(f,".petab.txt"=>"") for f in readdir(RESULTDIR) if endswith(f,".petab.txt")])
    passing = filter(petab_ran, passing)
    mine = [m for (i,m) in enumerate(passing) if (i-1) % ninst == idx]
    todo = filter(!exa_finished, mine)
    @info "instance $idx: $(length(mine)) models assigned, $(length(todo)) remaining"
    isempty(todo) && (@info "nothing to do"; return)

    warmup()
    for (i,m) in enumerate(todo)
        println("\n"*"="^60); println("[$idx][$i/$(length(todo))] $m"); println("="^60)
        bench_one(m)
    end
    @info "instance $idx complete"
end

main()
