# benchmark_bachmann_k3.jl — one-off ExaModels benchmark of Bachmann at K=3.
#
# Bachmann OOM'd at the standard K=6 (4.10M vars filled the 32 GB GV100). At K=3 the
# variable count scales ~(K+1) → ~3/7 of that (~1.76M vars, ~17 GB) and should FIT.
# Writes to a SEPARATE file Bachmann_MSB2011_K3_results.txt so the K=6 OOM record in
# the main file is preserved. exa_K=3 is stamped so the coarser mesh is explicit.
# Warmup on Bruno (NOT the target). CPU/GPU: pinned to the GPU id in ARGS[1] (default 0).
#   julia --project=. -t 1 examples/debugging/benchmark_bachmann_k3.jl 1

using ExaModelsPEtab, PEtab, CUDA, MadNLPGPU, CUDSS, ExaModels

const K             = 3                # coarser mesh to fit the 32 GB GPU (vs standard 6)
const TOL           = 1e-6
const COMPILE_LIMIT = 14400.0
const SOLVE_LIMIT   = 7200.0          # 2 hr (matches the benchmark scripts)
const MAX_ITER      = 100_000_000
const N_SGM_RERUNS  = 3
const WARMUP_MODEL  = "Bruno_JExpBot2016"
const TARGET        = "Bachmann_MSB2011"

const MODELDIR  = joinpath(@__DIR__, "..", "Benchmark-Models")
const RESULTDIR = joinpath(@__DIR__, "results")
const RESULTFILE = joinpath(RESULTDIR, "Bachmann_MSB2011_K3_results.txt")

get_yaml(m) = begin
    d = joinpath(MODELDIR, m); isdir(d) || return nothing
    fs = filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))
    isempty(fs) ? nothing : joinpath(d, first(fs))
end

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
    merged = merge(existing, Dict(string(k) => replace(string(v), '\n'=>' ', '\r'=>' ')
                                  for (k,v) in updates))
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

# build_model copied from benchmark_examodels.jl (incl. the steady-state branch); K=3.
function build_model(yaml, t_origin)
    PEmodel = PEtab.PEtabModel(yaml)
    PEprob  = PEtab.PEtabODEProblem(PEmodel)
    c = ExaModels.ExaCore(; backend=CUDA.CUDABackend(), concrete=Val(true))
    if ExaModelsPEtab._is_steady_state(PEmodel)
        c, PEinfo = ExaModelsPEtab._create_variables_ss(c, PEmodel, PEprob)
        t_phase1 = time() - t_origin
        c = ExaModelsPEtab._create_ss_constraints(c, PEmodel, PEprob, PEinfo)
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
    CUDA.synchronize()
    return mdl, mdl.meta.nvar, mdl.meta.ncon, t_phase1
end

function run_sgm_reruns(model)
    write_result(RESULTFILE, Dict("exa_sgm_status"=>"running","exa_sgm_n"=>N_SGM_RERUNS))
    ts = Float64[]
    for i in 1:N_SGM_RERUNS
        @info "[Bachmann K=$K] SGM solve $i/$N_SGM_RERUNS ..."
        try
            t0 = time()
            with_hard_deadline(SOLVE_LIMIT+3600.0) do
                madnlp(model; tol=TOL, max_iter=MAX_ITER, max_wall_time=SOLVE_LIMIT,
                       linear_solver=MadNLPGPU.CUDSSSolver)
            end
            push!(ts, round(time()-t0; digits=2))
        catch e
            write_result(RESULTFILE, Dict("exa_sgm_status"=>"error","exa_sgm_error"=>sprint(showerror,e)))
            return
        end
    end
    sgm = round(exp(sum(log,ts)/length(ts)); digits=2)
    write_result(RESULTFILE, Dict("exa_sgm_status"=>"ok","exa_sgm_n"=>N_SGM_RERUNS,"exa_sgm_solve_time"=>sgm))
    @info "[Bachmann K=$K] SGM done: $sgm s"
end

function warmup()
    yaml = get_yaml(WARMUP_MODEL); yaml === nothing && return
    @info "warmup build+solve on $WARMUP_MODEL ..."
    try
        t0 = time(); mdl,_,_,_ = build_model(yaml, t0); CUDA.synchronize()
        madnlp(mdl; tol=TOL, max_iter=MAX_ITER, max_wall_time=250.0, linear_solver=MadNLPGPU.CUDSSSolver)
        mdl = nothing; GC.gc(); CUDA.reclaim(); @info "warmup done"
    catch e; @warn "warmup failed" exception=(e,catch_backtrace()); end
end

function main()
    gpu_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
    CUDA.device!(gpu_id)
    mkpath(RESULTDIR)
    @info "Bachmann K=$K benchmark on GPU $gpu_id ($(CUDA.name(CUDA.device())))"
    warmup()
    yaml = get_yaml(TARGET)
    write_result(RESULTFILE, Dict("exa_K"=>K, "exa_compile_status"=>"compiling",
        "exa_solve_status"=>"skipped","exa_objective"=>"","exa_error"=>""))
    model = nothing
    # ── COMPILE ──
    try
        @info "[Bachmann K=$K] compiling..."
        t0 = time()
        mdl,nvar,ncon,tp1 = with_hard_deadline(COMPILE_LIMIT) do; build_model(yaml,t0); end
        model = mdl
        write_result(RESULTFILE, Dict("exa_compile_status"=>"ok",
            "exa_compile_time"=>round(time()-t0;digits=2), "exa_presolve_time"=>round(tp1;digits=2),
            "exa_nvar"=>nvar, "exa_ncon"=>ncon))
        @info "[Bachmann K=$K] compiled: nvar=$nvar ncon=$ncon"
    catch e
        write_result(RESULTFILE, Dict("exa_compile_status"=>"error","exa_error"=>sprint(showerror,e)))
        @error "compile failed" exception=(e,catch_backtrace()); return
    end
    # ── FIRST SOLVE ──
    write_result(RESULTFILE, Dict("exa_solve_status"=>"solving"))
    try
        t0 = time()
        res = with_hard_deadline(SOLVE_LIMIT+3600.0) do
            madnlp(model; tol=TOL, max_iter=MAX_ITER, max_wall_time=SOLVE_LIMIT,
                   linear_solver=MadNLPGPU.CUDSSSolver)
        end
        write_result(RESULTFILE, Dict("exa_solve_status"=>"ok","exa_solve_time"=>round(time()-t0;digits=2),
            "exa_term_status"=>string(res.status), "exa_objective"=>res.objective, "exa_iter"=>res.iter))
        @info "[Bachmann K=$K] solved: $(res.status) obj=$(res.objective)"
    catch e
        write_result(RESULTFILE, Dict("exa_solve_status"=>"error","exa_error"=>sprint(showerror,e)))
        @error "solve failed" exception=(e,catch_backtrace()); model=nothing; GC.gc(); CUDA.reclaim(); return
    end
    # ── SGM ──
    run_sgm_reruns(model)
    model = nothing; GC.gc(); CUDA.reclaim()
    @info "[Bachmann K=$K] done"
end

main()
