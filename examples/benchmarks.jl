# benchmarks.jl
#
# Benchmarks ExaModelsPEtab model COMPILATION and MadNLP (GPU / CUDABackend) SOLVE
# for every model in examples/Benchmark-Models.
#
# Parameters:  K = 5, tol = 1e-6, 10-min compile limit, 10-min solve limit,
#              max_iter huge so the wall-clock is always the bottleneck.
#
# Design notes
# ------------
# * All models run in ONE long-lived process. The first call into the build /
#   solve pipeline JIT-compiles ~5 min of generic Symbolics/ExaModels/MadNLP code;
#   running per-model subprocesses would re-pay that on every model and pollute the
#   measured compile times. A `warmup()` pass absorbs that one-time JIT cost so the
#   reported compile times reflect real model-build work.
# * Hard 10-min deadlines are enforced per phase by a watchdog on a separate OS
#   thread (needs `-t 2`) that writes the timeout status and then SIGKILLs the
#   process. MadNLP's own `max_wall_time` bounds the solve cleanly; the watchdog is
#   a backstop.
# * Results are written per-model to examples/results/<model>.exa.txt as they are
#   produced, so a watchdog kill loses nothing. The run is RESUMABLE: models with a
#   finished result file are skipped. Wrap with run_all.sh to auto-restart after a
#   compile-timeout SIGKILL.
#
# Usage:  julia --project=. -t 2 examples/benchmarks.jl

using Printf
using ExaModelsPEtab, CUDA, MadNLPGPU, CUDSS

const K             = 5
const TOL           = 1e-6
const COMPILE_LIMIT = 600.0        # s, hard
const SOLVE_LIMIT   = 600.0        # s, madnlp max_wall_time
const SOLVE_BACKSTOP = SOLVE_LIMIT + 90.0   # s, watchdog backstop above madnlp's limit
const MAX_ITER      = 100_000_000  # so wall-time is the bottleneck

const MODELDIR  = joinpath(@__DIR__, "Benchmark-Models")
const RESULTDIR = joinpath(@__DIR__, "results")
const WARMUP_MODEL = "Bruno_JExpBot2016"

function get_yaml_path(problem_name::String)
    problem_dir = joinpath(MODELDIR, problem_name)
    isdir(problem_dir) || return nothing
    yaml_files = filter(f -> endswith(lowercase(f), ".yaml"), readdir(problem_dir))
    isempty(yaml_files) && return nothing
    return joinpath(problem_dir, first(yaml_files))
end

# ---- key=value result files -------------------------------------------------
function write_result(path::String, d::AbstractDict)
    open(path, "w") do io
        for k in sort(collect(keys(d)))
            s = replace(string(d[k]), '\n' => ' ', '\r' => ' ')
            println(io, "$k=$s")
        end
        flush(io)
    end
end
function read_result(path::String)
    d = Dict{String,String}()
    isfile(path) || return d
    for line in eachline(path)
        i = findfirst('=', line)
        i === nothing && continue
        d[line[1:i-1]] = line[i+1:end]
    end
    return d
end
result_path(m) = joinpath(RESULTDIR, "$(m).exa.txt")

# A model is "finished" (skip on resume) if it has a terminal status.
function finished(m::String)
    d = read_result(result_path(m))
    st = get(d, "compile_status", "")
    return st in ("timeout", "error", "missing_yaml", "crashed") ||
           (st == "ok" && get(d, "solve_status", "") in ("ok", "timeout", "error"))
end

# ---- hard per-phase deadline via an EXTERNAL watchdog process ----------------
# A Julia watchdog *thread* that busy-waits starves CUDA/cuDSS async work and makes
# the GPU solve hang (observed: 20 s solve -> 690 s hang). So the watchdog is a
# separate `sleep N; kill -9 <pid>` subprocess with zero Julia-thread involvement;
# this lets us run at -t 1, which the GPU solve needs. Because the killer is external
# it cannot run a Julia callback, so callers PRE-WRITE the pessimistic (timeout)
# status before the phase and overwrite it on success.
function with_hard_deadline(f, seconds::Real)
    pid = getpid()
    watcher = run(`bash -c "sleep $(seconds); kill -9 $(pid)"`; wait = false)
    try
        return f()
    finally
        # Cancel: SIGTERM the bash. It is blocked in `sleep`, so it dies before
        # reaching the `kill`; the orphaned `sleep` simply exits later, harmlessly.
        try; kill(watcher); catch; end
    end
end

# ---- one model: compile + solve --------------------------------------------
function bench_one(m::String)
    rp = result_path(m)
    R = Dict{String,Any}(
        "model" => m, "compile_status" => "compiling", "compile_time" => "",
        "solve_status" => "skipped", "solve_time" => "", "term_status" => "",
        "objective" => "", "iter" => "", "nvar" => "", "ncon" => "", "error" => "",
    )
    write_result(rp, R)   # in-progress sentinel; if the watchdog SIGKILLs us mid-build
                          # this "compiling" survives and resume converts it to "timeout"

    yaml = get_yaml_path(m)
    if yaml === nothing || !isfile(yaml)
        R["compile_status"] = "missing_yaml"; write_result(rp, R); return
    end

    # COMPILE
    model = nothing
    @info "[$m] compiling (K=$K, limit=$(COMPILE_LIMIT)s)..."
    try
        t0 = time()
        model = with_hard_deadline(COMPILE_LIMIT) do
            x = petab_examodel(yaml; backend = CUDA.CUDABackend(), K = K)
            CUDA.synchronize()
            x
        end
        R["compile_time"]   = round(time() - t0; digits = 2)
        R["compile_status"] = "ok"
        R["nvar"] = model.meta.nvar
        R["ncon"] = model.meta.ncon
    catch e
        R["compile_status"] = "error"; R["compile_time"] = ""; R["error"] = sprint(showerror, e)
        write_result(rp, R)
        @error "[$m] compile failed" exception = (e, catch_backtrace())
        return
    end
    write_result(rp, R)

    # SOLVE.  MadNLP's own max_wall_time bounds it; the external watchdog is a
    # backstop for a pathological hang. "solving" sentinel survives a kill and
    # resume converts it to "timeout".
    @info "[$m] solving with MadNLP (limit=$(SOLVE_LIMIT)s)..."
    R["solve_status"] = "solving"; R["solve_time"] = ""
    write_result(rp, R)
    try
        t0 = time()
        res = with_hard_deadline(SOLVE_BACKSTOP) do
            madnlp(model; tol = TOL, max_iter = MAX_ITER,
                   max_wall_time = SOLVE_LIMIT, linear_solver = MadNLPGPU.CUDSSSolver)
        end
        R["solve_time"]   = round(time() - t0; digits = 2)
        R["term_status"]  = string(res.status)
        R["objective"]    = res.objective
        R["iter"]         = res.iter
        R["solve_status"] = "ok"
    catch e
        R["solve_status"] = "error"; R["solve_time"] = ""; R["error"] = sprint(showerror, e)
        @error "[$m] solve failed" exception = (e, catch_backtrace())
    end
    write_result(rp, R)
    @info "[$m] done: compile=$(R["compile_status"]) ($(R["compile_time"])s) solve=$(R["solve_status"]) ($(R["solve_time"])s) term=$(R["term_status"])"
    model = nothing; GC.gc(); CUDA.reclaim()
    return
end

# ---- warmup: absorb one-time JIT so measured compile times are real ---------
function warmup()
    yaml = get_yaml_path(WARMUP_MODEL)
    (yaml === nothing || !isfile(yaml)) && return
    @info "warmup: JIT-compiling the build/solve pipeline on $WARMUP_MODEL ..."
    try
        t0 = time()
        m = petab_examodel(yaml; backend = CUDA.CUDABackend(), K = K)
        CUDA.synchronize()
        @info "warmup compile done in $(round(time()-t0;digits=1))s; warmup solve (fully JIT the solver, 250s cap)..."
        madnlp(m; tol = TOL, max_iter = MAX_ITER, max_wall_time = 250.0,
               linear_solver = MadNLPGPU.CUDSSSolver)
        m = nothing; GC.gc(); CUDA.reclaim()
        @info "warmup complete in $(round(time()-t0;digits=1))s"
    catch e
        @warn "warmup failed (continuing anyway)" exception = (e, catch_backtrace())
    end
end

# Ordered smaller / likely-to-solve models first so useful results (and the
# downstream PEtab comparison) come early; the large, compile-timeout-prone models
# are deferred to the end. Order does not affect results (resume skips finished
# models); it only front-loads the informative cases.
const MODELS = [
    # smaller / known-working first
    "Bruno_JExpBot2016", "Crauste_CellSystems2017", "Boehm_JProteomeRes2014",
    "Blasi_CellSystems2016", "Borghans_BiophysChem1997", "Brannmark_JBC2010",
    "Bertozzi_PNAS2020", "Fujita_SciSignal2010", "Fiedler_BMCSystBiol2016",
    "Elowitz_Nature2000", "Sneyd_PNAS2002", "Weber_BMC2015",
    "Smith_BMCSystBiol2013", "Perelson_Science1996", "Rahman_MBS2016",
    "Zhao_QuantBiol2020", "Zheng_PNAS2012", "Okuonghae_ChaosSolitonsFractals2020",
    "Schwen_PONE2014", "Liu_IFACPapersOnLine2025", "Oliveira_NatCommun2021",
    "Laske_PLOSComputBiol2019",
    # already finished earlier (skipped on resume)
    "Alkan_SciSignal2018", "Armistead_CellDeathDis2024", "Bachmann_MSB2011",
    # large / compile-timeout-prone, deferred to the end
    "Beer_MolBioSystems2014", "Chen_MSB2009", "Froehlich_CellSystems2018",
    "Giordano_Nature2020", "Isensee_JCB2018", "Lang_PLOSComputBiol2024",
    "Lucarelli_CellSystems2018", "Raia_CancerResearch2011", "Raimundez_PCB2020",
    "SalazarCavazos_MBoC2020",
]

function print_summary()
    println("\n" * "#"^100)
    println("                              EXAMODELS / MADNLP BENCHMARK SUMMARY  (K=$K, tol=$TOL)")
    println("#"^100)
    @printf("%-36s | %-12s | %-8s | %-10s | %-8s | %-22s\n",
            "Model", "Compile", "C.time", "Solve", "S.time", "Term status")
    println("-"^100)
    for m in MODELS
        d = read_result(result_path(m))
        @printf("%-36s | %-12s | %-8s | %-10s | %-8s | %-22s\n",
                m, get(d, "compile_status", "?"), get(d, "compile_time", ""),
                get(d, "solve_status", "?"), get(d, "solve_time", ""),
                get(d, "term_status", ""))
    end
    println("#"^100)
end

function main()
    mkpath(RESULTDIR)
    # A model left mid-phase by a previous (killed) process: convert the in-progress
    # sentinel to a timeout — the only thing that kills us mid-phase is the watchdog.
    for m in MODELS
        d = read_result(result_path(m))
        cs = get(d, "compile_status", "")
        ss = get(d, "solve_status", "")
        if cs == "compiling"
            d["compile_status"] = "timeout"; d["compile_time"] = string(COMPILE_LIMIT)
            write_result(result_path(m), d)
            @warn "[$m] left mid-compile by a previous run; recording compile timeout"
        elseif cs == "ok" && ss == "solving"
            d["solve_status"] = "timeout"; d["solve_time"] = string(SOLVE_LIMIT)
            write_result(result_path(m), d)
            @warn "[$m] left mid-solve by a previous run; recording solve timeout"
        end
    end

    todo = filter(!finished, MODELS)
    if isempty(todo)
        @info "all models already finished; printing summary"
        print_summary(); return
    end
    @info "remaining models: $(length(todo)) / $(length(MODELS))"

    warmup()

    for (i, m) in enumerate(todo)
        println("\n" * "="^72)
        println("[$i/$(length(todo))] $m")
        println("="^72)
        bench_one(m)
    end

    print_summary()
end

main()
