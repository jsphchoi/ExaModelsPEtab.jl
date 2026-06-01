# compare.jl
#
# Reference benchmark with PEtab.jl + Optim.IPNewton(), to compare against the
# ExaModelsPEtab / MadNLP (GPU) results from benchmarks.jl.
#
# Only models that SUCCESSFULLY SOLVED with ExaModels/MadNLP are run here: the
# selection reads examples/results/<model>.exa.txt and keeps those whose MadNLP
# termination status reports an optimal solve.
#
# Like benchmarks.jl this is ONE long-lived, resumable process with a JIT warmup,
# and a watchdog (needs `-t 2`) that bounds each phase and SIGKILLs on overrun.
# Results -> examples/results/<model>.petab.txt. Wrap with run_all.sh compare.
#
# Usage:  julia --project=. -t 2 examples/compare.jl

using Printf
using Optim, PEtab

const PETAB_TIME_LIMIT = 600.0     # s, Optim.Options time_limit (parity with MadNLP solve limit)
const PETAB_MAX_ITER   = 100_000
const COMPILE_LIMIT    = 600.0     # s, hard cap on PEtab model compilation
const SOLVE_BACKSTOP   = PETAB_TIME_LIMIT + 120.0

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
exa_path(m)   = joinpath(RESULTDIR, "$(m).exa.txt")
petab_path(m) = joinpath(RESULTDIR, "$(m).petab.txt")

# Run the PEtab comparison only for models MadNLP solved to optimality.
function exa_solved(m::String)
    d = read_result(exa_path(m))
    return get(d, "solve_status", "") == "ok" &&
           occursin("SUCCEEDED", uppercase(get(d, "term_status", "")))
end

function finished(m::String)
    d = read_result(petab_path(m))
    st = get(d, "compile_status", "")
    return st in ("timeout", "error", "missing_yaml", "crashed") ||
           (st == "ok" && get(d, "solve_status", "") in ("ok", "timeout", "error"))
end

# External-process watchdog (see benchmarks.jl for rationale): `sleep N; kill -9`.
# Callers pre-write the pessimistic (timeout) status before each phase.
function with_hard_deadline(f, seconds::Real)
    pid = getpid()
    watcher = run(`bash -c "sleep $(seconds); kill -9 $(pid)"`; wait = false)
    try
        return f()
    finally
        try; kill(watcher); catch; end
    end
end

function bench_one(m::String)
    rp = petab_path(m)
    R = Dict{String,Any}(
        "model" => m, "compile_status" => "compiling", "compile_time" => "",
        "solve_status" => "skipped", "solve_time" => "", "objective" => "",
        "iterations" => "", "converged" => "", "error" => "",
    )
    write_result(rp, R)

    yaml = get_yaml_path(m)
    if yaml === nothing || !isfile(yaml)
        R["compile_status"] = "missing_yaml"; write_result(rp, R); return
    end

    # COMPILE
    PEprob = nothing
    @info "[$m] PEtab compiling (limit=$(COMPILE_LIMIT)s)..."
    try
        t0 = time()
        PEprob = with_hard_deadline(COMPILE_LIMIT) do
            PEmodel = PEtab.PEtabModel(yaml)
            PEtab.PEtabODEProblem(PEmodel)
        end
        R["compile_time"]   = round(time() - t0; digits = 2)
        R["compile_status"] = "ok"
    catch e
        R["compile_status"] = "error"; R["compile_time"] = ""; R["error"] = sprint(showerror, e)
        write_result(rp, R)
        @error "[$m] PEtab compile failed" exception = (e, catch_backtrace())
        return
    end
    write_result(rp, R)

    # SOLVE (Optim.IPNewton)
    @info "[$m] PEtab calibrating (Optim.IPNewton, time_limit=$(PETAB_TIME_LIMIT)s)..."
    R["solve_status"] = "solving"; R["solve_time"] = ""
    write_result(rp, R)
    try
        opts = Optim.Options(iterations = PETAB_MAX_ITER, time_limit = PETAB_TIME_LIMIT)
        t0 = time()
        res = with_hard_deadline(SOLVE_BACKSTOP) do
            calibrate(PEprob, get_x(PEprob), Optim.IPNewton(); options = opts)
        end
        R["solve_time"]   = round(time() - t0; digits = 2)
        R["objective"]    = res.fmin
        R["iterations"]   = res.niterations
        R["converged"]    = string(res.converged)
        R["solve_status"] = "ok"
    catch e
        R["solve_status"] = "error"; R["solve_time"] = ""; R["error"] = sprint(showerror, e)
        @error "[$m] PEtab calibrate failed" exception = (e, catch_backtrace())
    end
    write_result(rp, R)
    @info "[$m] done: compile=$(R["compile_status"]) ($(R["compile_time"])s) solve=$(R["solve_status"]) ($(R["solve_time"])s) fmin=$(R["objective"])"
    GC.gc()
    return
end

function warmup()
    yaml = get_yaml_path(WARMUP_MODEL)
    (yaml === nothing || !isfile(yaml)) && return
    @info "warmup: JIT-compiling PEtab build/calibrate on $WARMUP_MODEL ..."
    try
        t0 = time()
        PEmodel = PEtab.PEtabModel(yaml)
        PEprob  = PEtab.PEtabODEProblem(PEmodel)
        calibrate(PEprob, get_x(PEprob), Optim.IPNewton();
                  options = Optim.Options(iterations = 5))
        @info "warmup complete in $(round(time()-t0;digits=1))s"
        GC.gc()
    catch e
        @warn "warmup failed (continuing)" exception = (e, catch_backtrace())
    end
end

function selected_models()
    isdir(RESULTDIR) || error("No results dir; run benchmarks.jl first.")
    ms = String[]
    for f in sort(readdir(RESULTDIR))
        endswith(f, ".exa.txt") || continue
        m = replace(f, ".exa.txt" => "")
        exa_solved(m) && push!(ms, m)
    end
    return ms
end

function print_comparison(models)
    println("\n" * "#"^124)
    println("                     EXAMODELS/MADNLP  vs  PETAB.jl/OPTIM-IPNEWTON   (only MadNLP-solved models)")
    println("#"^124)
    @printf("%-34s | %-8s | %-8s | %-8s | %-8s | %-18s | %-18s\n",
            "Model", "Exa cmp", "Exa slv", "PEt cmp", "PEt slv", "Exa objective", "PEtab objective")
    println("-"^124)
    for m in models
        e = read_result(exa_path(m)); p = read_result(petab_path(m))
        @printf("%-34s | %-8s | %-8s | %-8s | %-8s | %-18s | %-18s\n",
                m, get(e, "compile_time", ""), get(e, "solve_time", ""),
                get(p, "compile_time", ""), get(p, "solve_time", ""),
                get(e, "objective", ""), get(p, "objective", ""))
    end
    println("#"^124)
end

function main()
    mkpath(RESULTDIR)
    for f in readdir(RESULTDIR)
        endswith(f, ".petab.txt") || continue
        name = replace(f, ".petab.txt" => "")
        d = read_result(petab_path(name))
        cs = get(d, "compile_status", ""); ss = get(d, "solve_status", "")
        if cs == "compiling"
            d["compile_status"] = "timeout"; d["compile_time"] = string(COMPILE_LIMIT)
            write_result(petab_path(name), d)
            @warn "[$name] left mid-compile by a previous run; recording compile timeout"
        elseif cs == "ok" && ss == "solving"
            d["solve_status"] = "timeout"; d["solve_time"] = string(PETAB_TIME_LIMIT)
            write_result(petab_path(name), d)
            @warn "[$name] left mid-solve by a previous run; recording solve timeout"
        end
    end

    models = selected_models()
    println("Models that solved with ExaModels/MadNLP (PEtab comparison set): $(length(models))")
    foreach(m -> println("  - ", m), models)

    todo = filter(!finished, models)
    if isempty(todo)
        @info "all comparison models already finished; printing comparison"
        print_comparison(models); return
    end

    warmup()
    for (i, m) in enumerate(todo)
        println("\n" * "="^72)
        println("[$i/$(length(todo))] PEtab: $m")
        println("="^72)
        bench_one(m)
    end

    print_comparison(models)
end

main()
