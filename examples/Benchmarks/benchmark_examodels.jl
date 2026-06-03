# benchmark_examodels.jl — ExaModelsPEtab + MadNLP GPU benchmark
#
# Builds petab_examodel (with CUDABackend) and solves with MadNLP on GPU for all
# 35 benchmark models. Results are written to Benchmarks/results/{Model}_results.txt
# using prefixed keys (exa_*) so petab results in the same file are preserved.
#
# Compilation timing is split into two phases:
#   Phase 1 (presolve)   — PEtab setup + ODE solve at nominal θ to init the mesh
#   Phase 2 (ExaModels)  — collocation/continuity/objective setup + ExaModel(c) build
# Both timings are stored; final_report.jl computes the %ExaModels column from them.
#
# The run is resumable: models with a terminal result are skipped. Wrap with
# benchmark_examodels.sh to restart after a watchdog SIGKILL on compile timeout.
#
# Two-GPU usage (strided partition, one instance per GPU):
#   julia --project=. -t 1 examples/Benchmarks/benchmark_examodels.jl <gpu_id> <num_instances> <instance_idx>
# Single-GPU usage:
#   julia --project=. -t 1 examples/Benchmarks/benchmark_examodels.jl 0 1 0

using ExaModelsPEtab, PEtab, CUDA, MadNLPGPU, CUDSS, ExaModels

# ─── CONFIGURABLE SETTINGS ────────────────────────────────────────────────────
const K             = 5               # collocation points per mesh interval
const TOL           = 1e-6            # MadNLP solver tolerance
const COMPILE_LIMIT = 1800.0          # hard compile deadline [s] (30 min)
const SOLVE_LIMIT   = 86400.0         # MadNLP max_wall_time [s] (24 hr)
const MAX_ITER      = 100_000_000     # large so wall time is always the bottleneck
const WARMUP_MODEL  = "Bruno_JExpBot2016"
# ──────────────────────────────────────────────────────────────────────────────

const MODELDIR  = joinpath(@__DIR__, "..", "Benchmark-Models")
const RESULTDIR = joinpath(@__DIR__, "results")

const ALL_MODELS = [
    "Alkan_SciSignal2018", "Armistead_CellDeathDis2024", "Bachmann_MSB2011",
    "Beer_MolBioSystems2014", "Bertozzi_PNAS2020", "Blasi_CellSystems2016",
    "Boehm_JProteomeRes2014", "Borghans_BiophysChem1997", "Brannmark_JBC2010",
    "Bruno_JExpBot2016", "Chen_MSB2009", "Crauste_CellSystems2017",
    "Elowitz_Nature2000", "Fiedler_BMCSystBiol2016", "Froehlich_CellSystems2018",
    "Fujita_SciSignal2010", "Giordano_Nature2020", "Isensee_JCB2018",
    "Lang_PLOSComputBiol2024", "Laske_PLOSComputBiol2019", "Liu_IFACPapersOnLine2025",
    "Lucarelli_CellSystems2018", "Okuonghae_ChaosSolitonsFractals2020",
    "Oliveira_NatCommun2021", "Perelson_Science1996", "Rahman_MBS2016",
    "Raia_CancerResearch2011", "Raimundez_PCB2020", "SalazarCavazos_MBoC2020",
    "Schwen_PONE2014", "Smith_BMCSystBiol2013", "Sneyd_PNAS2002",
    "Weber_BMC2015", "Zhao_QuantBiol2020", "Zheng_PNAS2012",
]

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

function exa_finished(m)
    d = read_result(result_path(m))
    cs = get(d, "exa_compile_status", "")
    ss = get(d, "exa_solve_status", "")
    return cs in ("timeout", "error", "missing_yaml") ||
           (cs == "ok" && ss in ("ok", "timeout", "error"))
end

function bench_one(m)
    rp = result_path(m)
    yaml = get_yaml(m)
    if yaml === nothing
        write_result(rp, Dict("exa_compile_status" => "missing_yaml",
                               "exa_solve_status"   => "skipped"))
        return
    end

    # ── COMPILE ──────────────────────────────────────────────────────────────
    write_result(rp, Dict(
        "exa_compile_status" => "compiling", "exa_compile_time" => "",
        "exa_presolve_time"  => "",          "exa_solve_status"  => "skipped",
        "exa_solve_time"     => "",          "exa_term_status"   => "",
        "exa_objective"      => "",          "exa_iter"          => "",
        "exa_nvar"           => "",          "exa_ncon"          => "",
        "exa_error"          => "",
    ))
    @info "[$m] compiling (K=$K, compile_limit=$(COMPILE_LIMIT)s)..."

    model = nothing
    t_compile = 0.0
    t_presolve = 0.0
    try
        t0 = time()
        result = with_hard_deadline(COMPILE_LIMIT) do
            PEmodel = PEtab.PEtabModel(yaml)
            PEprob  = PEtab.PEtabODEProblem(PEmodel)
            c = ExaModels.ExaCore(; backend=CUDA.CUDABackend(), concrete=Val(true))

            # Phase 1: presolve — variable creation + ODE solve at nominal θ
            c, PEinfo = ExaModelsPEtab._create_variables(c, PEmodel, PEprob, K)
            t_phase1 = time() - t0

            # Phase 2: ExaModels compilation — collocation + ExaModel build
            c = ExaModelsPEtab._create_collocation(c, PEmodel, PEprob, PEinfo)
            c = ExaModelsPEtab._create_continuity(c, PEmodel, PEprob, PEinfo)
            c, y0, sigma0 = ExaModelsPEtab._create_objective(c, PEmodel, PEprob, PEinfo)
            mdl = ExaModels.ExaModel(c)
            ExaModels.set_start!(mdl, c.y, y0)
            ExaModels.set_start!(mdl, c.sigma, sigma0)
            CUDA.synchronize()
            (mdl, mdl.meta.nvar, mdl.meta.ncon, t_phase1)
        end
        t_compile  = round(time() - t0; digits=2)
        mdl, nvar, ncon, t_phase1 = result
        t_presolve = round(t_phase1; digits=2)
        model = mdl
        write_result(rp, Dict(
            "exa_compile_status" => "ok",
            "exa_compile_time"   => t_compile,
            "exa_presolve_time"  => t_presolve,
            "exa_nvar"           => nvar,
            "exa_ncon"           => ncon,
        ))
    catch e
        write_result(rp, Dict(
            "exa_compile_status" => "error",
            "exa_compile_time"   => "",
            "exa_error"          => sprint(showerror, e),
        ))
        @error "[$m] compile failed" exception=(e, catch_backtrace())
        return
    end

    # ── SOLVE ─────────────────────────────────────────────────────────────────
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
        write_result(rp, Dict(
            "exa_solve_status" => "error",
            "exa_error"        => sprint(showerror, e),
        ))
        @error "[$m] solve failed" exception=(e, catch_backtrace())
    end
    model = nothing; GC.gc(); CUDA.reclaim()
    @info "[$m] done"
end

function warmup()
    yaml = get_yaml(WARMUP_MODEL); yaml === nothing && return
    @info "warmup: JIT build+solve on $WARMUP_MODEL ..."
    try
        PEmodel = PEtab.PEtabModel(yaml)
        PEprob  = PEtab.PEtabODEProblem(PEmodel)
        c = ExaModels.ExaCore(; backend=CUDA.CUDABackend(), concrete=Val(true))
        c, PEinfo = ExaModelsPEtab._create_variables(c, PEmodel, PEprob, K)
        c = ExaModelsPEtab._create_collocation(c, PEmodel, PEprob, PEinfo)
        c = ExaModelsPEtab._create_continuity(c, PEmodel, PEprob, PEinfo)
        c, y0, sigma0 = ExaModelsPEtab._create_objective(c, PEmodel, PEprob, PEinfo)
        m = ExaModels.ExaModel(c)
        ExaModels.set_start!(m, c.y, y0)
        ExaModels.set_start!(m, c.sigma, sigma0)
        CUDA.synchronize()
        madnlp(m; tol=TOL, max_iter=MAX_ITER, max_wall_time=250.0,
               linear_solver=MadNLPGPU.CUDSSSolver)
        m = nothing; GC.gc(); CUDA.reclaim()
        @info "warmup done"
    catch e; @warn "warmup failed" exception=(e, catch_backtrace()); end
end

function main()
    gpu_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
    ninst  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    idx    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0
    CUDA.device!(gpu_id)
    @info "instance $idx/$ninst on GPU $gpu_id ($(CUDA.name(CUDA.device())))"
    mkpath(RESULTDIR)

    # Convert in-progress sentinels from a prior killed run to timeout
    for m in ALL_MODELS
        d = read_result(result_path(m))
        if get(d, "exa_compile_status", "") == "compiling"
            write_result(result_path(m), Dict("exa_compile_status" => "timeout",
                                               "exa_compile_time"   => string(COMPILE_LIMIT)))
        elseif get(d, "exa_compile_status", "") == "ok" &&
               get(d, "exa_solve_status", "") == "solving"
            write_result(result_path(m), Dict("exa_solve_status" => "timeout"))
        end
    end

    mine = [m for (i, m) in enumerate(ALL_MODELS) if (i - 1) % ninst == idx]
    todo = filter(!exa_finished, mine)
    @info "instance $idx: $(length(mine)) models assigned, $(length(todo)) remaining"
    isempty(todo) && (@info "nothing to do"; return)

    warmup()
    for (i, m) in enumerate(todo)
        println("\n" * "="^60); println("[$idx][$i/$(length(todo))] $m"); println("="^60)
        bench_one(m)
    end
    @info "instance $idx complete"
end

main()
