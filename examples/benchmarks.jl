using ExaModelsPEtab
using MadNLPGPU, CUDA
using Printf

function get_yaml_path(problem_name::String)
    problem_dir = joinpath(pwd(), "examples", "Benchmark-Models", problem_name)
    
    # Check if the folder actually exists first
    if !isdir(problem_dir)
        @error "$problem_name not found in ~/Benchmark-Models"
        return nothing
    end
    
    # Look for any file ending with .yaml (case-insensitive)
    yaml_files = filter(f -> endswith(lowercase(f), ".yaml"), readdir(problem_dir))
    if isempty(yaml_files)
        @error "No .yaml file found in ~/Benchmark-Models/$problem_name"
        return nothing
    else
        # Return the absolute path to the first yaml file found
        return joinpath(problem_dir, first(yaml_files))
    end
end

function benchmark(problem_name::String)
    println("\n" * "="^50)
    println("BENCHMARK: $problem_name")
    
    yaml_path = get_yaml_path(problem_name)
    if isnothing(yaml_path) || !isfile(yaml_path)
        @warn "YAML file not found for $problem_name at path: $yaml_path"
        return (status = :skipped, term_status = "Missing File")
    end

    model = nothing
    try
        println("MODEL COMPILATION TIME: ")
        @time model = ExaModelsPEtab.petab_examodel(
            yaml_path;
            backend = CUDA.CUDABackend(),
            K = 10
        )
    catch e
        @error "Compilation failed for $problem_name" exception=(e, catch_backtrace())
        return (status = :compile_failed, term_status = "Compile Error")
    end

    try 
        println("            SOLVE TIME: ")
        @time res = madnlp(model; tol = 1e-6, max_iter = 10000)
        status = res.status
        println("Termination Status: $status")
        return (status = :success, term_status = string(status))
    catch e
        @error "Failed to solve $problem_name" exception=(e, catch_backtrace())
        return (status = :solve_failed, term_status = "Solver Crash")
    end
end
 
files = [
    # "Alkan_SciSignal2018",          # __u_model found error
    # "Armistead_CellDeathDis2024",   # :observableParameters NOT FOUND? NOT ALWAYS HERE. TODO PARSING AT OBSERVABLEFORMULA STAGE FIRST 
    # "Bachmann_MSB2011",             # BoundsError var var var...
    # "Beer_MolBioSystems2014",       # BoundsError var var var...
    # "Bertozzi_PNAS2020",            # BoundsError var var var...
    # "Blasi_CellSystems2016",
    # "Boehm_JProteomeRes2014",
    # "Borghans_BiophysChem1997",
    # "Brannmark_JBC2010",
    "Bruno_JExpBot2016",
    # "Chen_MSB2009",
    "Crauste_CellSystems2017"
    # "Elowitz_Nature2000",
    # "Fiedler_BMCSystBiol2016",
    # "Froehlich_CellSystems2018",
    # "Fujita_SciSignal2010",
    # "Giordano_Nature2020",
    # "Isensee_JCB2018",
    # "Lang_PLOSComputBiol2024",
    # "Laske_PLOSComputBiol2019",
    # "Liu_IFACPapersOnLine2025",
    # "Lucarelli_CellSystems2018",
    # "Okuonghae_ChaosSolitonsFractals2020",
    # "Oliveira_NatCommun2021",
    # "Perelson_Science1996",
    # "Rahman_MBS2016",
    # "Raia_CancerResearch2011",
    # "Raimundez_PCB2020",
    # "SalazarCavazos_MBoC2020",
    # "Schwen_PONE2014",
    # "Smith_BMCSystBiol2013",
    # "Sneyd_PNAS2002",
    # "Weber_BMC2015",
    # "Zhao_QuantBiol2020",
    # "Zheng_PNAS2012"
]

results = Dict{String, NamedTuple}()

for file in files
    res = benchmark(file)
    results[file] = res
    CUDA.reclaim() 
end

begin
    println("\n" * "="^60)
    println("                       BENCHMARK SUMMARY")
    println("#"^60)
    @printf("%-35s | %-15s | %-15s\n", "Model", "Internal Status", "Solver Status")
    println("-"^70)
    for file in files
        r = results[file]
        @printf("%-35s | %-15s | %-15s\n", file, string(r.status), r.term_status)
    end
    println("#"^60)
end