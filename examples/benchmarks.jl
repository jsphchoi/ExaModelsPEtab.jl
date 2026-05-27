using ExaModelsPEtab
using MadNLPGPU, CUDA

function get_yaml_path(problem_name::String)
    problem_dir = joinpath(pwd(), "examples", "Benchmark-Models", problem_name)
    
    # Check if the folder actually exists first
    if !isdir(problem_dir)
        return @error "$problem_name not found in ~/Benchmark-Models"
    end
    
    # Look for any file ending with .yaml (case-insensitive)
    yaml_files = filter(f -> endswith(lowercase(f), ".yaml"), readdir(problem_dir))
    if isempty(yaml_files)
        return @error "No .yaml file found in ~/Benchmark-Models/$problem_name"
    else
        # Return the absolute path to the first yaml file found
        return joinpath(problem_dir, first(yaml_files))
    end
end

function benchmark(problem_name::String)
    println("\n" * "="^50)
    println("BENCHMARK: $problem_name")
    
    yaml_path = get_yaml_path(problem_name)
    if !isfile(yaml_path)
        @warn "YAML file not found for $problem_name at path: $yaml_path"
        return
    end

    model = nothing
    try
        println("MODEL COMPILATION TIME: ")
        @time model = ExaModelsPEtab.petab_examodel(
            yaml_path;
            backend = CUDA.CUDABackend(), # CUDA.CUDAbackend(),
            K = 10
        )
    catch e
        @error "Compilation failed for $problem_name" exception=(e, catch_backtrace())
        return (status = :compile_failed, term_status = "Compile Error", p_star = NaN)
    end

    try 
        println("            SOLVE TIME: ")
        @time res = madnlp(model; tol = 1e-6)
        status = res.status
        pstar = res.p
        println("Termination Status: $term_status")
        println("Optimal Objective (p*): $p_star")
        return (status = :success, term_status = string(status), p_star = pstar)
    
    catch e
        @error "Failed to solve $problem_name" exception=(e, catch_backtrace())
        return (status = :solve_failed, term_status = "Solver Crash", p_star = NaN)
    end
    println("="^50)
end
 
files = [
    "Alkan_SciSignal2018",
    "Armistead_CellDeathDis2024",
    "Bachmann_MSB2011",
    "Beer_MolBioSystems2014",
    "Bertozzi_PNAS2020",
    "Blasi_CellSystems2016",
    "Boehm_JProteomeRes2014",
    "Borghans_BiophysChem1997",
    "Brannmark_JBC2010",
    "Bruno_JExpBot2016",
    "Chen_MSB2009",
    "Crauste_CellSystems2017",
    "Elowitz_Nature2000",
    "Fiedler_BMCSystBiol2016",
    "Froehlich_CellSystems2018",
    "Fujita_SciSignal2010",
    "Giordano_Nature2020",
    "Isensee_JCB2018",
    "Lang_PLOSComputBiol2024",
    "Laske_PLOSComputBiol2019",
    "Liu_IFACPapersOnLine2025",
    "Lucarelli_CellSystems2018",
    "Okuonghae_ChaosSolitonsFractals2020",
    "Oliveira_NatCommun2021",
    "Perelson_Science1996",
    "Rahman_MBS2016",
    "Raia_CancerResearch2011",
    "Raimundez_PCB2020",
    "SalazarCavazos_MBoC2020",
    "Schwen_PONE2014",
    "Smith_BMCSystBiol2013",
    "Sneyd_PNAS2002",
    "Weber_BMC2015",
    "Zhao_QuantBiol2020",
    "Zheng_PNAS2012"
]

for file in files
    benchmark(file)
    # CUDA.reclaim() 
end