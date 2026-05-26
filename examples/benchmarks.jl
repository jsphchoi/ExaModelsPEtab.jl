using ExaModelsPEtab
using MadNLPGPU, CUDA

function get_yaml_path(problem_name::String)
    return joinpath(
        pwd(), "examples", "Benchmark-Models", problem_name, "$(problem_name).yaml"
    )
end

function benchmark(problem_name::String)
    println("\n" * "="^50)
    println("BENCHMARK: $problem_name")
    
    yaml_path = get_yaml_path(problem_name)
    
    if !isfile(yaml_path)
        @warn "YAML file not found for $problem_name at path: $yaml_path"
        return
    end

    try
        println("MODEL COMPILATION TIME: ")
        model = @time ExaModelsPEtab.petab_examodel(
            yaml_path;
            backend = nothing, # CUDA.CUDAbackend(),
            K = 10
        )
        
        println("            SOLVE TIME: ")
        @time madnlp(model; tol = 1e-6)
        
        
    catch e
        @error "Failed to benchmark $problem_name due to an error:" exception=(e, catch_backtrace())
    end
    println("="^50)
end
 
files = [
    "Alkan_SciSignal2018",
    "Armistead_CellDeathDis2024",
    # "Bachmann_MSB2011",
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