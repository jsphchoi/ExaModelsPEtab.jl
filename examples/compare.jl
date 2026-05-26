using Optim, PEtab

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

function benchmark_petab(problem_name::String)
    println("\n" * "="^50)
    println("BENCHMARK: $problem_name")

    yaml_path = get_yaml_path(problem_name)

    if !isfile(yaml_path)
        @warn "YAML file not found for $problem_name at path: $yaml_path"
        return (problem_name, :missing_yaml)
    end

    try
        println("MODEL COMPILATION TIME: ")
        @time begin
            PEmodel = PEtab.PEtabModel(yaml_path)
            PEprob  = PEtab.PEtabODEProblem(PEmodel)
        end

        println("SOLVE TIME: ")
        res = nothing

        @time res = calibrate(PEprob, get_x(PEprob), Optim.IPNewton())

        # termination_status may not exist in older Optim versions
        try
            println("STATUS: ", Optim.termination_status(res))
        catch
        end

        return (
            problem_name,
            Optim.converged(res) ? :success : :failed
        )

    catch e
        @error "Failed to benchmark $problem_name" exception=(e, catch_backtrace())
        return (problem_name, :error)
    end
end

files = [                                   # {COMP., SOLVE} TIME
    "Alkan_SciSignal2018",                  # large model warnings, 42.9, 10.7
    "Armistead_CellDeathDis2024",           # 11.4, 117.1
    "Bachmann_MSB2011",                     # spam warnings, 16.5, 101.8
    "Beer_MolBioSystems2014",               # 57.2, 315.9
    "Bertozzi_PNAS2020",                    # 9.66, 126.7
    "Blasi_CellSystems2016",                # 14.6, 60.6
    "Boehm_JProteomeRes2014",               # 9.11, 111.1
    "Borghans_BiophysChem1997",             # 9.47, 893.1
    "Brannmark_JBC2010",                    # 10.9, 195.5
    "Bruno_JExpBot2016",                    # 9.27, 23.1
    "Chen_MSB2009",                         #
    "Crauste_CellSystems2017",              #
    "Elowitz_Nature2000",                   #
    "Fiedler_BMCSystBiol2016",              #
    "Froehlich_CellSystems2018",            #
    "Fujita_SciSignal2010",                 #
    "Giordano_Nature2020",                  #
    "Isensee_JCB2018",                      #
    "Lang_PLOSComputBiol2024",              #
    "Laske_PLOSComputBiol2019",             #
    "Liu_IFACPapersOnLine2025",             #
    "Lucarelli_CellSystems2018",            #
    "Okuonghae_ChaosSolitonsFractals2020",  #
    "Oliveira_NatCommun2021",               #
    "Perelson_Science1996",                 #
    "Rahman_MBS2016",                       #
    "Raia_CancerResearch2011",              #
    "Raimundez_PCB2020",                    #
    "SalazarCavazos_MBoC2020",              #
    "Schwen_PONE2014",                      #
    "Smith_BMCSystBiol2013",                #
    "Sneyd_PNAS2002",                       #
    "Weber_BMC2015",                        #
    "Zhao_QuantBiol2020",                   #
    "Zheng_PNAS2012"                        #
]

results = benchmark_petab.(files)

begin
    println("\nSUMMARY")
    println("========")
    println("Successful:")
    foreach(println, [name for (name, status) in results if status == :success])
    println("\nFailed / Missing / Error:")
    foreach(println, [name for (name, status) in results if status != :success])
end