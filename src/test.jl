begin
    using ExaModels

    # LONG-TERM TODO TRIM DEPENDENCY: ONLY import the PEtab .yaml file parser
    using PEtab # for parsing PEtab file as get symbolics
    import ModelingToolkitBase as MTK 
    using Symbolics
    import OrdinaryDiffEq as ODE # for solving ode at nominal guess to obtain intiail guess for discretized
    import SteadyStateDiffEq as SSDE # for solving steady-state pre-equilibration initial states
    # using TestEnv
    # TestEnv.activate()
    using CUDA, MadNLPGPU
    function get_yaml_path(problem_name::String)
        cd()
        problem_dir = joinpath(pwd(),
            "OneDrive - Massachusetts Institute of Technology",
            "Git","ExaModelsPEtab.jl",
            "examples", 
            "Benchmark-Models", 
            problem_name
        )
        
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

    function load_petab(problem_name::String)
        PEmodel = PEtab.PEtabModel(get_yaml_path(problem_name))
        PEprob = PEtab.PEtabODEProblem(PEmodel)
        return PEmodel, PEprob
    end
    
end

using ExaModelsPEtab

function load_petab(filename::String)
    dir = joinpath(pwd(), "examples", "Benchmark-Models", filename)
    yaml = filter(f -> endswith(lowercase(f), ".yaml"), readdir(dir))
    PEmodel = PEtab.PEtabModel(joinpath(dir, first(yaml)))
    PEprob = PEtab.PEtabODEProblem(PEmodel)
    return PEmodel, PEprob
end

jls = ["structs.jl","constants.jl","utils.jl","initialize.jl","variables.jl","collocation.jl","continuity.jl","objective.jl","userfuncs.jl"]
for jl in jls
    include(joinpath("src", jl))
end

# Boehm_JProteomeRes2014
# Bruno_JExpBot2016
# Schwen_PONE2014
# Isensee_JCB2018 x0SSpre INITIALIZATION NOT WORKING. PETAB ISSUE.
# Crauste_CellSystems2017
# Raia_CancerResearch2011 E(3)
PEmodel, PEprob = load_petab("Crauste_CellSystems2017")

c = ExaModels.ExaCore(; backend = CUDA.CUDABackend(), concrete = Val(true))

c, PEinfo = _create_variables(c, PEmodel, PEprob, 5)
c.nvar

c = _create_collocation(c, PEmodel, PEprob, PEinfo)
c.ncon
c = _create_continuity(c, PEmodel, PEprob, PEinfo)
c.ncon

c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)
c.ncon

DoF = c.nvar - c.ncon
check_sense = PEinfo.Np - DoF

m = ExaModels.ExaModel(c)

ExaModels.set_start!(model, c.y, y0)
ExaModels.set_start!(model, c.sigma, sigma0)

madnlp(m)

# or just
m = petab_examodel(get_yaml_path("Crauste_CellSystems2017"))

madnlp(m)