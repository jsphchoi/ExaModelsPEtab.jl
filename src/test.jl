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

cd()
cd(
    joinpath(
        pwd(),
        "OneDrive - Massachusetts Institute of Technology",
        "Git","ExaModelsPEtab.jl", "src"
    )
)
include("structs.jl")       # data structure for parameter estimation problem
include("constants.jl")     # get collocation equation constants
include("utils.jl")         # build helper functions
include("initialize.jl")    # get good initial conditions
include("variables.jl")     # create decision variables
include("collocation.jl")   # create collocation equality constraints
include("continuity.jl")    # create continuity equality constraints
include("objective.jl")     # create objective function
include("userfuncs.jl")     # user-end functions

# Boehm_JProteomeRes2014
# Bruno_JExpBot2016
# Schwen_PONE2014
# Isensee_JCB2018 x0SSpre INITIALIZATION NOT WORKING. PETAB ISSUE.
# Crauste_CellSystems2017
# Raia_CancerResearch2011 E(3)
PEmodel, PEprob = load_petab("Crauste_CellSystems2017")

c = ExaModels.ExaCore(; concrete = Val(true))

K = 10
# Create decision variables
c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
c.nvar

# Create constraints
c = _create_collocation(c, PEmodel, PEprob, PEinfo)
c.ncon
c = _create_continuity(c, PEmodel, PEprob, PEinfo)
c.ncon

# Create objective
c = _create_objective(c, PEmodel, PEprob, PEinfo)
c.ncon

DoF = c.nvar - c.ncon
check_sense = PEinfo.Np - DoF

m = ExaModels.ExaModel(c)

# or just
filename = joinpath(pwd(), "examples", "Crauste_CellSystems2017", "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")
m = petab_examodel(filename)

madnlp(m)