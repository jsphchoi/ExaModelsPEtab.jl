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
    function load_petab(problem_name::String)
        # Dynamically build the path based on the folder structure
        filename = joinpath(pwd(), "examples", problem_name, problem_name, "$(problem_name).yaml")
        if !isfile(filename)
            error("Could not find the YAML file at: $filename\nMake sure your Julia REPL is working in the project root directory.")
        end
        PEmodel = PEtab.PEtabModel(filename)
        PEprob = PEtab.PEtabODEProblem(PEmodel)
        return PEmodel, PEprob
    end
end

include("structs.jl")       # data structure for parameter estimation problem
include("constants.jl")     # get collocation equation constants
include("utils.jl")         # build helper functions
include("initialize.jl")    # get good initial conditions
include("variables.jl")     # create decision variables
include("collocation.jl")   # create collocation equality constraints
include("continuity.jl")    # create continuity equality constraints
include("objective.jl")     # create objective function

# Boehm_JProteomeRes2014
# Bruno_JExpBot2016
# Schwen_PONE2014
PEmodel, PEprob = load_petab("Schwen_PONE2014")

c = ExaModels.ExaCore(; concrete = Val(true))

K = 10
# Create decision variables
c, PEinfo = _create_variables(c, PEmodel, PEprob, K)

# Create constraints
c = _create_collocation(c, PEmodel, PEprob, PEinfo)
c = _create_continuity(c, PEmodel, PEprob, PEinfo)

# Create objective
c = _create_objective(c, PEmodel, PEprob, PEinfo)

return ExaModels.ExaModel(c)