"""
    ExaModelsPEtab

(description)

## Example Header

(description)
"""
module ExaModelsPEtab

    # Imports
    import ExaModels: ExaCore, ExaModels

    # LONG-TERM TODO TRIM DEPENDENCY: ONLY import the PEtab .yaml file parser
    import PEtab: PEtabModel, PEtabODEProblem, PEtab
    import ModelingToolkitBase as MTK 
    import Symbolics
    import OrdinaryDiffEq as ODE # for solving ode at nominal guess to obtain intiail guess for discretized
    import SteadyStateDiffEq as SSDE # for solving steady-state pre-equilibration initial states
    import LinearAlgebra # conservation-law detection for steady-state models (svd/qr null space)
    
    # Includes
    include("structs.jl")       # data structure for parameter estimation problem
    include("constants.jl")     # get collocation equation constants
    include("utils.jl")         # build helper functions
    include("initialize.jl")    # get good initial conditions
    include("variables.jl")     # create decision variables
    include("collocation.jl")   # create collocation equality constraints
    include("continuity.jl")    # create continuity equality constraints
    include("objective.jl")     # create objective function
    include("steadystate.jl")   # steady-state (time = inf) model path

    # Exports
    include("userfuncs.jl")     # user-end functions
    export petab_examodel
    # TODO want to support plot(filename, result) or something similar using specified data visualization file in the future

end