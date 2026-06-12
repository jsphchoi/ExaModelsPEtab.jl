"""
    ExaModelsPEtab

(description)

## Example Header

(description)
"""
module ExaModelsPEtab

    # Imports
    import ExaModels: ExaCore, ExaModels
    import PEtab: PEtabModel, PEtabODEProblem, PEtab    # TODO trim dependency: only import the PEtab .yaml file parser
    import ModelingToolkitBase as MTK 
    import Symbolics
    import OrdinaryDiffEq as ODE        # used to solve ODE using stiff solver at nominal p to obtain mesh and good initial guess
    import SteadyStateDiffEq as SSDE    # used to solve the ODE model at steady-state to obtain initial guess for x0SSpre
    import LinearAlgebra                # used to detect and eliminate conservation law redundant DOF in steady-state model
    
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
    # TODO add plot(filename, result) or something similar using specified data visualization file in the future

end