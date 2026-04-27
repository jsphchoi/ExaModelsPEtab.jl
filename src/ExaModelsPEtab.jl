"""
    ExaModelsPEtab

(description)

## Example Header

(description)
"""
module ExaModelsPEtab

    # Imports
    import ExaModels

    # LONG-TERM TODO TRIM DEPENDENCY: ONLY import the PEtab .yaml file parser
    import PEtab
    import ModelingToolkitBase as MTK 
    import OrdinaryDiffEq # used to simulate model at nominal guess to obtain good initial guesses for state variables
    import Symbolics

    # Includes
    include("structs.jl")       # data structure for parameter estimation problem
    include("constants.jl")     # get collocation equation constants
    include("utils.jl")         # build helper functions
    include("initialize.jl")    # get good initial conditions
    include("variables.jl")     # create decision variables
    include("collocation.jl")   # create collocation equality constraints
    include("continuity.jl")    # create continuity equality constraints
    include("objective.jl")     # create objective function

    # Exports
    include("userfuncs.jl")     # user-end functions
    export petab_examodel
    # TODO want to support plot(filename, result) or something similar using specified data visualization file

end
