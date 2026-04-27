"""
    ExaModelsPEtab

(description)

## Example Header

(description)
"""
module ExaModelsPEtab

    # Imports
    import ExaModels
    import PEtab # TODO trim dependencies, implement minimal parser

    # Include functions
    include("structs.jl")       # data structure for parameter estimation problem
    include("constants.jl")     # get collocation equation constants
    include("utils.jl")         # build helper functions
    include("initialize.jl")    # get good initial conditions
    include("variables.jl")     # create decision variables
    include("collocation.jl")   # create collocation equality constraints
    include("continuity.jl")    # create continuity equality constraints
    include("objective.jl")     # create objective function

    include("userfuncs.jl")        # user-end functions

    # Exports
    export petab_model

end
