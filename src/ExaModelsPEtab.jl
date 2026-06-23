"""
    ExaModelsPEtab

Solve [PEtab](https://github.com/PEtab-dev/PEtab) parameter-estimation problems as an NLP 
using simultaneous method for dynamic optimization (orthogonal collocation), built as an 
[ExaModels](https://github.com/exanauts/ExaModels.jl) `ExaModel`.

# Usage
[`petab_examodel`](@ref) builds the `ExaModel` from a PEtab problem YAML.

```julia
using ExaModelsPEtab, MadNLP
m = petab_examodel("path/to/problem.yaml")
madnlp(m)
```
"""
module ExaModelsPEtab

    # Imports   
    import ExaModels: ExaCore, ExaModels
    import PEtab: PEtabModel, PEtabODEProblem, PEtab    # TODO trim dependency: only import the PEtab .yaml file parser
    import ModelingToolkitBase as MTK 
    import Symbolics
    import OrdinaryDiffEq as ODE        # used to solve ODE using stiff solver at nominal p to obtain mesh and good initial guess
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