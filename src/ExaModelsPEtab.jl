"""
    ExaModelsPEtab

Solve [PEtab](https://github.com/PEtab-dev/PEtab) parameter-estimation problems as an NLP 
using simultaneous method for dynamic optimization (orthogonal collocation), built as an 
[ExaModels](https://github.com/exanauts/ExaModels.jl) `ExaModel`.

# Usage
[`examodel_petab`](@ref) builds the `ExaModel` from a PEtab problem YAML

```julia
using ExaModelsPEtab, MadNLP
model = examodel_petab("path/to/problem.yaml")
res = madnlp(model)
```
"""
module ExaModelsPEtab

import ExaModels: ExaCore, ExaModels

# for creating collocation constraints
import ExaModelsCollocation as EMC

# for parsing PEtab file -> symbolic model
import PEtab: PEtabModel, PEtabODEProblem, get_x, get_odeproblem

# for building callback functions from symbolic model
import ModelingToolkitBase as MTK 
import Symbolics

# for initial solve & steady-state model analysis
import OrdinaryDiffEq as ODE
import LinearAlgebra
    
for file in [
        "structs",
        "utils",
        "initialize",
        "variables",
        "collocation",
        "continuity",
        "objective",
        "steadystate",
    ]
    include("nlp/$file.jl")
end

include("exports.jl")
export examodel_petab

# ExaModelsPEtabPlotsExt.jl
include("plotres.jl")
export plotsol

end
