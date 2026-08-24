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

# for parsing SBML -> symbolic model
import SBMLImporter

# for building callback functions from symbolic model
import ModelingToolkitBase as MTK
import Symbolics

# for initial solve & steady-state model analysis
import OrdinaryDiffEq as ODE
import LinearAlgebra

for file in [
        "tables",
        "model",
        "controls",
        "spec",
        "mesh",
        "guesses",
        "variables",
        "dynamics",
        "objective",
        "steadystate",
        "api",
    ]
    include("$file.jl")
end

export examodel_petab

end
