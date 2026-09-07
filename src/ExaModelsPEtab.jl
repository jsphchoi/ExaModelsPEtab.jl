"""
    ExaModelsPEtab

Formulates [ExaModels](https://github.com/madsuite-org/ExaModels.jl) from
[PEtab](https://github.com/PEtab-dev/PEtab) models.

# Usage
[`examodel_petab`](@ref) returns an `ExaModel` from a [PEtab](https://github.com/PEtab-dev/PEtab) `.yaml` file

```julia
using ExaModelsPEtab, MadNLP

# Build the ExaModel from a PEtab problem `.yaml` file
model = examodel_petab("path/to/petab.yaml")
result = madnlp(model)

# Solve using GPU
using MadNLPGPU, CUDA, CUDSS

model = examodel_petab(
  "path/to/petab.yaml";
  backend = CUDA.CUDABackend()
)
result = madnlp(model; tol = 1e-6)
```
"""
module ExaModelsPEtab

import ExaModels: ExaCore, ExaModels

# for creating collocation constraints
import ExaModelsCollocation as EMC

# for SBML file -> symbolic model
import SBMLImporter

# for symbolic model -> callback functions
import ModelingToolkitBase as MTK
import Symbolics
import Symbolics.SymbolicUtils

# for initial solve & steady-state model analysis
import OrdinaryDiffEq as ODE
import LinearAlgebra

for file in [
        "helpers/structs",
        "helpers/utils",
        "helpers/get_PEtabInfo",
        "helpers/create_variables",
        "helpers/create_constraints",
        "helpers/create_objective",
        "helpers/create_variables_steadystate",
        "helpers/create_constraints_steadystate",
        "api/examodel_petab",
        "api/evaluate_objective",
    ]
    include("$file.jl")
end

export examodel_petab, evaluate_objective

end
