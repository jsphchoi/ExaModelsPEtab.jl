"""
    ExaModelsPEtab

Formulates [ExaModels](https://github.com/madsuite-org/ExaModels.jl) from
[PEtab](https://github.com/PEtab-dev/PEtab) models.

# Usage
[`examodel_petab`](@ref) returns a `CollocationExaModel` from a [PEtab](https://github.com/PEtab-dev/PEtab) `.yaml` file

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
        "meshinit",
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
