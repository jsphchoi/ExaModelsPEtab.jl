# ExaModelsPEtab.jl

Formulates [ExaModels](https://github.com/madsuite-org/ExaModels.jl) from
[PEtab](https://github.com/PEtab-dev/PEtab) models.

Unlike [PEtab.jl](https://github.com/sebapersson/PEtab.jl), which uses the sequential method for
dynamic optimization, `ExaModelsPEtab.jl` applies the simultaneous method (orthogonal collocation) to
formulate the problem as a large-scale NLP. The `ExaModel` can then be solved with 
[MadNLP](https://github.com/madsuite-org/MadNLP.jl) using CPU or GPU.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mit-shin-group/ExaModelsPEtab.jl/blob/main/LICENSE) [![Build Status](https://github.com/mit-shin-group/ExaModelsPEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mit-shin-group/ExaModelsPEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Usage

```julia
examodel_petab("path/to/petab.yaml"; backend = nothing, kwargs...)
```
Returns a `CollocationExaModel` from a [PEtab](https://github.com/PEtab-dev/PEtab) `.yaml` file.

### Keyword Arguments
- `backend` : array backend
- `kwargs...` passed on to `CollocationExaCore`

### Example

```julia
using ExaModelsPEtab, MadNLP

# Build the ExaModel from a PEtab `.yaml` file
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
