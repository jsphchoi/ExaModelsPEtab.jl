# ExaModelsPEtab.jl

Formulates [ExaModels](https://github.com/exanauts/ExaModels.jl) from
[PEtab](https://github.com/PEtab-dev/PEtab) parameter-estimation models.

Unlike [PEtab.jl](https://github.com/sebapersson/PEtab.jl), which uses the sequential method for
dynamic optimization, `ExaModelsPEtab.jl` applies the simultaneous method (orthogonal collocation) to
formulate the problem as a large-scale NLP. The `ExaModel` can then be solved with 
[MadNLP](https://github.com/MadNLP/MadNLP.jl) using a CPU or GPU.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mit-shin-group/ExaModelsPEtab.jl/blob/main/LICENSE) [![Build Status](https://github.com/jsphchoi/ExaModelsPEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jsphchoi/ExaModelsPEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Usage

```julia
model = examodel_petab("path/to/petab.yaml"; backend = nothing, kwargs...)
```
Builds an `ExaModel` from a PEtab file, with the solution maps attached as `model.petab`.

### Keyword Arguments
- `backend` : array backend, `nothing` (default) for CPU, `CUDA.CUDABackend()` for GPU
- `K` : degree of the collocating polynomial (default 4)
- `subdivide` : equal parts each required mesh interval is split into (default 4)
- `p0` : start values for the estimated parameters (estimation scale, parameters-table order)
- kwargs passed on to `CollocationExaCore`

### Example

```julia
using ExaModelsPEtab, MadNLP

# Build the ExaModel from a PEtab problem YAML file
model = examodel_petab("path/to/petab.yaml")   # CPU
res = madnlp(model)

# Build the ExaModel using CUDA backend and solve with MadNLPGPU
using CUDA, MadNLPGPU
model = examodel_petab(
  "path/to/petab.yaml";
  backend = CUDA.CUDABackend()
)
res = madnlp(model; tol = 1e-6)
```
