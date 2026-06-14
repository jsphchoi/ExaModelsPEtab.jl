# ExaModelsPEtab.jl

Formulates [ExaModels](https://github.com/exanauts/ExaModels.jl) from
 [PEtab](https://github.com/PEtab-dev/PEtab) parameter-estimation models.

Unlike [PEtab.jl](https://github.com/sebapersson/PEtab.jl), which uses the sequential method for
dynamic optimization, `ExaModelsPEtab.jl` applies the simultaneous method (orthogonal collocation) to
formulate the problem as an NLP. The `ExaModel` can then be solved with [MadNLP](https://github.com/MadNLP/MadNLP.jl)
using a CPU or GPU backend.

[![Build Status](https://github.com/jsphchoi/ExaModelsPEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jsphchoi/ExaModelsPEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Usage

An `ExaModel` is formulated in a single function call:

```julia
examodel = petab_examodel(
  filename::String;
  backend = nothing,
  K::Int = 4
)
```
**Keyword arguments**
- `backend` — Array backend: `nothing` for CPU (default) or `CUDA.CUDABackend()` for GPU.
- `K` — Degree of the Lagrange interpolating polynomial (number of collocation points per interval);
  defaults to `4`. Steady-state models ignore `K` since no mesh is required to solve `f(z, p) = 0`.

### Example

```julia
using ExaModelsPEtab, MadNLP

# Build the ExaModels NLP from a PEtab problem YAML file.
model_CPU = petab_examodel("path/to/problem.yaml")   # CPU
result_CPU = madnlp(model_CPU)

# Build the ExaModel using CUDA backend and solve with GPU solver
using CUDA, MadNLPGPU
model_GPU = petab_examodel(
  "path/to/problem.yaml";
  backend = CUDA.CUDABackend(),
  K = 4
)
result_GPU = madnlp(model_GPU; tol = 1e-6)
```


## References

1. Shin, S., Anitescu, M., & Pacaud, F. (2024). Accelerating optimal power flow with GPUs: SIMD
   abstraction of nonlinear programs and condensed-space interior-point methods. *Electric Power
   Systems Research*, 236, 110651.

2. Persson, S., Fröhlich, F., Grein, S., Loman, T., Ognissanti, D., Hasselgren, V., Hasenauer, J.,
   & Cvijovic, M. (2025). PEtab.jl: advancing the efficiency and utility of dynamic modelling.
   *Bioinformatics*, 41(9), btaf497.

3. Schmiester, L., Schälte, Y., Bergmann, F. T., Camba, T., Dudkin, E., Egert, J., Fröhlich, F.,
   Fuhrmann, L., Hauber, A. L., Kemmer, S., et al. (2021). PEtab—Interoperable specification of
   parameter estimation problems in systems biology. *PLoS Computational Biology*, 17(1), e1008646.