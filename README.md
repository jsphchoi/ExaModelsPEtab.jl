# ExaModelsPEtab

[![Build Status](https://github.com/jsphchoi/ExaModelsPEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jsphchoi/ExaModelsPEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)

Solve [PEtab](https://github.com/PEtab-dev/PEtab) parameter-estimation models with
[ExaModels](https://github.com/exanauts/ExaModels.jl).

Unlike [PEtab.jl](https://github.com/sebapersson/PEtab.jl), which uses the sequential method for
dynamic optimization, ExaModelsPEtab applies the simultaneous method (orthogonal collocation) to
formulate the problem as an NLP. The `ExaModel` can then be solved with [MadNLP](https://github.com/MadNLP/MadNLP.jl)
using a CPU or GPU backend.

## Usage

An `ExaModel` is formulated in a single function call:

```julia
petab_examodel(
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

```bibtex
@article{schmiester2021petab,
  title={PEtab—Interoperable specification of parameter estimation problems in systems biology},
  author={Schmiester, Leonard and Sch{\"a}lte, Yannik and Bergmann, Frank T and Camba, Tacio and Dudkin, Erika and Egert, Janine and Fr{\"o}hlich, Fabian and Fuhrmann, Lara and Hauber, Adrian L and Kemmer, Svenja and others},
  journal={PLoS computational biology},
  volume={17},
  number={1},
  pages={e1008646},
  year={2021},
  publisher={Public Library of Science}
}

@article{persson2025petab,
  title={PEtab. jl: advancing the efficiency and utility of dynamic modelling},
  author={Persson, Sebastian and Fr{\"o}hlich, Fabian and Grein, Stephan and Loman, Torkel and Ognissanti, Damiano and Hasselgren, Viktor and Hasenauer, Jan and Cvijovic, Marija},
  journal={Bioinformatics},
  volume={41},
  number={9},
  pages={btaf497},
  year={2025},
  publisher={Oxford University Press}
}

@article{shin2024accelerating,
  title={Accelerating optimal power flow with GPUs: SIMD abstraction of nonlinear programs and condensed-space interior-point methods},
  author={Shin, Sungho and Anitescu, Mihai and Pacaud, Fran{\c{c}}ois},
  journal={Electric Power Systems Research},
  volume={236},
  pages={110651},
  year={2024},
  publisher={Elsevier}
}
```