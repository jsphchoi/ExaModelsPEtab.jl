# ExaModelsPEtab

[![Build Status](https://github.com/jsphchoi/ExaModelsPEtab.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jsphchoi/ExaModelsPEtab.jl/actions/workflows/CI.yml?query=branch%3Amain)

Solve [PEtab](https://github.com/PEtab-dev/PEtab) parameter-estimation problems with
[ExaModels](https://github.com/exanauts/ExaModels.jl) + [MadNLP](https://github.com/MadNLP/MadNLP.jl),
on the GPU.

Instead of repeatedly integrating the ODE inside an outer optimizer, ExaModelsPEtab transcribes the
estimation problem into a single large-scale nonlinear program by **orthogonal collocation**: the ODE
trajectory is discretized on a mesh and the collocation equations become equality constraints, so the
states and parameters are optimized *simultaneously*. The resulting NLP is built as an `ExaModel`
(SIMD-structured, GPU-resident) and solved with MadNLP using the cuDSS sparse linear solver on CUDA.

## Installation

```julia
using Pkg
Pkg.develop(path = "/path/to/ExaModelsPEtab")
```

Requires Julia ≥ 1.10. The GPU path needs a CUDA-capable device (via `CUDA.jl` / `MadNLPGPU` /
`CUDSS`); the package also builds and solves on CPU with `backend = nothing`.

## Usage

```julia
using ExaModelsPEtab, MadNLP

# Build the collocation NLP from a PEtab problem YAML.
model = petab_examodel("path/to/problem.yaml"; K = 10)   # CPU
stats = madnlp(model)

# GPU:
using CUDA, MadNLPGPU
model = petab_examodel("path/to/problem.yaml"; backend = CUDA.CUDABackend(), K = 10)
stats = madnlp(model; linear_solver = MadNLPGPU.CUDSSSolver)
```

`petab_examodel(filename; backend = nothing, K = 10)` is the single entry point:
- `backend` — `nothing` for CPU, `CUDA.CUDABackend()` for GPU.
- `K` — number of collocation points per mesh interval (accuracy/cost knob). Steady-state models
  (all measurements at t→∞) take a separate no-mesh path where `K` is unused.

## Tests

```julia
using Pkg; Pkg.test()
```

The test suite builds and solves a small set of benchmark models that span the construction paths
(steady-state, time-course collocation, multi-condition) on CPU, and additionally on GPU when
`CUDA.functional()`, checking both convergence and the objective value.

## Benchmarks

`examples/Benchmarks/` benchmarks ExaModelsPEtab (MadNLP/CUDA) against PEtab.jl (Optim.IPNewton)
across the PEtab benchmark-model collection in `examples/Benchmark-Models/`.

- `options.jl` — single source of truth for the model lists and all solver/benchmark settings.
- `run_examodels.{jl,sh}`, `run_petab.{jl,sh}`, `run_bruno.jl` — the benchmark drivers.
- `results/` + `results.jl` — per-model results and the regenerated scoreboard (`results.txt`).
- `RUNNING_BENCHMARKS.md` — operational runbook for launching and interpreting a run.
- `MULTISTART_FINDINGS.md` — notes on warm-start / local-optimum behavior.

`examples/debugging/` holds standalone diagnostic scripts (build-staging, warm-start feasibility,
objective-consistency, event/gate validation) used while bringing models up.
