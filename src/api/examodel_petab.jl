# the main ExaModelsPEtab API function examodels_petab
"""
    examodel_petab(filename; backend = nothing, kwargs...)

Returns an `ExaModel` from a [PEtab](https://github.com/PEtab-dev/PEtab) `.yaml` file.

# Keyword Arguments
- `backend` : array backend
- `kwargs...` passed on to `ExaCore`/`CollocationExaCore`

# Example
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
"""
function examodel_petab(
        filename::String;
        backend = nothing,
        kwargs...
    )
    # Check for unsupported kwargs
    for keyword in (:roots, :basis, :polynomial, :unknown_horizon)
        haskey(kwargs, keyword) && throw(ArgumentError("examodel_petab: CollocationExaCore's `$keyword` is not configurable!"))
    end

    # Build and return ExaModel
    if _is_steadystate(filename)
        return _build_examodel_petab_steadystate(filename, backend; kwargs...)
    else
        return _build_examodel_petab(filename, backend; kwargs...)
    end
end

# Checks if a PEtab model is a steady-state model
function _is_steadystate(filename)
    yaml = strip.(readlines(filename))
    i = findfirst(==("measurement_files:"), yaml)
    meas_path = joinpath(dirname(filename), strip(yaml[i+1][2:end]))

    lines = filter(!isempty, readlines(meas_path))
    icol = findfirst(==("time"), split(lines[1], '\t'))
    times = [
        parse(Float64, split(line, '\t')[icol]) 
        for line in lines[2:end]
    ]

    any(isinf, times) || return false
    all(isinf, times) || throw(ArgumentError(
        "examodel_petab: Cannot have both finite-time and steady-state (time=Inf) measurements.",
    ))
    return true
end

function _build_examodel_petab(
        filename,
        backend;
        kwargs...
    )
    # Parse and solve the model at nominal guess theta0
    PEinfo = _get_PEtabInfo(filename)

    # Create CollocationExaCore
    core = EMC.CollocationExaCore(PEinfo.nodes, PEinfo.K; backend, kwargs...)

    # Create decision variables
    core = _create_variables(core, PEinfo)

    # Create collocation and continuity constraints
    core = _create_constraints(core, PEinfo)

    # Create objective function
    core = _create_objective(core, PEinfo)

    return ExaModels.ExaModel(core)
end

function _build_examodel_petab_steadystate(
        filename,
        backend;
        kwargs...
    )
    # Parse and solve the model at nominal guess theta0
    PEinfo = _get_PEtabInfo(filename)

    # Create ExaCore
    core = ExaModels.ExaCore(; backend, kwargs...)

    # Create decision variables 
    # {theta (transformed params), zss (steadystate states) cv (condition-dep. vars), y (observable), sigma (noise)}
    core = _create_variables_steadystate(core, PEinfo)

    # Create steady-state constraints
    core = _create_constraints_steadystate(core, PEinfo)

    # Create objective function
    core = _create_objective(core, PEinfo)

    return ExaModels.ExaModel(core)
end