"""
    examodel_petab(filename::String; backend = nothing, kwargs...)

Builds an `ExaModels.ExaModel` from a PEtab YAML file at `filename`.

kwargs:
- `backend` : array backend, `nothing` (default) for CPU, `CUDA.CUDABackend()` for GPU
- kwargs passed on to `CollocationExaCore`

# Example
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
"""
function examodel_petab(
        filename::String;
        backend = nothing,
        K::Int = 4,
        kwargs...
    )
    # Parse PEtab YAML file using PEtab.jl
    PEmodel = PEtabModel(filename)
    PEprob = PEtabODEProblem(PEmodel)

    if _is_steady_state(PEmodel)
        # Create ExaCore
        core = ExaModels.ExaCore(; backend = backend, concrete = Val(true))

        # Build ExaModel
        return _build_examodel_petab_ss(core, PEmodel, PEprob)
    else
        # Solve ODE at nominal p0 to get mesh
        t_nodes, sol, t_meas = _get_mesh_nodes(PEmodel, PEprob)

        # Create CollocationExaCore
        core = EMC.CollocationExaCore(t_nodes, K; backend = backend, kwargs...)

        # Build ExaModel
        return _build_examodel_petab(core, PEmodel, PEprob, sol, t_meas)
    end
end

# Builds the ExaModel
function _build_examodel_petab(
        c::ExaCore,
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        sol,
        t_meas::Vector
    )
    # Create decision variables {p,z,cv,y,sigma,zss} and problem info
    c, PEinfo = _create_variables(c, PEmodel, PEprob, sol, t_meas)
    
    # Create collocation constraints
    c = _create_collocation(c, PEmodel, PEprob, PEinfo)

    # Create cross-interval continuity, initial condition, auxiliary variable {cv,} constraints
    c = _create_continuity(c, PEmodel, PEprob, PEinfo)

    # Create objective function (Gaussian negative log-likelihood) and auxiliary variable {y,sigma} constraints
    c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)

    # Carry the PEtab labelling onto the model
    c = _attach_petab_meta(c, PEmodel, PEprob, PEinfo)

    # Create ExaModel
    model = ExaModels.ExaModel(c)

    # Provide good initial guess for objective function auxiliary variables {y,sigma} 
    ExaModels.set_start!(model, c.y, y0)
    ExaModels.set_start!(model, c.sigma, sigma0)

    return model
end

# Builds the ExaModel for steady-state models (no collocation mesh)
function _build_examodel_petab_ss(
        c::ExaCore,
        PEmodel::PEtabModel, 
        PEprob::PEtabODEProblem
    )
    # Check inconsistent PEtab model info
    _check_x0SSpre(PEprob) && error(
        "ExaModelsPEtab: unsupported pre-equilibration with steady-state (time=Inf) measurements."
    )
    # Create decision variables {p,cv,zss,y,sigma} and obtain problem info 
    c, PEinfo = _create_variables_ss(c, PEmodel, PEprob)

    # Create steady-state ODE RHS constraints, f(zss...) = 0
    c = _create_constraints_ss(c, PEmodel, PEprob, PEinfo)

    # Create objective function (Gaussian negative log-likelihood) and auxiliary variable {y,sigma} constraints
    # (Evaluated at zss)
    c, y0, sigma0 = _create_objective_ss(c, PEmodel, PEprob, PEinfo)

    # Carry the PEtab labelling onto the model
    c = _attach_petab_meta(c, PEmodel, PEprob, PEinfo)

    # Create ExaModel
    model = ExaModels.ExaModel(c)

    # Provide good initial guess for objective function auxiliary variables {y,sigma} 
    ExaModels.set_start!(model, c.y, y0)
    ExaModels.set_start!(model, c.sigma, sigma0)

    return model
end
