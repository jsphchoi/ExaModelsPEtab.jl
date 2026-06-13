"""
    petab_examodel(filename::String; backend = nothing, K = 4, adaptive_mesh = false)

Builds an `ExaModels.ExaModel` from a PEtab YAML file at `filename`.

kwargs:
- `backend` array backend: `nothing` for CPU or `CUDA.CUDABackend()` for GPU
- `K::Int` degree of Lagrange interpolating basis polynomial
- `adaptive_mesh::Bool` set to `true` to enable adaptive mesh refinement

# Example
```julia
# Using CPU solver with default settings
using ExaModelsPEtab, MadNLP
mCPU = petab_examodel("Crauste_CellSystems2017.yaml")
madnlp(mCPU)

# Using GPU solver with adaptive mesh refinement
using MadNLPGPU, CUDA, CUDSS
mGPU = petab_examodel(
    "Crauste_CellSystems2017.yaml"; 
    backend = CUDA.CUDABackend(),
    adaptive_mesh = true
)
madnlp(mGPU; tol = 1e-6)
```
"""
function petab_examodel(
        filename::String;
        backend = nothing,
        K::Int = 4,
        adaptive_mesh::Bool = false
    )
    # Parse PEtab YAML file using PEtab.jl
    PEmodel = PEtab.PEtabModel(filename)    # TODO trim dependencies
    PEprob = PEtab.PEtabODEProblem(PEmodel) # TODO trim dependencies

    # Build the ExaModel depending on if steady-state or dynamic
    if _is_steady_state(PEmodel)
        return _build_petab_examodel_ss(PEmodel, PEprob, backend)
    else
        return _build_petab_examodel(PEmodel, PEprob, backend, K; adaptive_mesh = adaptive_mesh)
    end
end

# Builds the ExaModel
function _build_petab_examodel(
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        backend,
        K::Int;
        adaptive_mesh::Bool = false
    )
    # Create ExaCore
    c = ExaModels.ExaCore(; backend, concrete = Val(true))

    # Create decision variables {p,z,cv,y,sigma,zss} and mesh info
    c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
    
    # Create collocation constraints
    c = _create_collocation(c, PEmodel, PEprob, PEinfo; adaptive_mesh = adaptive_mesh)

    # Create cross-interval continuity, initial condition, auxiliary variable {cv,} constraints
    c = _create_continuity(c, PEmodel, PEprob, PEinfo)

    # Create objective function (Gaussian negative log-liklihood) and auxiliary variable {y,sigma} constraints
    c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)

    # Create ExaModel
    model = ExaModels.ExaModel(c)

    # Provide good initial guess for objective function auxiliary variables {y,sigma} 
    ExaModels.set_start!(model, c.y, y0)
    ExaModels.set_start!(model, c.sigma, sigma0)

    return model
end

# Builds the ExaModel for steady-state models (no collocation mesh)
function _build_petab_examodel_ss(
        PEmodel::PEtabModel, 
        PEprob::PEtabODEProblem, 
        backend
    )
    # Check inconsistent PEtab model info
    _check_x0SSpre(PEprob) && error(
        "Pre-equilibration combined with steady-state (time = inf) measurements is not " *
        "supported yet."
    )

    # Create ExaCore
    c = ExaModels.ExaCore(; backend, concrete = Val(true))

    # Create decision variables {p,cv,zss,y,sigma} and obtain problem info 
    c, PEinfo = _create_variables_ss(c, PEmodel, PEprob)

    # Create steady-state ODE RHS constraints, f(zss...) = 0
    c = _create_constraints_ss(c, PEmodel, PEprob, PEinfo)

    # Create objective function (Gaussian negative log-liklihood) and auxiliary variable {y,sigma} constraints
    # (Evaluated at zss)
    c, y0, sigma0 = _create_objective_ss(c, PEmodel, PEprob, PEinfo)

    # Create ExaModel
    model = ExaModels.ExaModel(c)

    # Provide good initial guess for objective function auxiliary variables {y,sigma} 
    ExaModels.set_start!(model, c.y, y0)
    ExaModels.set_start!(model, c.sigma, sigma0)

    return model
end