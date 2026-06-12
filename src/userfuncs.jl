"""
    petab_examodel(filename::String; backend = nothing, K = 3, mutable_mesh = false)

Build an `ExaModel` that applies orthogonal collocation to the PEtab parameter-estimation
problem in the PEtab problem YAML at `filename`. The result is a standard `NLPModels`
model, ready to solve with MadNLP.

- `backend` — array backend: `nothing` for CPU, or `CUDA.CUDABackend()` for GPU. Pass it
  from your own `using CUDA`; ExaModelsPEtab itself does not depend on CUDA.
- `K` — number of collocation points per mesh interval.

# Example
```julia
using ExaModelsPEtab, MadNLP
m = petab_examodel("Crauste_CellSystems2017.yaml")   # CPU
stats = madnlp(m)

# GPU (CUDSS sparse solver from MadNLPGPU):
using CUDA, MadNLPGPU
m = petab_examodel("Crauste_CellSystems2017.yaml"; backend = CUDA.CUDABackend())
stats = madnlp(m; linear_solver = MadNLPGPU.CUDSSSolver)
```
"""
function petab_examodel(
        filename::String;
        backend = nothing,
        K = 3,
        mutable_mesh::Bool = false
    )
    PEmodel = PEtab.PEtabModel(filename)    # TODO trim dependencies
    PEprob = PEtab.PEtabODEProblem(PEmodel) # TODO trim dependencies
    return _build_petab_examodel(PEmodel, PEprob, backend, K; mutable_mesh = mutable_mesh)
end

function _build_petab_examodel(
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        backend,
        K;
        mutable_mesh::Bool = false
    )
    # Steady-state models (all measurements at time = inf) carry no time-course information:
    # the data observe the t→∞ steady state, not a trajectory. They take a separate path with
    # NO collocation mesh — the NLP is just the steady-state equations f(zss)=0 plus the
    # objective. (See steadystate.jl.) K is unused there.
    if _is_steady_state(PEmodel)
        return _build_petab_examodel_ss(PEmodel, PEprob, backend)
    end

    # Create ExaCore
    c = ExaModels.ExaCore(; backend, concrete = Val(true))

    # Create decision variables
    c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
    
    # Create constraints
    c = _create_collocation(c, PEmodel, PEprob, PEinfo; mutable_mesh = mutable_mesh)
    c = _create_continuity(c, PEmodel, PEprob, PEinfo)

    # Create objective (also returns feasible initial guesses for the auxiliary
    # observable (y) and noise (sigma) variables, evaluated at the z/p starts)
    c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)

    model = ExaModels.ExaModel(c)

    # Warm-start y and sigma with values consistent with the z/p initial guesses.
    # set_start! requires a built ::ExaModel (not the ::ExaCore), so it happens here.
    ExaModels.set_start!(model, c.y, y0)
    ExaModels.set_start!(model, c.sigma, sigma0)

    return model
end