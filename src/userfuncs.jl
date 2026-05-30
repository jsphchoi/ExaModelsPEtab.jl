"""
    petab_examodel(
        filename::String;
        backend = nothing
        K = 10
    )

Returns an ExaModel applying orthogonal collocation to problem in PEtab .yaml file.

## Example
```jldoctest
julia> m = petab_examodel("Crause_CellSystems2017.yaml", backend = CUDABackend())
An ExaModel{Float64, CuArray{Float64, 1, CUDA.DeviceMemory}, ...}
[...]

julia> madnlp(m)
"Execution stats: Optimal Solution Found (tol = 1.0e-06)."
```
"""
function petab_examodel(
        filename::String;
        backend = nothing, 
        K = 10
    )
    PEmodel = PEtab.PEtabModel(filename)    # TODO trim dependencies
    PEprob = PEtab.PEtabODEProblem(PEmodel) # TODO trim dependencies
    return _build_petab_examodel(PEmodel, PEprob, backend, K)
end

function _build_petab_examodel(
        PEmodel::PEtabModel, 
        PEprob::PEtabODEProblem, 
        backend, 
        K
    )
    # Create ExaCore
    c = ExaModels.ExaCore(; backend, concrete = Val(true))

    # Create decision variables
    c, PEinfo = _create_variables(c, PEmodel, PEprob, K)
    
    # Create constraints
    c = _create_collocation(c, PEmodel, PEprob, PEinfo)
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