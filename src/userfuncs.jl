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
    return _build_petab_model(PEmodel, PEprob, backend, K)
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
    PEinfo = _create_variables(c, PEprob, PEmodel)
    
    # Create constraints
    _create_collocation(c, PEprob, PEmodel, PEinfo)
    _create_continuity(c, PEprob, PEmodel, PEinfo)

    # Create objective
    _create_objective(c, PEprob, PEmodel, PEinfo)

    return ExaModels.ExaModel(c)
end