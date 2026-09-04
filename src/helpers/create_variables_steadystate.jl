# Create decision variables
# - theta (transformed params)
# - zss (steadystate states)
# - cv (condition-dep. vars)

# objective decision variables created in create_objective.jl
# - y (observable)
# - sigma (noise)
function _create_variables_steadystate(
        core::ExaCore,
        PEinfo::PEtabInfo
    )
    core = _create_theta(core, PEinfo)

    core = _create_zss(core, PEinfo)

    if _has_cv(PEinfo)
        core = _create_cv(core, PEinfo)
    end

    return core
end
