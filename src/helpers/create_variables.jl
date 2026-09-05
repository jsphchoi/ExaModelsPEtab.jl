# Create decision variables 
# - theta (transformed params)
# - z (states), 
# - cv (condition-dep. vars)
# - zss (pre-eqbm steadystate)

# objective decision variables created in create_objective.jl
# - y (observable)
# - sigma (noise)
function _create_variables(
        core::EMC.CollocationExaCore,
        PEinfo::PEtabInfo
    )
    core = _create_theta(core, PEinfo)
    
    core = _create_z(core, PEinfo)

    if _has_cv(PEinfo)
        core = _create_cv(core, PEinfo)
    end

    if _has_zss(PEinfo)
        core = _create_zss(core, PEinfo)
    end

    return core
end

# Create theta (unknown transformed parameter vector)
function _create_theta(core, PEinfo)
    # Unpack PEtabInfo
    Ntheta = _get_Ntheta(PEinfo)
    estimated = [parameter for parameter in PEinfo.parameters if parameter.estimate]
    thetaL = [_logscale(parameter.lb, parameter.scale) for parameter in estimated]
    thetaU = [_logscale(parameter.ub, parameter.scale) for parameter in estimated]

    # Create ExaModels variable
    ExaModels.@add_var(core,
        theta,
        1:Ntheta;
        lvar  = thetaL,
        uvar  = thetaU,
        start = PEinfo.theta0,
    )

    return core
end

# Create z (discretized states)
function _create_z(core, PEinfo)
    # Unpack PEtabInfo
    Nz = _get_Nz(PEinfo)

    # Create CollocationVariable
    EMC.@add_var_collocation(core,
        z,
        1:Nz;
        start = PEinfo.z0,
    )

    return core
end

# Create cv (condition-dependent variables)
function _create_cv(core, PEinfo)
    # Unpack PEtabInfo
    Ncv, Nc = _get_Ncv(PEinfo), _get_Nc(PEinfo)

    # Create ExaModels variable
    ExaModels.@add_var(core,
        cv,
        1:Ncv, 1:Nc;
        start = PEinfo.cv0,
    )

    return core
end

# Create zss (pre-equilibration steady states)
function _create_zss(core, PEinfo)
    # Unpack PEtabInfo
    Nz, Nss = _get_Nz(PEinfo), _get_Nss(PEinfo)

    # Create ExaModels variable
    ExaModels.@add_var(core,
        zss,
        1:Nz, 1:Nss;
        start = reduce(hcat, PEinfo.zss0),
    )

    return core
end

_has_cv(PEinfo::PEtabInfo) = _get_Ncv(PEinfo) > 0
_has_zss(PEinfo::PEtabInfo) = _get_Nss(PEinfo) > 0

_get_Ntheta(PEinfo::PEtabInfo) = count(parameter -> parameter.estimate, PEinfo.parameters)
_get_Nz(PEinfo::PEtabInfo) = _get_Nz(PEinfo.model)
_get_Nc(PEinfo::PEtabInfo) = length(PEinfo.conditions)
_get_Ncv(PEinfo::PEtabInfo) = size(PEinfo.cv0, 1)
_get_Nss(PEinfo::PEtabInfo) = length(PEinfo.preeq_conditions)
