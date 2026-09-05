# Creates the steady-state / cv auxiliary variable constraints
function _create_constraints_steadystate(
        core::ExaCore,
        PEinfo::PEtabInfo
    )
    core = _create_steadystate_constraints(core, PEinfo)

    core = _create_cv_constraints(core, PEinfo)

    return core
end

# Create steady-state constraints f(zss) = 0 for every pre-equilibration condition
function _create_steadystate_constraints(
        core::ExaCore,
        PEinfo::PEtabInfo
    )
    # Unpack variables and model functions
    zss, theta = core.zss, core.theta
    cvof = _get_cvof(core, PEinfo)
    f = _get_f(PEinfo)

    # Create steady-state constraint iterator, u after every event
    data = _get_data(PEinfo, PEinfo.preeq_conditions)
    u = [_get_u_value(PEinfo, id, Inf) for id in _get_u_ids(PEinfo)]
    Nd, Nu = size(data, 1), length(u)
    itr = [
        (ssidx, ntuple(d -> data[d,ssidx], Nd), ntuple(g -> u[g], Nu))
        for ssidx in 1:_get_Nss(PEinfo)
    ]

    # Create steady-state constraints
    for v in 1:_get_Nz(PEinfo)
        ExaModels.@add_con(core,
            f[v](theta[:], zss[:,ssidx], cvof(ssidx), ds, us, 0.0)
            for (ssidx, ds, us) in itr
        )
    end

    return core
end
