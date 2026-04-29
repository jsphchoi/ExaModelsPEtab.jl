#
function _create_continuity(
    c::ExaCore,
    PEmodel::PEtabModel,
    PEprob::PEtabODEProblem,
    PEinfo::PEInfo
)
    c = _create_interval_continuity(c, PEmodel, PEprob, PEinfo)
    c = _create_initial_conditions(c, PEmodel, PEprob, PEinfo)
    return c
end

# Create cross-interval continuity constraints
function _create_interval_continuity(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    (; Nz, N, Nc, L1) = PEinfo

    itr_cont1 = [(v,i,cidx) for v in 1:Nz, i in 1:N-1, cidx in 1:Nc]
    # TODO support ev (from ODEProblem callbacks) 
    # idea: create dict i=1:N -> f(t_event) to continuity constraint!
    ExaModels.@add_con(c,
        -z[v,i+1,0,cidx]
        for (v,i,cidx) in itr_cont1
    )
    itr_cont1! = [(v,i,cidx,j,L1[j+1]) for v in 1:Nz, i in 1:N-1, cidx in 1:Nc, j in 0:K]
    ExaModels.@add_con(c,
        (v,i,cidx) => L1j*z[v,i,j,cidx]
        for (v,i,cidx,j,L1j) in itr_cont1!
    )
    return c
end

# Create initial condition continuity constraints
function _create_initial_conditions(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    


    # x0fix
    ExaModels.@add_con(c,
        z[v,1,0,cidx] - z0
        for (z,cidx,z0) in itr_cont2
    )

    # x0 = p
    ExaModels.@add_con(c,
        z[v,1,0,cidx] - p[pidx]
        for (z,pidx) in itr_cont2
    )

    # x0(p)
    ExaModels.@add_con(c,
        z[v,1,0,cidx] - p[pidx]
        for (z,pidx) in itr_cont2
    )

    # x0SSpre(p)
    ExaModels.@add_con(c,
        z[v,1,0,cidx] - zss[v,cidx]
        for (z,cidx) in itr_cont2
    )
    return c
end