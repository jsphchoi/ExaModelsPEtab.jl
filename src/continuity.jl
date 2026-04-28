#
function _create_continuity(
    c::ExaCore,
    PEmodel::PEtabModel,
    PEprob::PEtabODEProblem,
    PEinfo::PEInfo
)
    _create_interval_continuity(c, PEmodel, PEprob, PEinfo)
    _create_initial_conditions(c, PEmodel, PEprob, PEinfo)
end

# Create cross-interval continuity constraints
function _create_interval_continuity(c::ExaCore, PEinfo::PEInfo)
    Nz = PEinfo.Nz
    N = PEinfo.N
    Nc = PEinfo.Nc
    itr_cont1 = [(v,i,cidx) for v in 1:Nz, i in 1:N-1, cidx in 1:Nc]
    # TODO support ev (from ODEProblem callbacks) 
    # idea: create dict i=1:N -> f(t_event) to continuity constraint!
    ExaModels.@add_con(c,
        -z[v,i+1,0,cidx]
        for (v,i,cidx) in itr_cont1
    )
    L1 = PEinfo.L1
    itr_cont1! = [(v,i,cidx,j,L1[j+1]) for v in 1:Nz, i in 1:N-1, cidx in 1:Nc, j in 0:K]
    ExaModels.@add_con(c,
        (v,i,cidx) => L1j*z[v,i,j,cidx]
        for (v,i,cidx,j,L1j) in itr_cont1!
    )
end

# Create initial condition continuity constraints
function _create_initial_conditions()

end