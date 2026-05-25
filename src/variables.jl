#######################################################
# STATE AS OF: 05/20/26
# COMPLETE: >=1 experimental conditions
# TODO: support u(t), u(t,p)
#######################################################

# (*) Main function for creating ExaModels decision variables (*)
function _create_variables(
        c::ExaCore,
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        K::Int
    )
    # Create necessary variables (discretized state, unknown params) and obtain problem details (::PEInfo)
    c, Np = _create_p(c, PEprob)
    c, Nz, N, K, Nc, t_meas, t_vec_mesh, h, taus, L1 = _create_z(c, PEmodel, PEprob, K)

    Ncv = length(_get_cv_syms(PEmodel)) # number of condition-dependent variables
    Nm = length(eachrow(PEmodel.petab_tables[:measurements])) # number of data measurements
    Ny = length(_get_obsids(PEmodel)) # number of model observables
    PEinfo = PEInfo(Np, Nz, Nc, Ncv, Nm, Ny, N, K, t_meas, t_vec_mesh, h, taus, L1)

    # OBJECTIVE FUNCTION VARIABLES
    # Create auxiliary variables for model observables, y
    # Create auxiliary variables for measurement errors, sigma
    c = _create_y(c, PEmodel, PEprob, PEinfo)
    c = _create_sigma(c, PEmodel, PEprob, PEinfo)

    # If there are condition-dependent variables...
    if Ncv >= 1 
        c = _create_cv(c, PEinfo)
    end

    # If initial conditions are steady-state equilibrium...
    if _check_x0SSpre(PEprob) 
        c = _create_zss(c, PEmodel, PEprob, PEinfo)
    end

    # # If there are time-dependent input functions u(t) or u(t,p)...
    # if _check_ut(PEprob)
    #     c = _create_utp(c, PEmodel, PEprob, PEinfo) # TODO +u(t), +u(t,p)
    # end
    
    return c, PEinfo
end

# Creates ExaModels decision variables for unknown parameters
# p[1:Np]
function _create_p(c::ExaCore, PEprob::PEtabODEProblem)
    Np = PEprob.nparameters_estimate # number of unknown parameters to fit
    (; lower_bounds, upper_bounds, xnames, xnominal, model_info) = PEprob
    p_LB = PEtab.transform_x(Array(lower_bounds), xnames, model_info.xindices; to_xscale=false)
    p_UB = PEtab.transform_x(Array(upper_bounds), xnames, model_info.xindices; to_xscale=false)
    p_init = PEtab.transform_x(Array(xnominal),   xnames, model_info.xindices; to_xscale=false)
    ExaModels.@add_var(c,
        p,
        1:Np;
        lvar = p_LB,
        uvar = p_UB,
        start = p_init
    )
    return c, Np
end

# Creates ExaModels decision variables for discretized states
# z[1:Nz,1:N,0:K,1:Nc]
function _create_z(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, K::Int)
    z_init, Nz, N, K, Nc, t_meas, t_vec_mesh, h, taus, L1 = _get_z_init(PEmodel, PEprob, K)
    ExaModels.@add_var(c,
        z,
        1:Nz, 1:N, 0:K, 1:Nc;
        lvar = 0.0, # TODO find lower bounds for states, if at all
        uvar = Inf, # TODO find upper bounds for states, if at all
        start = z_init
    )
    return c, Nz, N, K, Nc, t_meas, t_vec_mesh, h, taus, L1
end

# Creates ExaModels decision variables for condition-dependent variables
# cv[1:Ncv,1:Nc]
function _create_cv(c::ExaCore, PEinfo::PEInfo)
    # Unpack problem info
    (; Nc, Ncv) = PEinfo
    ExaModels.@add_var(c,
        cv, 
        1:Ncv, 1:Nc # will either be fixed values or p
    )
    return c
end

# Creates ExaModels decision variables for steady-state pre-equilibrium state 
# zss[1:Nz,1:Nc]
function _create_zss(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    # Unpack problem info
    (; Nz, Nc) = PEinfo
    zss_init = _get_zss_init(PEmodel, PEprob, PEinfo)
    ExaModels.@add_var(c,
        zss,
        1:Nz, 1:Nc;
        lvar = 0.0, # TODO find lower bounds for states, if at all
        uvar = Inf, # TODO find upper bounds for states, if at all
        start = zss_init
    )
    return c
end

# Creates ExaModels (auxiliary) decision variables for observable model variable
# y[1:Nm]
function _create_y(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    (; Nm) = PEinfo
    ExaModels.@add_var(c,
        y,
        1:Nm
    )
    return c
end

# Creates ExaModels (auxiliary) decision variables for standard deviation of error
# sigma[1:Nm]
function _create_sigma(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    (; Nm) = PEinfo
    ExaModels.@add_var(c,
        sigma,
        1:Nm;
        lvar = 1e-10, # avoid divide by 0
        start = 1.0
    )
    return c
end

# TODO
# Creates ExaModels decision variables for time-dependent input functions u(t) or u(t,p)
# utp[1:Nz,1:Nc]
function _create_utp(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    
    return c
end

