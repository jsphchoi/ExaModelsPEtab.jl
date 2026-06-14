# (*) Main function for creating ExaModels decision variables (*)
function _create_variables(
        c::ExaCore,
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        K::Int
    )
    _assert_supported_events(PEmodel) # not supporting SBML <event> models

    # Create necessary variables (discretized state, unknown params) and obtain problem details (::PEInfo)
    c, Np = _create_p(c, PEprob)
    c, Nz, N, K, Nc, t_meas, h, taus, L1, t_nodes = _create_z(c, PEmodel, PEprob, K)

    # Get ::PEInfo details
    Ncv = length(_get_cv_syms(PEmodel)) # number of condition-dependent variables
    Nm = length(eachrow(PEmodel.petab_tables[:measurements])) # number of data measurements
    pscale = _get_pscale(PEprob) # per-parameter estimation scale (:log10/:log/:lin), aligned 1:Np
    gate_vals, gate_vals_ss = _get_gate_vals(PEmodel, PEprob, h, taus, t_nodes) # get gate value profiles and steady-state values
    PEinfo = PEInfo(Np, Nz, Nc, Ncv, Nm, N, K, t_meas, t_nodes, h, taus, L1, pscale, gate_vals, gate_vals_ss)

    # OBJECTIVE FUNCTION: Create auxiliary variables for model observables, y
    c = _create_y(c, PEmodel, PEinfo)
    
    # OBJECTIVE FUNCTION: Create auxiliary variables for measurement errors, sigma
    c = _create_sigma(c, PEinfo)

    # If there are condition-dependent variables...
    if Ncv >= 1
        c = _create_cv(c, PEmodel, PEprob, PEinfo)
    end

    # If initial conditions are steady-state equilibrium...
    if _check_x0SSpre(PEprob) 
        c = _create_zss(c, PEmodel, PEprob, PEinfo)
    end
    
    return c, PEinfo
end

# Creates ExaModels decision variables for unknown parameters
# p[1:Np]
function _create_p(c::ExaCore, PEprob::PEtabODEProblem)
    (; lower_bounds, upper_bounds, nparameters_estimate) = PEprob
    Np = nparameters_estimate  # number of unknown parameters to fit
    # p[1:Np] := θ, the actual decision variable for ExaModels.
    θ_LB   = Array(lower_bounds)        # estimation scale
    θ_UB   = Array(upper_bounds)        # estimation scale
    θ_init = Array(PEtab.get_x(PEprob)) # estimation-scale nominal (the ODE-solve point)
    @assert all(θ_LB .<= θ_init .<= θ_UB) "Nominal θ values fall outside estimation-scale bounds."
    ExaModels.@add_var(c,
        p,
        1:Np;
        lvar  = θ_LB,
        uvar  = θ_UB,
        start = θ_init
    )
    return c, Np
end

# Creates ExaModels decision variables for discretized states
# z[1:Nz,1:N,0:K,1:Nc]
function _create_z(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, K::Int)
    z_init, Nz, N, K, Nc, t_meas, h, taus, L1, t_nodes = _get_z_init(PEmodel, PEprob, K)
    ExaModels.@add_var(c,
        z,
        1:Nz, 1:N, 0:K, 1:Nc;
        start = z_init,
        lvar = -Inf,
        uvar = Inf
    )
    return c, Nz, N, K, Nc, t_meas, h, taus, L1, t_nodes
end

# Creates ExaModels decision variables for condition-dependent variables
# cv[1:Ncv,1:Nc]
function _create_cv(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    (; Np, Ncv, pscale) = PEinfo
    conditions_df  = PEmodel.petab_tables[:conditions]
    cv_names       = _get_cv_names(PEmodel)
    cv_rows        = _get_cv_cid_rows(PEmodel, PEprob)  # cv column => conditions-table row
    Ncc            = length(cv_rows)
    dict_pstr_pidx = _get_dict_pstr_pidx(PEprob)        # parameter name => p decision-var index
    θ0             = _var_starts(c, c.p)                # p (= θ, estimation scale) initial guesses
    p0             = [_p_phys_val(θ0, m, pscale) for m in 1:Np] # physical parameter starts (cv == linearized p value)

    # Only two possible paths: cv = fixed value or cv = p, so init with fixed val or p0
    cv_init = zeros(Float64, Ncv, Ncc)
    for cidx in 1:Ncc
        for cvidx in 1:Ncv
            val = conditions_df[cv_rows[cidx], cv_names[cvidx]]
            if val isa Number
                cv_init[cvidx, cidx] = Float64(val)
            elseif val isa String || val isa Symbol
                str_val    = String(val)
                parsed_val = tryparse(Float64, str_val)
                if parsed_val !== nothing
                    cv_init[cvidx, cidx] = parsed_val
                elseif haskey(dict_pstr_pidx, str_val)
                    cv_init[cvidx, cidx] = p0[dict_pstr_pidx[str_val]] # cv = p => use p's start
                else
                    error("Condition variable '$str_val' not found in unknown parameter list.")
                end
            end
        end
    end
    ExaModels.@add_var(c,
        cv,
        1:Ncv, 1:Ncc;
        start = cv_init
    )
    return c
end

# Creates ExaModels decision variables for steady-state state (used for x0SSpre and pure steady-state only models)
# zss[1:Nz,1:Nc]
function _create_zss(
        c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo;
        init = _get_zss_init(PEmodel, PEprob, PEinfo)
    )
    (; Nz, Nc) = PEinfo
    ExaModels.@add_var(c,
        zss,
        1:Nz, 1:Nc;
        start = init,
        lvar = -Inf,
        uvar = Inf
    )
    return c
end

# Creates ExaModels (auxiliary) decision variables for observable model variable
# y[1:Nm]
function _create_y(c::ExaCore, PEmodel::PEtabModel, PEinfo::PEInfo)
    (; Nm) = PEinfo
    # Impose nonnegativity for log transformed variables 
    transforms = _get_meas_transforms(PEmodel)
    y_LB = [t === :log || t === :log10 ? 0.0 : -Inf for t in transforms]
    ExaModels.@add_var(c,
        y,
        1:Nm;
        lvar = y_LB # set_start! added later
    )
    return c
end

# Creates ExaModels (auxiliary) decision variables for standard deviation of error
# sigma[1:Nm]
function _create_sigma(c::ExaCore, PEinfo::PEInfo)
    (; Nm) = PEinfo
    ExaModels.@add_var(c,
        sigma,
        1:Nm # set_start! added later
    )
    return c
end

