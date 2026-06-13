# (*) Main function for creating ExaModels decision variables (*)
function _create_variables(
        c::ExaCore,
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        K::Int
    )
    _assert_supported_events(PEmodel)   # reject true SBML <event> models (silently-wrong otherwise)
    # Create necessary variables (discretized state, unknown params) and obtain problem details (::PEInfo)
    c, Np = _create_p(c, PEprob)
    c, Nz, N, K, Nc, t_meas, h, taus, L1, t_nodes = _create_z(c, PEmodel, PEprob, K)

    Ncv = length(_get_cv_syms(PEmodel)) # number of condition-dependent variables
    Nm = length(eachrow(PEmodel.petab_tables[:measurements])) # number of data measurements
    Ny = length(_get_obsids(PEmodel)) # number of model observables
    pscale = _get_pscale(PEprob) # per-parameter estimation scale (:log10/:log/:lin), aligned 1:Np
    # Fixed-time event gates: keep piecewise(time) gates live and resolve their per-(interval,
    # condition) values from a stepped warm-start integrator (no-op / empty when the model has none).
    gate_syms = _get_gate_syms(PEprob)
    gate_vals, gate_vals_ss = _get_gate_vals(PEmodel, PEprob, gate_syms, h, taus, t_nodes)
    # gate_syms is NOT stored in PEInfo (recomputed at each consumer via _get_gate_syms, like
    # z_syms/cv_syms); only the value arrays — expensive + mesh-dependent — are cached.
    PEinfo = PEInfo(Np, Nz, Nc, Ncv, Nm, Ny, N, K, t_meas, t_nodes, h, taus, L1, pscale,
                    gate_vals, gate_vals_ss)

    # OBJECTIVE FUNCTION VARIABLES
    # Create auxiliary variables for model observables, y
    c = _create_y(c, PEmodel, PEinfo)
    # Create auxiliary variables for measurement errors, sigma
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
# The decision variable p := θ lives on PEtab's per-parameter ESTIMATION scale
# (log10/log/lin from the parameter file), matching how PEtab optimizes. This
# nondimensionalizes parameters spanning many orders of magnitude (e.g. a log10
# parameter on [1e-10, 1e3] becomes a θ-box [-10, 3]), which is essential for the
# IPM's conditioning. The physical parameter that enters the ODE RHS / observable /
# noise formulas is recovered on the fly via _p_phys (10^θ / exp(θ) / θ).
#
# start and bounds come straight from PEtab — they are ALREADY on the estimation
# scale (get_x, lower_bounds, upper_bounds), so NO transform is applied here.
function _create_p(c::ExaCore, PEprob::PEtabODEProblem)
    (; lower_bounds, upper_bounds, nparameters_estimate) = PEprob
    Np = nparameters_estimate  # number of unknown parameters to fit
    θ_LB   = Array(lower_bounds)        # estimation scale
    θ_UB   = Array(upper_bounds)        # estimation scale
    θ_init = Array(PEtab.get_x(PEprob)) # estimation-scale nominal (the ODE-solve point)
    @assert all(θ_LB .<= θ_init .<= θ_UB) "Nominal θ values fall outside estimation-scale bounds!"
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
        lvar = -Inf,    # unbounded: state sign/range is model-dependent (not assumed >= 0)
        uvar = Inf
    )
    return c, Nz, N, K, Nc, t_meas, h, taus, L1, t_nodes
end

# Creates ExaModels decision variables for condition-dependent variables
# cv[1:Ncv,1:Nc]
# Each cv entry is either a fixed numeric value or exactly equal to an unknown
# parameter (see the cv equality constraints in collocation.jl). We initialize the
# start accordingly — numeric literal, or the parameter's own initial guess — so cv
# (and any observable/noise formula that references it) is warm-started consistently.
function _create_cv(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    (; Np, Ncv, pscale) = PEinfo
    conditions_df  = PEmodel.petab_tables[:conditions]
    cv_cols        = _get_cv_names(PEmodel)          # cv column names, aligned 1:Ncv
    # cv columns span the simulation conditions (1:Nc) PLUS any distinct pre-equilibration
    # conditions (so the steady-state residual can read the pre-eq inputs); see _get_cv_cids.
    cv_rows        = _get_cv_cid_rows(PEmodel, PEprob)  # cv column => conditions-table row
    Ncc            = length(cv_rows)
    dict_pstr_pidx = _get_dict_pstr_pidx(PEprob)        # parameter name => p decision-var index
    θ0             = _var_starts(c, c.p)                # p (= θ, estimation scale) initial guesses
    # physical parameter starts (cv == p means cv equals the PHYSICAL parameter value)
    p0             = [_p_phys_val(θ0, m, pscale) for m in 1:Np]

    cv_init = zeros(Float64, Ncv, Ncc)
    for cidx in 1:Ncc
        for cvidx in 1:Ncv
            val = conditions_df[cv_rows[cidx], cv_cols[cvidx]]
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

# Creates ExaModels decision variables for steady-state state zss[1:Nz,1:Nc].
# Used by two paths: (a) x0SSpre pre-equilibration (default `init` = pre-equilibrated u0,
# via _get_zss_init) and (b) pure steady-state models, where zss IS the observed state and
# the caller passes a forward-simulated steady-state warm start (_get_zss_init_sim).
function _create_zss(
        c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo;
        init = _get_zss_init(PEmodel, PEprob, PEinfo)
    )
    (; Nz, Nc) = PEinfo
    ExaModels.@add_var(c,
        zss,
        1:Nz, 1:Nc;
        start = init,
        lvar = -Inf,    # unbounded: same state quantities as z (sign/range model-dependent)
        uvar = Inf
    )
    return c
end

# Creates ExaModels (auxiliary) decision variables for observable model variable
# y[1:Nm]
# For log/log10-transformed observables the NLL contains log(y[midx]), so those y
# need a positivity bound (the true observable is positive); the IPM barrier then
# keeps y strictly > 0. lin observables stay unbounded.
function _create_y(c::ExaCore, PEmodel::PEtabModel, PEinfo::PEInfo)
    (; Nm) = PEinfo
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
        1:Nm # bounds? set_start! added later
    )
    return c
end

