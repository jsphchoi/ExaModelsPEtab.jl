# !!! for steady-state models := measurement time = Inf
# Instead of collocation equations for the ODE RHS, set f(zss...) = 0

# Creates all of the constraints for pure steady-state models
function _create_constraints_ss(
        c::ExaCore, 
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        PEinfo::PEInfo
    )
    c = _create_cv_constraints(c, PEmodel, PEprob, PEinfo)
    W, b, keep_rows = _conservation_ss(c, PEmodel, PEprob, PEinfo)
    c = _create_residual_ss(c, PEmodel, PEprob, PEinfo, identity; keep_rows = keep_rows)
    c = _add_conservation_constraints(c, PEinfo, W, b)
    return c
end

# A model is (pure) steady-state iff every measurement time is inf. Error any mixed cases
function _is_steady_state(PEmodel::PEtabModel)::Bool
    times   = Float64.(PEmodel.petab_tables[:measurements][!, :time])
    any_inf = any(isinf, times)
    any_inf || return false
    all(isinf, times) || error(
        "ExaModelsPEtab: unsupported mix of finite-time and steady-state (time=Inf) measurements."
    )
    return true
end

# Returns the steady-state values for every simulation condition
# by solving the ODE to steady-state (t=1e8, as PEtab does) and taking final states
function _get_zss_init_sim(PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    (; Nz, Nc) = PEinfo
    p_nominal = get_x(PEprob)
    cids      = Symbol.(_get_cids(PEmodel))
    solver    = PEprob.probinfo.solver.solver
    abstol    = PEprob.probinfo.solver.abstol
    reltol    = PEprob.probinfo.solver.reltol

    # One forward solve per condition to t=1e8; the terminal state is the steady state
    zss0 = zeros(Float64, Nz, Nc)
    for (cidx, cid) in enumerate(cids)
        odesys, callbacks = get_odeproblem(p_nominal, PEprob; condition = cid)
        odesys = ODE.remake(odesys; tspan = (odesys.tspan[1], 1.0e8))
        sol = ODE.solve(
            odesys, solver;
            callback = callbacks, abstol = abstol, reltol = reltol
        )
        zss0[:, cidx] = sol.u[end][1:Nz]
    end
    return zss0
end

# Create decision variables for the steady-state state variables
# zss[:,cidx]
function _create_variables_ss(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem)
    _assert_supported_events(PEmodel) # reject true SBML <event> models

    # Create unknown parameter decision variables
    c, Np = _create_p(c, PEprob)

    # Get problem info
    Nz  = Int64(PEprob.model_info.nstates)              # number of state variables
    Nc  = length(_get_cids(PEmodel))                    # number of simulation conditions
    Ncv = length(_get_cv_syms(PEmodel))                 # number of condition-dependent variables
    Nm  = length(eachrow(PEmodel.petab_tables[:measurements]))  # number of measurements
    pscale = _get_pscale(PEprob)

    # Get steady-state gate values
    gate_syms = _get_gate_syms(PEprob)
    gate_vals_ss = _get_gate_vals_ss(PEmodel, PEprob)

    # Create PEinfo (no mesh on the steady-state path; t_meas is the time=Inf sentinel)
    PEinfo = PEInfo(Np, Nz, Nc, Ncv, Nm, [Inf],
        pscale, zeros(Float64, length(gate_syms), 0, Nc), gate_vals_ss
    )

    # OBJECTIVE FUNCTION: Create auxiliary variables for model observables, y
    c = _create_y(c, PEmodel, PEinfo)

    # OBJECTIVE FUNCTION: Create auxiliary variables for measurement errors, sigma
    c = _create_sigma(c, PEinfo)

    # If there are condition-dependent variables...
    if Ncv >= 1
        c = _create_cv(c, PEmodel, PEprob, PEinfo)
    end

    # Create steady-state variables
    c = _create_zss(c, PEmodel, PEprob, PEinfo; init = _get_zss_init_sim(PEmodel, PEprob, PEinfo))

    return c, PEinfo
end

# Creates objective observable y and noise sigma constraints for the steady-state model
# Returns feasible y/sigma initial guesses
function _create_objective_ss(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    (; Np, Ncv, Nz, Nc, Nm, pscale) = PEinfo
    p     = c.p
    y     = c.y
    sigma = c.sigma
    zss   = c.zss
    if Ncv >= 1
        cv = c.cv
    end

    # Initialize initial guess derivation for y, sigma
    zss0 = reshape(_var_starts(c, zss), Nz, Nc)
    θ0   = _var_starts(c, p)
    p0   = [_p_phys_val(θ0, m, pscale) for m in 1:Np]
    cv0  = Ncv >= 1 ? reshape(_var_starts(c, cv), Ncv, :) : zeros(Float64, 0, Nc)
    y0     = zeros(Float64, Nm)
    sigma0 = zeros(Float64, Nm)

    PEtable         = PEmodel.petab_tables
    measurements_df = PEtable[:measurements]
    observables_df  = PEtable[:observables]

    # Create objective function
    c = _add_nll_objective(c, PEmodel, PEinfo) # MLE
    c = _add_prior_objective(c, PEmodel, PEprob) # MAP

    # Parsed table values => ExaModels variable index mappings
    dict_cid_cidx = _get_dict_cid_cidx(PEmodel)

    # Substitute in fixed constant values
    dict_all_val = Dict(PEprob.model_info.model.parametermap)
    fixed_syms = setdiff(
        keys(Dict(dict_all_val)),
        union(_get_p_syms(PEprob), _get_cv_syms(PEmodel))
    )

    # Parametermap fixed values and table-only fixed parameter numeric values
    dict_fixed_val = merge(
        _get_table_fixed_vals(PEmodel, PEprob),
        Dict(sym => val for (sym, val) in dict_all_val if (sym in fixed_syms)),
    )

    apply_rules = _assignment_substitutor(PEprob; remove_t = true)

    z_syms = [
        Symbolics.Num(Symbolics.variable(Symbol(split(string(z_sym), "(")[1])))
        for z_sym in _get_z_syms(PEprob)
    ]
    p_syms  = _get_p_syms(PEprob)
    cv_syms = _get_cv_syms(PEmodel)

    dict_obsid_obsrow = Dict(
        string(observables_df[i, :observableId]) => observables_df[i, :]
        for i in 1:size(observables_df, 1)
    )
    _safe_str(v) = (ismissing(v) || isnothing(v)) ? "" : strip(string(v))
    has_obs_params_col   = :observableParameters in propertynames(measurements_df)
    has_noise_params_col = :noiseParameters      in propertynames(measurements_df)

    ###############################################
    # Observable (y) constraints with state read at zss[:,cidx]
    ###############################################
    itr_y_state = Tuple{Int, Int, Int}[]   # (midx, zidx, cidx): observable is a single state
    # Group measurements sharing an (observable, observableParameter override)
    obs_y_groups = Dict{Tuple{String,String}, Vector{Int}}()
    for midx in 1:Nm
        row     = measurements_df[midx, :]
        obs_id  = string(row[:observableId])
        obs_key = has_obs_params_col ? _safe_str(row[:observableParameters]) : ""
        push!(get!(obs_y_groups, (obs_id, obs_key), Int[]), midx)
    end

    # Per group: parse the observable formula and add y[midx] = obs(zss[:,cidx]) (single state ⇒ direct)
    for ((obs_id, obs_params_str), group_midxs) in obs_y_groups
        obs_expr_raw = string(dict_obsid_obsrow[obs_id][:observableFormula])
        obs_expr_sub = _sub_placeholders(obs_expr_raw, obs_params_str, "observable", obs_id)
        obs_sym      = apply_rules(_to_symbolic(obs_expr_sub))

        zidx = findfirst(x -> isequal(x, obs_sym), z_syms)  # observable is a single state?
        if zidx !== nothing
            for midx in group_midxs
                cidx = dict_cid_cidx[string(measurements_df[midx, :simulationConditionId])]
                push!(itr_y_state, (midx, zidx, cidx))
                y0[midx] = zss0[zidx, cidx]
            end
        else
            obs_expr_final = Symbolics.substitute(obs_sym, dict_fixed_val)
            obs_func = Symbolics.build_function(
                obs_expr_final,
                [z_syms; p_syms; cv_syms]...,
                expression = Val{false}
            )
            itr_y_func = Tuple{Int, Int}[]   # (midx, cidx)
            for midx in group_midxs
                cidx = dict_cid_cidx[string(measurements_df[midx, :simulationConditionId])]
                push!(itr_y_func, (midx, cidx))
                y0[midx] = obs_func(
                    ntuple(v -> zss0[v, cidx], Nz)...,
                    ntuple(m -> p0[m], Np)...,
                    ntuple(m -> cv0[m, cidx], Ncv)...
                )
            end
            if !isempty(itr_y_func)
                ExaModels.@add_con(c,
                    y[midx] - obs_func(
                        ntuple(v -> zss[v,cidx], Nz)...,
                        ntuple(m -> _p_phys(p,m,pscale), Np)...,
                        ntuple(m -> cv[m,cidx], Ncv)...
                    )
                    for (midx, cidx) in itr_y_func
                )
            end
        end
    end
    if !isempty(itr_y_state)
        ExaModels.@add_con(c,
            y[midx] - zss[zidx,cidx]
            for (midx, zidx, cidx) in itr_y_state
        )
    end

    ###############################################
    # Noise (sigma) constraints
    ###############################################
    itr_sigma_fix = Tuple{Int, Float64}[]  # sigma = numeric literal
    itr_sigma_p   = Tuple{Int, Int}[]      # sigma = p[pidx]

    # Group measurements sharing an (observable, noiseParameter override)
    obs_sigma_groups = Dict{Tuple{String,String}, Vector{Int}}()
    for midx in 1:Nm
        row        = measurements_df[midx, :]
        obs_id     = string(row[:observableId])
        noise_key  = has_noise_params_col ? _safe_str(row[:noiseParameters]) : ""
        push!(get!(obs_sigma_groups, (obs_id, noise_key), Int[]), midx)
    end

    Y_sym = Symbolics.Num(Symbolics.variable(:__sigma_obs_Y__))
    dict_obsid_obssym = Dict{String, Any}()

    # sigma is \in {numeric value, p[pidx], f(observable y), f(z...)}
    for ((obs_id, noise_params_str), group_midxs) in obs_sigma_groups
        sigma_expr_raw = string(dict_obsid_obsrow[obs_id][:noiseFormula])
        sigma_expr_sub = _sub_placeholders(sigma_expr_raw, noise_params_str, "noise", obs_id)

        # sigma is a numeric value
        sigma_val = tryparse(Float64, strip(sigma_expr_sub))
        if sigma_val !== nothing
            for midx in group_midxs
                push!(itr_sigma_fix, (midx, sigma_val))
                sigma0[midx] = sigma_val
            end
            continue
        end

        sigma_parsed_sym = apply_rules(_to_symbolic(sigma_expr_sub))
        sigma_expr_final = Symbolics.substitute(sigma_parsed_sym, dict_fixed_val)

        sigma_free   = Symbolics.get_variables(sigma_expr_final)
        sigma_p_vars = filter(v -> any(isequal(v, pv) for pv in p_syms), sigma_free)

        # sigma is an unknown parameter
        if length(sigma_p_vars) == 1 && isempty(filter(v -> any(isequal(v, zv) for zv in z_syms), sigma_free))
            pidx = findfirst(pv -> isequal(pv, only(sigma_p_vars)), p_syms)
            for midx in group_midxs
                push!(itr_sigma_p, (midx, pidx))
                sigma0[midx] = p0[pidx]
            end
        else
            # sigma can only be a function of the observable y, so it is indirectly state-dependent
            obs_sym = get!(dict_obsid_obssym, obs_id) do
                obs_raw = string(dict_obsid_obsrow[obs_id][:observableFormula])
                Symbolics.substitute(apply_rules(_to_symbolic(obs_raw)), dict_fixed_val)
            end
            sigma_reduced, reduced_ok = _reduce_sigma_to_obs(sigma_expr_final, obs_sym, Y_sym, z_syms)
            allowed    = [Y_sym; p_syms; cv_syms]
            conforming = reduced_ok && all(
                rv -> any(isequal(rv, a) for a in allowed),
                Symbolics.get_variables(sigma_reduced)
            )

            if conforming
                sigma_fun = Symbolics.build_function(
                    sigma_reduced,
                    [Y_sym; p_syms; cv_syms]...,
                    expression = Val{false}
                )
                itr_sigma_obs = Tuple{Int, Int}[]  # (midx, cidx)
                for midx in group_midxs
                    cidx = dict_cid_cidx[string(measurements_df[midx, :simulationConditionId])]
                    push!(itr_sigma_obs, (midx, cidx))
                    sigma0[midx] = sigma_fun(y0[midx], p0..., cv0[:, cidx]...)
                end
                ExaModels.@add_con(c,
                    sigma[midx] - sigma_fun(
                        y[midx],
                        ntuple(m -> _p_phys(p,m,pscale), Np)...,
                        ntuple(m -> cv[m,cidx], Ncv)...
                    )
                    for (midx, cidx) in itr_sigma_obs
                )
            else
                @warn "Noise model for observable '$obs_id' references model states outside " *
                      "the observable formula; falling back to a state-dependent expression " *
                      "evaluated at the steady state."
                sigma_func = Symbolics.build_function(
                    sigma_expr_final,
                    [z_syms; p_syms; cv_syms]...,
                    expression = Val{false}
                )
                itr_sigma_func = Tuple{Int, Int}[]  # (midx, cidx)
                for midx in group_midxs
                    cidx = dict_cid_cidx[string(measurements_df[midx, :simulationConditionId])]
                    push!(itr_sigma_func, (midx, cidx))
                    sigma0[midx] = sigma_func(
                        ntuple(v -> zss0[v, cidx], Nz)...,
                        ntuple(m -> p0[m], Np)...,
                        ntuple(m -> cv0[m, cidx], Ncv)...
                    )
                end
                if !isempty(itr_sigma_func)
                    ExaModels.@add_con(c,
                        sigma[midx] - sigma_func(
                            ntuple(v -> zss[v,cidx], Nz)...,
                            ntuple(m -> _p_phys(p,m,pscale), Np)...,
                            ntuple(m -> cv[m,cidx], Ncv)...
                        )
                        for (midx, cidx) in itr_sigma_func
                    )
                end
            end
        end
    end

    # Create sigma auxiliary variable constraints for sigma = fixed val, sigma = p[pidx]
    if !isempty(itr_sigma_fix)
        ExaModels.@add_con(c,
            sigma[midx] - val
            for (midx, val) in itr_sigma_fix
        )
    end
    if !isempty(itr_sigma_p)
        for sc in (:log10, :log, :lin)
            grp = [t for t in itr_sigma_p if pscale[t[2]] === sc]
            isempty(grp) && continue
            if sc === :log10
                ExaModels.@add_con(c, sigma[midx] - exp(log(10.0)*p[pidx]) for (midx,pidx) in grp)
            elseif sc === :log
                ExaModels.@add_con(c, sigma[midx] - exp(p[pidx])           for (midx,pidx) in grp)
            else
                ExaModels.@add_con(c, sigma[midx] - p[pidx]                for (midx,pidx) in grp)
            end
        end
    end

    return c, y0, sigma0
end

# Detects conservation laws to remove additional constraint to make f(zss) = 0 sqaure wrt zss
# left null space of jacobian RHS is the conserved species
function _conservation_ss(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    (; Nz, Nc, Np, Ncv, pscale, gate_vals_ss) = PEinfo
    zss0 = reshape(_var_starts(c, c.zss), Nz, Nc)
    cv0  = Ncv >= 1 ? reshape(_var_starts(c, c.cv), Ncv, :) : zeros(Float64, 0, Nc)
    θ    = get_x(PEprob)
    p0   = [_p_phys_val(θ, m, pscale) for m in 1:Np]

    # Derive initial guess for the ss variables
    gss = size(gate_vals_ss, 2) >= 1 ? gate_vals_ss[:, 1] : Float64[]
    fs  = _get_rhs_funcs(PEmodel, PEprob)
    cvc = Ncv >= 1 ? cv0[:, 1] : Float64[]
    Fz  = z -> Float64[f(z..., p0..., cvc..., gss..., 0.0) for f in fs]

    # Finite difference approx of jacobian at initial states (initial guess satisfies conservation law)
    z1 = zss0[:, 1]
    F0 = Fz(z1)
    J  = zeros(Float64, Nz, Nz)
    for j in 1:Nz
        hj = 1.0e-7 * (abs(z1[j]) + 1.0)
        zp = copy(z1); zp[j] += hj
        J[:, j] = (Fz(zp) .- F0) ./ hj
    end

    # Calculate left null space of J
    Fsvd = LinearAlgebra.svd(J)
    smax = isempty(Fsvd.S) ? 0.0 : Fsvd.S[1]
    tol  = 1.0e-7 * max(smax, 1.0)
    null_idx = findall(s -> s < tol, Fsvd.S)
    r = length(null_idx)
    r == 0 && return (zeros(Float64, 0, Nz), zeros(Float64, 0, Nc), collect(1:Nz))

    W = Matrix(transpose(Fsvd.U[:, null_idx]))   # r×Nz (rows are left-null / conservation vectors)

    # Drop redundant equations from conservation law
    drop_rows = sort(LinearAlgebra.qr(W, LinearAlgebra.ColumnNorm()).p[1:r])
    keep_rows = setdiff(1:Nz, drop_rows)

    # Make sure that our finite diff on J did not depend on initial theta guess
    u0s, ic_theta_dep = _initial_conditions_ss(PEmodel, PEprob, PEinfo)
    ic_theta_dep && error(
        "ExaModelsPEtab: unsupported parameter-dependent conserved total in steady-state conservation law."
    )
    b = zeros(Float64, r, Nc)
    for cidx in 1:Nc, k in 1:r
        b[k, cidx] = LinearAlgebra.dot(view(W, k, :), view(u0s, :, cidx))
    end
    return (W, b, keep_rows)
end

# Get the initial conditions for the steady-state model
function _initial_conditions_ss(PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    (; Nz, Nc) = PEinfo
    cids = Symbol.(_get_cids(PEmodel))
    θ    = get_x(PEprob)
    # Baseline u0 per condition at nominal theta
    u0s  = zeros(Float64, Nz, Nc)
    for (cidx, cid) in enumerate(cids)
        oprob, _ = get_odeproblem(θ, PEprob; condition = cid)
        u0s[:, cidx] = oprob.u0[1:Nz]
    end
    # Check for theta dependence of initial condition
    θ2 = θ .+ 0.1
    ic_theta_dep = false
    for (cidx, cid) in enumerate(cids)
        oprob, _ = get_odeproblem(θ2, PEprob; condition = cid)
        if maximum(abs, oprob.u0[1:Nz] .- view(u0s, :, cidx)) > 1.0e-8 * (1.0 + maximum(abs, view(u0s, :, cidx)))
            ic_theta_dep = true
            break
        end
    end
    return u0s, ic_theta_dep
end

# Creates the conservation law constraints
function _add_conservation_constraints(c::ExaCore, PEinfo::PEInfo, W::AbstractMatrix, b::AbstractMatrix)
    r = size(W, 1)
    r == 0 && return c
    (; Nz, Nc) = PEinfo
    zss = c.zss

    # Ax=b
    itr_base = [(k, cidx, b[k, cidx]) for cidx in 1:Nc for k in 1:r]
    con = ExaModels.@add_con(c, 
        -bval 
        for (k, cidx, bval) in itr_base
    )
    itr_aug = [
        (pos, W[k, v], v, cidx)
        for (pos, (k, cidx, bval)) in enumerate(itr_base) for v in 1:Nz if W[k, v] != 0.0
    ]
    ExaModels.@add_con!(c, con, 
        pos => Wkv*zss[v, cidx]
        for (pos, Wkv, v, cidx) in itr_aug
    )
    return c
end