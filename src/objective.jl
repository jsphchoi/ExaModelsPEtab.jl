#######################################################
# STATE AS OF: 05/30/26
# TODO: verify implementation
# TODO: y0 initial guess just idx+1, if idx=N then use weighted sum.
#######################################################

# (*) Main function for creating ExaModels objective function (*)
function _create_objective(
        c::ExaCore,
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        PEinfo::PEInfo
    )
    ###############################################
    # Unpack problem info
    ###############################################
    (; Np, Ncv, Nz, Nc, Nm, Ny, N, K, t_meas, h, L1) = PEinfo
    z = c.z
    p = c.p
    y = c.y
    sigma = c.sigma
    if Ncv >= 1
        cv = c.cv
    end

    # ---- Warm-start support -------------------------------------------------
    # y and sigma are auxiliary variables defined entirely by z, p (and cv), which
    # already carry good PEtab initial guesses. We evaluate their defining formulas
    # at the initial point here so y/sigma can be given matching (feasible) starts
    # via set_start! after the model is built — critical for the IPM solver.
    z0  = reshape(_var_starts(c, z), Nz, N, K + 1, Nc) # z0[v, i, j+1, cidx]
    p0  = _var_starts(c, p)                            # p0[m]
    cv0 = Ncv >= 1 ? reshape(_var_starts(c, cv), Ncv, Nc) : zeros(Float64, 0, Nc)
    y0     = zeros(Float64, Nm) # computed observable values at the initial guess
    sigma0 = zeros(Float64, Nm) # computed noise (std) values at the initial guess

    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    measurements_df = PEtable[:measurements] # :observableId, :preequilibrationConditionId, :simulationConditionId, :measurement, :time, :observableParameters, :noiseParameters, :datasetId
    observables_df = PEtable[:observables] # :observableId, :observableName, :observableFormula, :noiseFormula, :observableTransformation, :noiseDistribution

    ###############################################
    # Objective function
    ###############################################
    # Create objective function (negative log-likelihood): min ∑ₘ(y - yₘ)²/σ²ₘ
    itr_obj = [(midx, Float64(measurements_df[midx, :measurement])) for midx in 1:Nm]
    ExaModels.@add_obj(c,
        (y[midx] - ymeas)^2/sigma[midx]
        for (midx, ymeas) in itr_obj
    )

    ###############################################
    # Auxiliary variable constraints for y, sigma
    ###############################################
    # Parsed table values => ExaModels variable index mappings
    dict_cid_cidx = _get_dict_cid_cidx(PEmodel)
    dict_t_tidx   = _get_dict_t_tidx(h, t_meas)

    # Substitute in fixed constant values
    dict_all_val = Dict(PEprob.model_info.model.parametermap)
    fixed_syms = setdiff(
        keys(Dict(dict_all_val)),
        union(_get_p_syms(PEprob), _get_cv_syms(PEmodel))
    )
    dict_fixed_val = Dict(sym => val for (sym,val) in dict_all_val if (sym in fixed_syms))

    # Symbolics of variables that may appear in parsed table formulas
    z_syms = [
        Symbolics.Num(Symbolics.variable(Symbol(split(string(z_sym), "(")[1])))
        for z_sym in _get_z_syms(PEprob)
    ]
    p_syms  = _get_p_syms(PEprob)
    cv_syms = _get_cv_syms(PEmodel)

    # Fast lookup: observableId => row in observables_df
    dict_obsid_obsrow = Dict(
        string(observables_df[i, :observableId]) => observables_df[i, :]
        for i in 1:size(observables_df, 1)
    )

    # Helper: normalize a raw table cell to String (handle missing/nothing)
    _safe_str(v) = (ismissing(v) || isnothing(v)) ? "" : strip(string(v))

    # Some PEtab models omit these columns entirely when no row uses them
    has_obs_params_col   = :observableParameters in propertynames(measurements_df)
    has_noise_params_col = :noiseParameters      in propertynames(measurements_df)

    ###############################################
    # Observable formula (y) constraints
    # Group measurements by (obsId, observableParameters) so that each unique
    # formula gets its own compiled ExaModels constraint.
    ###############################################
    itr_y_z    = Int[]
    itr_y_z!   = Tuple{Int, Int, Int, Int, Int, Float64}[]
    itr_y_z_ic = Tuple{Int, Int, Int}[]  # state observable at t=0 -> initial-condition node

    obs_y_groups = Dict{Tuple{String,String}, Vector{Int}}()
    for midx in 1:Nm
        row     = measurements_df[midx, :]
        obs_id  = string(row[:observableId])
        obs_key = has_obs_params_col ? _safe_str(row[:observableParameters]) : ""
        push!(get!(obs_y_groups, (obs_id, obs_key), Int[]), midx)
    end

    for ((obs_id, obs_params_str), group_midxs) in obs_y_groups
        obs_expr_raw = string(dict_obsid_obsrow[obs_id][:observableFormula])

        # Substitute observableParameter${n}_${obs_id} placeholders with the
        # n-th semicolon-delimited entry from this group's observableParameters.
        obs_expr_sub = obs_expr_raw
        if !isempty(obs_params_str)
            parts        = strip.(split(obs_params_str, ";"))
            replace_pairs = ["observableParameter$(n)_$(obs_id)" => parts[n] for n in eachindex(parts)]
            obs_expr_sub = replace(obs_expr_sub, replace_pairs...)
        end

        parsed = Meta.parse(obs_expr_sub)

        if parsed isa Symbol
            # Observable is a single state variable: y[midx] = state at the measurement time.
            obs_sym = Symbolics.Num(Symbolics.variable(parsed))
            zidx    = findfirst(x -> isequal(x, obs_sym), z_syms)
            for midx in group_midxs
                row  = measurements_df[midx, :]
                cid  = string(row[:simulationConditionId])
                time = Float64(row[:time])
                cidx = dict_cid_cidx[cid]
                idx  = dict_t_tidx[time]
                if idx == 0
                    # t = 0: state is the initial-condition node z[zidx,1,0,cidx] directly
                    # (the L1 interval-interpolation has no interval 0 to extrapolate).
                    push!(itr_y_z_ic, (midx, zidx, cidx))
                    y0[midx] = z0[zidx, 1, 1, cidx]
                else
                    # y[midx] = Σ_j L1[j+1] * z[zidx, idx, j, cidx] (τ=1 endpoint of interval idx)
                    push!(itr_y_z, midx)
                    append!(itr_y_z!, [(midx, zidx, idx, cidx, j, L1[j+1]) for j in 0:K])
                    y0[midx] = sum(L1[j+1] * z0[zidx, idx, j+1, cidx] for j in 0:K)
                end
            end
        else
            # Observable is an arbitrary expression: compile obs_func for this group
            obs_parsed_sym = Symbolics.parse_expr_to_symbolic(parsed, @__MODULE__)
            obs_expr_final = Symbolics.substitute(obs_parsed_sym, dict_fixed_val)
            obs_func = Symbolics.build_function(
                obs_expr_final,
                [z_syms; p_syms; cv_syms]...,
                expression = Val{false}
            )

            itr_y_func   = Tuple{Int, Int, Int}[]  # idx < N : state = z[·,idx+1,0,·] (one node)
            itr_y_func_N = Tuple{Int, Int}[]       # idx = N : state = L1 endpoint of interval N
            for midx in group_midxs
                row  = measurements_df[midx, :]
                cid  = string(row[:simulationConditionId])
                time = Float64(row[:time])
                cidx = dict_cid_cidx[cid]
                idx  = dict_t_tidx[time]
                # warm start: evaluate obs_func at the initial guess, mirroring the constraint
                if idx == N
                    push!(itr_y_func_N, (midx, cidx))
                    zatt = ntuple(v -> sum(L1[jj+1] * z0[v, N, jj+1, cidx] for jj in 0:K), Nz)
                else
                    push!(itr_y_func, (midx, idx, cidx))
                    zatt = ntuple(v -> z0[v, idx+1, 1, cidx], Nz)
                end
                y0[midx] = obs_func(
                    zatt...,
                    ntuple(m -> p0[m], Np)...,
                    ntuple(m -> cv0[m, cidx], Ncv)...
                )
            end

            # idx < N: single-node kernel (state = node 0 of the next interval, by continuity)
            if !isempty(itr_y_func)
                ExaModels.@add_con(c,
                    y[midx] - obs_func(
                        ntuple(v -> z[v,idx+1,0,cidx], Nz)...,
                        ntuple(m -> p[m], Np)...,
                        ntuple(m -> cv[m,cidx], Ncv)...
                    )
                    for (midx, idx, cidx) in itr_y_func
                )
            end
            # idx == N: final-time group; the L1 (τ=1) endpoint sum is inlined here only.
            if !isempty(itr_y_func_N)
                ExaModels.@add_con(c,
                    y[midx] - obs_func(
                        ntuple(v -> sum(L1[jj+1]*z[v,N,jj,cidx] for jj in 0:K), Nz)...,
                        ntuple(m -> p[m], Np)...,
                        ntuple(m -> cv[m,cidx], Ncv)...
                    )
                    for (midx, cidx) in itr_y_func_N
                )
            end
        end
    end

    ###############################################
    # Noise formula (sigma) constraints
    # Group measurements by (obsId, noiseParameters) so that each unique
    # formula gets its own compiled ExaModels constraint.
    ###############################################
    itr_sigma_fix = Tuple{Int, Float64}[]  # sigma = numeric literal
    itr_sigma_p   = Tuple{Int, Int}[]      # sigma = p[pidx]

    obs_sigma_groups = Dict{Tuple{String,String}, Vector{Int}}()
    for midx in 1:Nm
        row        = measurements_df[midx, :]
        obs_id     = string(row[:observableId])
        noise_key  = has_noise_params_col ? _safe_str(row[:noiseParameters]) : ""
        push!(get!(obs_sigma_groups, (obs_id, noise_key), Int[]), midx)
    end

    for ((obs_id, noise_params_str), group_midxs) in obs_sigma_groups
        sigma_expr_raw = string(dict_obsid_obsrow[obs_id][:noiseFormula])

        # Substitute noiseParameter${n}_${obs_id} placeholders.
        sigma_expr_sub = sigma_expr_raw
        if !isempty(noise_params_str)
            parts        = strip.(split(noise_params_str, ";"))
            replace_pairs = ["noiseParameter$(n)_$(obs_id)" => parts[n] for n in eachindex(parts)]
            sigma_expr_sub = replace(sigma_expr_raw, replace_pairs...)
        end

        # Case A: substituted formula is a numeric literal
        sigma_val = tryparse(Float64, strip(sigma_expr_sub))
        if sigma_val !== nothing
            for midx in group_midxs
                push!(itr_sigma_fix, (midx, sigma_val))
                sigma0[midx] = sigma_val # warm start
            end
            continue
        end

        # Parse as symbolic expression
        sigma_parsed     = Meta.parse(sigma_expr_sub)
        sigma_parsed_sym = sigma_parsed isa Symbol ?
            Symbolics.Num(Symbolics.variable(sigma_parsed)) :
            Symbolics.parse_expr_to_symbolic(sigma_parsed, @__MODULE__)
        sigma_expr_final = Symbolics.substitute(sigma_parsed_sym, dict_fixed_val)

        sigma_free   = Symbolics.get_variables(sigma_expr_final)
        sigma_p_vars = filter(v -> any(isequal(v, pv) for pv in p_syms), sigma_free)

        if length(sigma_p_vars) == 1 && isempty(filter(v -> any(isequal(v, zv) for zv in z_syms), sigma_free))
            # Case B: sigma = p[pidx]
            pidx = findfirst(pv -> isequal(pv, only(sigma_p_vars)), p_syms)
            for midx in group_midxs
                push!(itr_sigma_p, (midx, pidx))
                sigma0[midx] = p0[pidx] # warm start
            end
        else
            # Case C: sigma is a general expression — compile sigma_func for this group
            sigma_func = Symbolics.build_function(
                sigma_expr_final,
                [z_syms; p_syms; cv_syms]...,
                expression = Val{false}
            )

            itr_sigma_func   = Tuple{Int, Int, Int}[]  # idx < N : one node
            itr_sigma_func_N = Tuple{Int, Int}[]       # idx = N : L1 endpoint of interval N
            for midx in group_midxs
                row  = measurements_df[midx, :]
                cid  = string(row[:simulationConditionId])
                time = Float64(row[:time])
                cidx = dict_cid_cidx[cid]
                idx  = dict_t_tidx[time]
                # warm start: evaluate sigma_func at the initial guess, mirroring the constraint
                if idx == N
                    push!(itr_sigma_func_N, (midx, cidx))
                    zatt = ntuple(v -> sum(L1[jj+1] * z0[v, N, jj+1, cidx] for jj in 0:K), Nz)
                else
                    push!(itr_sigma_func, (midx, idx, cidx))
                    zatt = ntuple(v -> z0[v, idx+1, 1, cidx], Nz)
                end
                sigma0[midx] = sigma_func(
                    zatt...,
                    ntuple(m -> p0[m], Np)...,
                    ntuple(m -> cv0[m, cidx], Ncv)...
                )
            end

            # idx < N: single-node kernel
            if !isempty(itr_sigma_func)
                ExaModels.@add_con(c,
                    sigma[midx] - sigma_func(
                        ntuple(v -> z[v,idx+1,0,cidx], Nz)...,
                        ntuple(m -> p[m], Np)...,
                        ntuple(m -> cv[m,cidx], Ncv)...
                    )
                    for (midx, idx, cidx) in itr_sigma_func
                )
            end
            # idx == N: final-time group; L1 (τ=1) endpoint sum inlined here only.
            if !isempty(itr_sigma_func_N)
                ExaModels.@add_con(c,
                    sigma[midx] - sigma_func(
                        ntuple(v -> sum(L1[jj+1]*z[v,N,jj,cidx] for jj in 0:K), Nz)...,
                        ntuple(m -> p[m], Np)...,
                        ntuple(m -> cv[m,cidx], Ncv)...
                    )
                    for (midx, cidx) in itr_sigma_func_N
                )
            end
        end
    end

    ###############################################
    # Emit batched constraints for accumulated iterators
    ###############################################
    if !isempty(itr_y_z)
        # y[midx] = Σ_{j=0}^{K} L_j(1) * z[zidx, i, j, cidx]
        con_y_z = ExaModels.@add_con(c,
            -y[midx]
            for midx in itr_y_z
        )
        ExaModels.@add_con!(c,
            con_y_z,
            midx => L1j*z[v,i,j,cidx]
            for (midx, v, i, cidx, j, L1j) in itr_y_z!
        )
    end

    if !isempty(itr_y_z_ic)
        # t=0 state observables: y[midx] = z[zidx, 1, 0, cidx] (initial-condition node)
        ExaModels.@add_con(c,
            y[midx] - z[zidx,1,0,cidx]
            for (midx, zidx, cidx) in itr_y_z_ic
        )
    end

    if !isempty(itr_sigma_fix)
        ExaModels.@add_con(c,
            sigma[midx] - val
            for (midx, val) in itr_sigma_fix
        )
    end

    if !isempty(itr_sigma_p)
        ExaModels.@add_con(c,
            sigma[midx] - p[pidx]
            for (midx, pidx) in itr_sigma_p
        )
    end

    return c, y0, sigma0
end
